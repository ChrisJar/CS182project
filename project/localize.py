# localize.py
#
# This is the central model and it orchestrates the entire localization process
# by unpackaging data from the dataset and feeding into the Monte Carlo
# Localization object

import montecarlo
import os
import joblib
import math
import gmplot
import numpy as np
import sys
import matplotlib.pyplot as plt
from geopy.distance import distance

# Given a list of images and a particle density run the localization algorithm
def localize(images, particles):

    # Load the data
    d = np.load('data/clean/testdata.npy').item()
    print "Dataset loaded"

    # Create a list of the data just from images given
    valList = []
    for image in images:
        valList.append(d[image])

    # Calculate the maximum and minimum latitude and longitudes
    maxLat = max(valList,key=lambda item:item[1])[1]
    minLat = min(valList,key=lambda item:item[1])[1]
    maxLon = max(valList,key=lambda item:item[2])[2]
    minLon = min(valList,key=lambda item:item[2])[2]

    # Calculate the total area
    b1 = distance((maxLat, minLon), (maxLat, maxLon)).meters
    b2 = distance((minLat, minLon), (minLat, maxLon)).meters
    s = distance((maxLat, minLon), (minLat, minLon)).meters
    h = math.sqrt(s ** 2 - ((b2 - b1) / 2) ** 2)
    area = h * (b1 + b2) / 2

    # Calculate the number of particles needed
    n = int(area / particles)

    # Find the starting timestamp
    t = d[images[0]][0]

    # Loac the localization module
    mc = montecarlo.LocalizationModule(n, t)
    print "Localization module loaded"

    # Initialize the map
    mc.initializeUniformly(maxLat,minLat,maxLon,minLon)
    print str(n) + " samples initialized"

    # Iterate through the images and feed them into the MCL module
    targets = []
    predictions = []
    errors = []
    for image in images:

        # Pull then images metadata
        info = d[image]

        # Complete a step
        mc.step(image, info[0], info[3], info[4])
        print "Step Completed"

        # Add the expected location at the step to the list of expected locations
        target = (info[1],info[2])
        targets.append(target)

        # Add the predicted location at the step to a list of predicted locations
        prediction = mc.estimate()
        predictions.append(prediction)

        # Calculate the error for the step
        errors.append(distance(target,prediction).meters)

    return (predictions, targets, errors)

# Given a step, number of frames, and particle density, orchestrate the
# localization process and generate plots
def run(step,frames,particles):

    # Iterate through images and grab the correct number of frames with the
    # correct step
    images = []
    for image in os.listdir('data/raw/data_collection_20100915/imgs')[10000:][::step][:frames]:
        if image.endswith(".jpg"):
            images.append(image)

    print len(images)

    # Run the localization process
    predictions, targets, errors = localize(images, particles)

    # Plot the errors
    plt.plot(range(1, 1 + len(errors)),errors)
    plt.ylabel('Error (m)')
    plt.xlabel('Frame')
    plt.show
    plt.savefig('errors.png')

    # Plot the predicted coordinates and target coordinates on a map
    index = int(len(targets) / 2)
    coordinates = targets[index]

    targetLats = [i[0] for i in targets]
    targetLongs = [i[1] for i in targets]
    predictedLats = [i[0] for i in predictions]
    predictedLongs = [i[1] for i in predictions]

    gmap1 = gmplot.GoogleMapPlotter(coordinates[0],coordinates[1],16)
    gmap2 = gmplot.GoogleMapPlotter(coordinates[0],coordinates[1],16)

    gmap1.plot(targetLats, targetLongs, 'red')
    gmap2.plot(predictedLats, predictedLongs, 'red')

    gmap1.draw("targets.html")
    gmap2.draw("predictions.html")

    # Save the results
    joblib.dump((predictions, targets, errors), 'data/results/' + str(step) + '_' + str(frames) + '_' + str(particles) + '.pkl')

# Run preset run
def presetRun():
    return run(5, 50, 36)

if len(sys.argv) == 4:
    run(sys.argv[1],sys.argv[2],sys.argv[3])
else:
    presetRun()
