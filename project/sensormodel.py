# sensormodel.py
#
# This module constructs and saves a sensor model by fitting a gaussian
# probability function to a large batch of similarity scores between observed
# images from a training set and their StreetView counterparts using Kernel
# density estimation

from sklearn.neighbors import KernelDensity
import cv2
import streetview
import os
import math
import joblib
import numpy as np

# Given steps and quantity of frames to be processed return a list of similarity
# scores
def calcSimilarities(step,frames):
    # Load image metadata
    d = np.load('data/clean/traindata.npy').item()

    similarities = []
    # Iterate through images in train dataset directory
    for image in os.listdir('data/raw/data_collection_20100901/imgs')[::step][:frames]:
        if image.endswith(".jpg"):
            img = cv2.imread(('data/raw/data_collection_20100901/imgs/' + image),0)

            # Grab latitude, longitude, and heading for the image
            info = d[image]
            lat = info[1]
            lon = info[2]
            heading = info[3]

            # Calculate angle of camera by checking which camera took the image
            # and making the necessary adjustment
            camHeading = 0
            split = image.split("_")
            if split[2] == "c0":
                camHeading = (math.degrees(heading) - 45) % 360
            else:
                camHeading = (math.degrees(heading) + 45) % 360

            # Calculate similarity and add it to list of similarities
            similarity = streetview.streetViewSimilarity(img, lat, lon, camHeading)
            similarities.append(similarity)

    return similarities

def fitData(similarities):
    # Convert similarity list into array
    X = np.array(similarities).reshape(-1,1)

    # Initialize kernel density model
    kde = KernelDensity()

    # Fit model to data
    kde.fit(X)

    # Save data to model directory
    joblib.dump(kde, 'model/sensormodel.pkl')

# Test model by scoring a test data point and sampling
def test():
    kde = joblib.load('model/sensormodel.pkl')
    img = cv2.imread('tester.jpg', 0)
    sample = 300
    np_sample = np.array(sample)
    np_sample = np_sample.reshape(-1,1)
    score = kde.score(np_sample)
    print math.exp(score)
    print kde.sample(100)

sim = calcSimilarities(10, 1000)
fitData(sim)
