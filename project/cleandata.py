# cleandata
#
# This module provides functions that reformat data from the raw timestamp based
# datasets into a format that associates each image with its relevant metadata

import os
import numpy as np

# Given a dataset clean the data
def getData(dataset):
    # Open the GPS data file
    gps = open((dataset + '/GPS.txt'))

    # iterate through the GPS data
    timeData = []
    for i,line in enumerate(gps):
        if i == 0:
            continue

        # Pull out timestamp, coordinates, heading, and speed
        entries = line.split()
        timestamp = float(entries[1])
        lat = float(entries[2])
        lon = float(entries[3])
        heading = float(entries[13])
        speed = float(entries[14])

        # Add data to the timeData list
        timeData.append((timestamp,lat,lon,heading,speed))

    gps.close()

    # Iterate through images and find the associated metadata at the time the
    # image was taken
    data = {}
    images = os.listdir((dataset + '/imgs'))
    for image in images:
        if image.endswith(".jpg"):
            split = image.split("_")
            # timestamp = float(split[3][:-6]) / 1000000
            #
            # for entry in timeData:
            #     data[image] = (timestamp, entry[1], entry[2], entry[3], entry[4])
            #     if entry[0] >= timestamp:
            #         break
            frame = float(split[1])
            scale = len(timeData) / float(len(images))
            data[image] = timeData[int(scale * frame)]

    return data

# Save the test data in the clean directory
def saveTestData():
    d = getData('data/raw/data_collection_20100915')
    np.save('data/clean/testdata.npy', d)

# Save the training data in the celean directory
def saveTrainData():
    d = getData('data/raw/data_collection_20100901')
    np.save('data/clean/traindata.npy', d)

saveTestData()
saveTrainData()
