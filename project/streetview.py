# streetview.py
#
# This module handles all functions related to Google StreetView imagery
# including grabbing images from either the cache or the StreetView api and
# calculating a similarity score between an image and a StreetView image at the
# given coordinates

import urllib
import os
import cv2
import math
import numpy as np

# Given a latitude longitude and heading, retrieve a StreetView image either
# from the cache or the StreetView api
def getStreetView(lat, lon, heading):
    fileName = 'data/streetview/' + str(lat) + str(lon) + str(heading) + '.jpg'

    # Check if requested is cached and if not request the image from the API
    if not os.path.isfile(fileName):
        # Generate the request URL
        base = "https://maps.googleapis.com/maps/api/streetview?size=640x480"
        key = "key=AIzaSyAbtCt0WgqiCMe6cWRgB4q5eLuZcea7Yk0"
        location = "location=" + str(lat) + "," + str(lon)
        heading = "heading=" + str(math.degrees(heading))
        s = "&"
        seq = [base, location, heading, key]
        url = s.join(seq)

        # Retrieve and save the image
        urllib.urlretrieve(url, fileName)

    # Open and return the image
    return cv2.imread(fileName, 0)

# Given an image, coordinates, and a heading, calculate the similarity of the
# image to the StreetView image at those coordinates
def streetViewSimilarity(img,lat,lon,heading):
    # Grab the StreetView image
    sv_img = getStreetView(lat, lon, heading)

    # Initialize the SIFT object
    sift = cv2.SIFT()

    # Compute keypoints and descriptors for observered image
    kp, des = sift.detectAndCompute(img,None)

    # Compute keypoints and descriptors for StreetView image
    sv_kp, sv_des = sift.detectAndCompute(sv_img,None)

    # Brute force match keypoints in the images
    bf = cv2.BFMatcher()
    matches = bf.match(des,sv_des)

    # Calulate the sum of the euclidean distances between descriptors in the images
    totalDistance = 0
    for match in matches:
        totalDistance += match.distance

    # Return the average of distances between descriptors
    return totalDistance / float(len(matches))
