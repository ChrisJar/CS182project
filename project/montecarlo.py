# # montecarlo.py
#
# This module contains the localization module object that implements the Monte
# Carlo Localization algorithm

import numpy as np
import cv2
import random
import geopy
import joblib
import math
from streetview import streetViewSimilarity
from sklearn import preprocessing

class LocalizationModule:
    """
    A localization module that approximately tracks a car given dash camera footage.
    """

    # Initialize the localization module
    def __init__(self, numParticles, initialTime):
        # Set the number of particles
        self.numParticles = numParticles

        # Load the sensor model from memory
        self.sensorModel = joblib.load('model/sensormodel.pkl')

        # Set the time to initialTimeStamp
        self.time = initialTime

    # Initialize the samples from a uniform distribution
    def initializeUniformly(self, maxLat, minLat, maxLon, minLon):

        # Sample particles randomly
        particles = []
        for _ in range(self.numParticles):
            lat = random.uniform(maxLat,minLat)
            lon = random.uniform(maxLon,minLon)
            particles.append((lat,lon))
        self.maxLat = maxLat
        self.minLat = minLat
        self.maxLon = maxLon
        self.minLon = minLon
        self.particles = particles

    # Perform the sensor update step and resample particles based on calculated
    # weights
    def observe(self):
        sensorModel = self.sensorModel

        # Iterate through particles and calculate weights using the sensor model
        weights = []
        for particle in self.particles:

            # Open the observed image
            img = cv2.imread(('data/raw/data_collection_20100915/imgs/' + self.image),0)

            lat = particle[0]
            lon = particle[1]
            heading = self.heading

            # Calculate camera heading by determinng which of the two cameras
            # captured the image and making the necessary adjustments
            camHeading = 0
            split = self.image.split("_")
            if split[2] == "c0":
                camHeading = (math.degrees(heading) - 45) % 360
            else:
                camHeading = (math.degrees(heading) + 45) % 360

            # Calculate the similarity
            similarity = streetViewSimilarity(img, lat, lon, camHeading)

            # Convert the similarity score to a probability using the trained
            # probability density function
            score = sensorModel.score(np.array([similarity]).reshape(-1,1))

            # Convert the log probability to raw probability
            weight = math.exp(score)

            # Add the probability to the list of weights
            weights.append(weight)


        # Normalize the weights
        totalWeight = sum(weights)
        if totalWeight == 0:
            self.initializeUniformly(self.maxLat, self.minLat, self.maxLon, self.minLon)
            return

        normalizedWeights = []
        for weight in weights:
            normalizedWeights.append(weight/totalWeight)

        # np_weights = np.array(weights).reshape(1,-1)
        # normalizedWeights = preprocessing.normalize(np_weights).tolist()
        # normalizedWeights = normalizedWeights[0]

        # Generate new particles by sampling current particles with a
        # probability proportional to the weights
        tempParticles = []
        for _ in range(self.numParticles):

            # Sample a particle with a probability proportional to the weights
            newParticleIndex = np.random.choice(range(self.numParticles), p=normalizedWeights)
            newParticle = self.particles[newParticleIndex]
            tempParticles.append(newParticle)

        # Set the particles to the netwly sampled particles
        self.particles = tempParticles

    # Move the particles based on the current speed, heading, and time elapsed
    # since the last move
    def move(self):

        # Iterate through the particles and calculate the new location of each
        tempParticles = []
        for particle in self.particles:

            # Generate a noisey heading reading
            headingNoise = np.random.normal(0,0.1)
            b = self.heading + headingNoise

            # Generate a noisey distance calculation
            translationNoise = np.random.normal(0,0.0001)
            d = (self.speed * self.elapsed * 0.001) + translationNoise

            # Use current position as well as noisey heading and distance to
            # calculate a new position
            origin = geopy.Point(particle[0],particle[1])
            destination = geopy.distance.VincentyDistance(kilometers=d).destination(origin, b)

            # Update the position
            lat, lon = destination.latitude, destination.longitude
            tempParticles.append((lat,lon))

        # Set the particles with the updated positions
        self.particles = tempParticles

    # Update the current observation
    def step(self, image, timestamp, heading, speed):

        # Set time elapsed since last observation
        self.elapsed = timestamp - self.time

        # Set speed, heading, the timestamp of the image
        self.speed = speed
        self.heading = heading
        self.time = timestamp
        self.image = image

        # Perform the motion update
        self.move()

        # Perform the sensor update
        self.observe()

    # Estimate the position by calculating the average of the current samples
    def estimate(self):

        # Iterate through particles and sum latitudes and longitudes
        latSum, lonSum = 0, 0
        for particle in self.particles:
            latSum += particle[0]
            lonSum += particle[1]

        # Calculate average latitude and longitude
        avgLat = latSum / self.numParticles
        avgLon = lonSum / self.numParticles

        return (avgLat,avgLon)
