import numpy as np
import pandas as pd
import random
import os
from numpy import array


def getBlWhImage(image):
    # Turns the image into black and white
    pixels = []
    for x in range(28):
        for y in range(28):
            if (int(image[x][y]) > 128):
                pixels.append(1)
            else:
                pixels.append(0)
    return np.reshape(pixels, (28, 28))


def verticalIntersections(image):
    # Gets the number of intersections in black and white image
    counts = []
    prev = 0
    for y in range(28):
        count = 0
        for x in range(28):
            current = int(image[x][y])
            if (prev != current):
                count += 1
            prev = current
        counts.append(count)
    average = sum(counts)/28
    maximum = max(counts)
    return average, maximum


def calculateDensity(image):
    # calculates the density
    count = 0
    for x in range(28):
        for y in range(28):
            count = count + int(image[x][y])
    return count/(28*28)


def open_images(path):
    # Opens the csv files and extracts the images from them and returns them
    images = []
    data = pd.read_csv(path)
    headers = data.columns.values

    labels = data[headers[0]]
    labels = labels.values.tolist()

    pixels = data.drop(headers[0], axis=1)

    for i in range(0, data.shape[0]):
        row = pixels.iloc[i].to_numpy()
        grid = np.reshape(row, (28, 28))
        images.append(grid)
    return labels, images
