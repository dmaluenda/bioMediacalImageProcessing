#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:27:09 2020

@author: dMaluenda
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt


def getDynamicRange(image):
    """ Returns the minimum and the maximum of an image.
    """
    return image.min(), image.max()


def getImageType(image):
    """ Returns the pixels' type of an image.
    """
    return type(image.flat[0])


def printImageInfo(image):
    """ Prints the basic info of an image.
    """
    minDR, maxDR = getDynamicRange(image)
    print(f"\nImage type: {getImageType(image)} [{minDR}, {maxDR}]")
    print("The size of the image is: {}x{}x{}".format(*image.shape))


def getChannels(image, doSeparatelly=False):
    """ Returns the channels of an image.
        If doSeparatelly=False returns a list of channels
        If doSeparatelly=True returns the channels in different variables
    """
    chanels = tuple() if doSeparatelly else []
    for chIdx in range(image.shape[2]):
        chanel = image[:, :, chIdx]
        if doSeparatelly:
            chanels += (chanel, )  # append in tuples
        else:
            chanels.append(chanel)
    return chanels


def getLuminance(image):
    """ Returns a matrix with the luminance of an image.
    """
    R, G, B = getChannels(image, doSeparatelly=True)
    lum = 0.299*R + 0.587*G + 0.114*B
    dtype = 'uint8' if isUint8(image) else 'float64'
    return np.array(lum, dtype=dtype)


def getAverage(image):
    """ Returns a matrix with the average of the image's channels.
    """
    chanels = getChannels(image)
    avg = sum([ch/len(chanels) for ch in chanels])
    
    return np.array(avg, dtype='uint8' if isUint8(image) else 'float64')


def im2float(image):
    """ This function returns an image (npArray) of floats.
    """
    return np.array(image/image.max(), dtype='float64')


def isUint8(image):
    """ Returns True if image is made of uint8.
    """
    return isinstance(image.flat[0], np.uint8)


def gammaFilter(image, gamma=1):
    """ Applies a gamma filter to an image.
    """
    if isUint8(image):
        image = im2float(image)
    return image**gamma


def importEyes():
    """ Returns two true-color images.
    """
    eye1 = plt.imread('eye1.jpg')
    eye2 = plt.imread('eye2.jpg')
    
    return eye1, eye2


def checkIsOneChannel(image):
    """ Do nothing if image is just a matrix.
        Raise and error when image has more than one channel.
    """
    if len(image.shape) > 2 and image.shape[2] > 1:
        raise Exception("Plain images (matrix) only allowed. "
                        "Consider to get Luminance or use just one channel.")


def thresholding(image, threshold):
    """ Returns a binarized image according to a certain threshold.
        Threshold can be a scalar or a matrix of the same shape of image
    """
    checkIsOneChannel(image)
    if isUint8(image):
        im_thresholded = 255 * np.uint8(image > threshold)
    else:
        im_thresholded = np.float64(image > threshold)

    return im_thresholded


def medianFilter(image, kernel=9):
    """ Returns a binarized image by means of applying a median filter.
        Kernel is the size of the window to find the median value.
    """
    checkIsOneChannel(image)
    mFilter = medfilt(image, kernel_size=kernel)
    return thresholding(image, mFilter), mFilter


def getNewPixel(pixel, newDynamicRange):
    """ Takes a pixel value and returns the closer value
        according to the newDynamicRange.
    """
    return (int(pixel * newDynamicRange) / (newDynamicRange - 1)
            if pixel < 1 else 1)


# getSegmentedImage returns a one channel image with
#  of a certain dynamic range (see getNewPixel)
getSegmentedImage = np.vectorize(getNewPixel)


def dithering(imageIN, matrix, newDynamicRange=2):
    """ Returns a one channel image of made of values
        within the newDynamicRange applying a dithering
        according to the matrix.
    """
    checkIsOneChannel(imageIN)
    Nimage = imageIN.shape[0]
    Mimage = imageIN.shape[1]

    Nmatrix = np.shape(matrix)[0]
    Mmatrix = np.shape(matrix)[1]

    # making a clone to work with it
    imageOUT = imageIN.astype('float64')

    for j in range(Mimage):
        for i in range(Nimage):
            # loop over all pixels
            oldPixel = imageOUT[i, j]

            newPixel = getNewPixel(oldPixel, newDynamicRange)

            # applying the new value
            imageOUT[i, j] = newPixel

            # computing the error during the segmentation
            error = oldPixel - newPixel

            # spreading that error according to the matrix
            for iMatrix in range(Nmatrix):
                for jMatrix in range(Nmatrix):
                    # loop over all the matrix values
                    weigth = matrix[iMatrix][jMatrix]

                    # coordinates of the pixel where spreading
                    newI = i + iMatrix - int(Nmatrix/2)
                    newJ = j + jMatrix - int(Mmatrix/2)

                    # applying the spreading if it make sense
                    if weigth > 0 and 0 <= newI < Nimage and 0 <= newJ < Mimage:
                        imageOUT[newI, newJ] = imageOUT[newI, newJ] + error * weigth

    # making sure that the final image is within the newDynamicRange
    return getSegmentedImage(imageOUT, newDynamicRange)
