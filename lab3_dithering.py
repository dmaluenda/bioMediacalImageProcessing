#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:56:57 2020

@author: dMaluenda
"""

from imageTools import *
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

FSmatrix = [[0, 0, 0],
            [0, 0, 7],
            [3, 5, 1]]
FSmatrix = np.divide(FSmatrix, np.sum(FSmatrix))

JJNmatrix = [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 7, 5],
             [3, 5, 7, 5, 3],
             [1, 3, 5, 3, 1]]
JJNmatrix = np.divide(JJNmatrix, np.sum(JJNmatrix))


artImageColor = im2float(plt.imread('artImage.jpg'))
artImageGray = getLuminance(artImageColor)

plt.imsave('lab3/dith_NONE.png', artImageGray, cmap='gray')
for DRlevel in [2, 4]:
    plt.imsave('lab3/bin_%d.png' % DRlevel,
               getSegmentedImage(artImageGray, DRlevel),
               cmap='gray')
    for label, mat in zip(['FS', 'JJN'], [FSmatrix, JJNmatrix]):
        ditherImage = dithering(artImageGray, mat, DRlevel)
        plt.imsave('lab3/dith_%s_%d.png' % (label, DRlevel),
                   ditherImage, cmap='gray')


artHSV = rgb_to_hsv(artImageColor)

artH, artS, artV = getChannels(artHSV, doSeparatelly=True)

artDith = np.copy(artHSV)


dithV = dithering(artV, FSmatrix, 6)

artDith[:, :, 2] = dithV

plt.imsave('lab3/dithColor_FS.png', hsv_to_rgb(artDith))
