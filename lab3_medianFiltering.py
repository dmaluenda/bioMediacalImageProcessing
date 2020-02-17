#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:56:57 2020

@author: dMaluenda
"""

from imageTools import *

eye1, eye2 = importEyes()

eyeF = im2float(eye1)
eyeL = getLuminance(eyeF)

plt.figure()
kernels = [3, 9, 15, 21, 31, 91]
for Nkernel in kernels:

    eye_mFiltered, mFilter = medianFilter(eyeL, kernel=Nkernel)

    plt.imsave('lab3/medianFilter_%dx%d.png' % (Nkernel, Nkernel),
               mFilter, cmap='gray')
    plt.imsave('lab3/mFiltered_%dx%d.png' % (Nkernel, Nkernel),
               eye_mFiltered, cmap='gray')
