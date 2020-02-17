#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:22:59 2020

@author: dMaluenda
"""

from imageTools import *

# %% Read images from file

eye1, eye2 = importEyes()

printImageInfo(eye1)

plt.imshow(eye1)
plt.imsave("lab2/eye1.png", eye1)


#%% Manipulate components of image

channels = getChannels(eye1)
Nchannels = len(channels)

# alpha channel is included just in case
chLabels = ['R', 'G', 'B', 'alpha']
plt.figure()
for chIdx in range(Nchannels):
    color = channels[chIdx]

    # upper row
    plt.subplot(2, Nchannels, 1+chIdx)
    plt.imshow(color, cmap='gray', vmin=0, vmax=255)
    plt.title(chLabels[chIdx] + " channel")
    plt.axis('off')

    # downer row
    colorIm = np.zeros(eye1.shape, 'uint8')
    colorIm[:, :, chIdx] = color
    
    plt.subplot(2, Nchannels, Nchannels+1+chIdx)
    plt.imshow(colorIm)
    title = ['0']*Nchannels
    title[chIdx] = chLabels[chIdx]
    plt.title("image %s" % title)
    plt.axis('off')

plt.savefig("lab2/channels.png")


#%% Luminance, colormap, pseudocolor and gamma


#%%% Average
    
X1 = 564
Y2 = 345

avg_1 = ( eye1[X1, Y2, 0] + eye1[X1, Y2, 1] + eye1[X1, Y2, 2] ) / 3

avg_2 = eye1[X1, Y2, 0]/3 + eye1[X1, Y2, 1]/3 + eye1[X1, Y2, 2]/3

print("\n\n >>> avg_1=%.2f (%s) is different than avg_2=%.2f (%s)), "
      "which one is the correct?\n\n" % (avg_1, type(avg_1), avg_2, type(avg_2)))


#%%% Average II


plt.figure()
plt.imshow(getAverage(eye1), cmap='gray', vmin=0, vmax=255)
plt.title("Average: (R + G + B)/3")
plt.savefig("lab2/average.png")


#%%% Luminance

luminance = getLuminance(eye1)

plt.figure()
idx = 1
for cmap in ['gray', 'jet', 'prism', 'hsv']:
    plt.subplot(2, 2, idx)
    plt.imshow(luminance, cmap=cmap)
    plt.title("Colormap: %s" % cmap)
    plt.axis('off')
    idx += 1

plt.savefig("lab2/luminance.png")


#%%% Gamma


imFloat = im2float(eye2)
printImageInfo(imFloat)

plt.figure()
for idx, g in enumerate([0.1, 0.5, 0.8, 1, 1.2, 2, 5, 10]):    
    plt.subplot(2, 4, idx+1)
    plt.imshow(imFloat**g)
    plt.axis('off')
    plt.title(f'gamma={g}')

plt.savefig("lab2/gammaFilters.png")









