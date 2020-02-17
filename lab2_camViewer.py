#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:38:36 2020

@author: dMaluenda
"""

import time
from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows

from imageTools import *

#%% Choosing a filter to apply

filterStr = ""
while not filterStr.isdigit():
    print("""
          Which filter do you want to use?
          0) None
          1) luminance
          2) average
          3) gamma contrast
    """)
    filterStr = input()
    if not filterStr.isdigit() or not -1<int(filterStr)<4:
        print("Wrong answer, type a number...")
        continue
    elif int(filterStr) == 3:
        gamma = float(input("Which gamma do you want to apply?"))
    filterNum = int(filterStr)
    
filters = [lambda x: x, getLuminance, getAverage, gammaFilter]
filterFunc = filters[filterNum]
arg = (gamma,) if filterNum == 3 else tuple()
  

#%% Showing the camera view with a filter
       
cap = VideoCapture(0)
time.sleep(2)

print("cap: %s" % cap)
print("cam is open? %s" % cap.isOpened())

while True:
    try:
        ret, frame = cap.read()
        imshow('Camera', filterFunc(frame, *arg))
        if waitKey(1) == ord('q'):
            break
    except Exception as exc:
        print(exc)
        break
    except:
        break

cap.release()
destroyAllWindows()


print("Session ended")
