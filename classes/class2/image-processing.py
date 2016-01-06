#!/usr/bin/env python
import improc
import numpy as np
import time

img = improc.read("gilman-hall.jpg")

# removed red channel
#img_nored = improc.rm_red(img)

# convert to gray
#img_gray_avg = improc.rgb_to_gray_avg(img)
#img_gray_lum = improc.rgb_to_gray_lum(img)

#bigger = improc.scale(img,1.5)
#smaller = improc.scale(bigger,2/3)
#diff = improc.difference(smaller,img)

#improc.show_n([img,smaller,diff])

n = 500
A = np.random.rand(n)
A2 = np.array(A)
t0 = time.time()
improc.selectionsort(A,n)
t1 = time.time()
improc.mergesort(A2,n)
t2 = time.time()
print(t1-t0, t2-t1)
