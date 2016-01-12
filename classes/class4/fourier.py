#!/usr/bin/env python
import fft
import numpy as np
import matplotlib.pyplot as plt
import time
import improc

PI = 2*np.arcsin(1)
#N = 512
#L = 2*PI
#dx = L / (N-1)
#x = np.zeros(N)
#f = np.zeros(N)
#ys = np.zeros(2*N)
#yf = np.zeros(2*N)
#
#sigma = PI / 16. / 2
#dsigma2 = 1. / (sigma*sigma)
#C = 1. / (np.sqrt(2.*PI)*sigma)
#I = 0
#for i in range(N):
#  x[i] = i*dx
#  f[i] = i
#  ys[2*i] = np.sin(x[i]) + np.sin(4.*x[i]) + np.random.rand(1)
#  ys[2*i+1] = 0
#  yf[2*i] = C * np.exp(-x[i]*x[i]*dsigma2*0.5)
#  yf[2*i] = yf[2*i] + C * np.exp(-(x[i-1]-L)*(x[i-1]-L)*dsigma2*0.5)
#  yf[2*i+1] = 0
#  I = I + yf[2*i]
#for i in range(N):
#  yf[2*i] = yf[2*i] / I
#  yf[2*i+1] = yf[2*i+1] / I
#
#fft.plot_c(x,ys)
#fft.plot_c(x,yf)
#conv = fft.convolution(ys,yf)
#fft.plot_c(x,conv)

#fft.plot_c(x,y)
#t0 = time.time()
#y = fft.fft_slow(y,1.)
#t1 = time.time()
#fft.plot_c(f[:0.5*N],y[:N])
#y = fft.fft_slow(y,-1.)
#fft.plot_c(x,y)
#
#t2 = time.time()
#fft.fft(y,1.)
#t3 = time.time()
#fft.plot_c(f[:0.5*N],y[:N])
#fft.fft(y,-1.)
#
#print("Fast = " + str(t3-t2))

#plt.plot(x,y)

img = improc.rgb_to_gray_lum(improc.read("gilman-hall.jpg"))
data_real = img[:,0,0]
N = len(data_real)
N2 = fft.pow2(N)
x = np.zeros(N2)
L = N
L2 = N2
data = np.zeros(2*N2)
gaus = np.zeros(2*N2)

sigma = 2.*PI
dsigma2 = 1. / (sigma*sigma)
C = 1. / (np.sqrt(2.*PI)*sigma)
I = 0
for i in range(N2):
  x[i] = i
  if i < N:
    data[2*i] = data_real[i]
    data[2*i+1] = 0
  gaus[2*i] = C * np.exp(-x[i]*x[i]*dsigma2*0.5)
  gaus[2*i] = gaus[2*i] + C * np.exp(-(x[i]-L2)*(x[i]-L2)*dsigma2*0.5)
  gaus[2*i+1] = 0
  I = I + gaus[2*i]
for i in range(N2):
  gaus[2*i] = gaus[2*i] / I

data_slow = np.array(data)
gaus_slow = np.array(gaus)

t0 = time.time()
conv_fast = fft.convolution(data,gaus)
t1 = time.time()
conv_slow = fft.convolution_slow(data_slow,gaus_slow)
t2 = time.time()

print("Fast: ", t2-t1, "Slow: ", t1-t0)

#plt.show()
