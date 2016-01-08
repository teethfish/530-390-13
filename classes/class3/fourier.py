#!/usr/bin/env python
import fft
import numpy as np
import matplotlib.pyplot as plt
import time

PI = 2*np.arcsin(1)
N = 64
L = 2*PI
dx = L / (N-1)
x = np.zeros(N)
f = np.zeros(N)
y = np.zeros(2*N)

for i in range(N):
  x[i] = i*dx
  f[i] = i
  y[2*i] = np.sin(x[i]) + np.sin(4.*x[i])
  y[2*i+1] = 0

fft.plot_c(x,y)
y = fft.fft_slow(y,1.)
fft.plot_c(f[:0.5*N],y[:N])
y = fft.fft_slow(y,-1.)
fft.plot_c(x,y)

#plt.plot(x,y)
plt.show()
