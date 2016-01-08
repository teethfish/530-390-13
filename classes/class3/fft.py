import numpy as np
import matplotlib.pyplot as plt

PI = 2*np.arcsin(1)

# plot complex number
def plot_c(x,y):
  N = len(x)
  yr = np.zeros(N)
  yi = np.zeros(N)
  for i in range(N):
    yr[i] = y[2*i]
    yi[i] = y[2*i+1]
  plt.figure()
  plt.plot(x,yr,x,yi)
  plt.legend(["real","imaginary"])
  plt.show(block=False)

def fft_slow(data, isign):
  N = len(data) >> 1
  tmp = np.zeros(2*N)
  if N < 2 or N & (N-1): # checks if power of two
    print("data must be length of power of two")
    return data
  else:
    for n in range(N):
      for k in range(N):
        theta = isign*2.*PI*k*n/N
        ct = np.cos(theta)
        st = np.sin(theta)
        tmp[2*n] = tmp[2*n] + ct*data[2*k] - st*data[2*k+1]
        tmp[2*n+1] = tmp[2*n+1] + ct*data[2*k+1] + st*data[2*k]
    if isign < 0:
      for i in range(2*N):
        tmp[i] = tmp[i] / N
    return tmp
