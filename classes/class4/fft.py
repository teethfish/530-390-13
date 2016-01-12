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

def fft(data, isign):
  n = len(data) >> 1
  if n < 2 or n & (n-1):
    print("data must be length of a power of two")
    return data
  else:
    nn = n << 1
    j = 1
    for i in range(1,nn,2):
      if j > i:
        tmp = data[i-1]
        data[i-1] = data[j-1]
        data[j-1] = tmp
        tmp = data[i]
        data[i] = data[j]
        data[j] = tmp
      m = n
      while m >= 2 and j > m:
        j = j - m
        m = m >> 1
      j = j + m
    mmax = 2
    while nn > mmax:
      istep = mmax << 1
      theta = isign*2.*PI/mmax
      wtemp = np.sin(0.5*theta)
      wpr = -2.*wtemp*wtemp
      wpi = np.sin(theta)
      wr = 1.
      wi = 0.
      for m in range(1,mmax,2):
        for i in range(m,nn,istep):
          j = i + mmax
          tempr = wr*data[j-1] - wi*data[j]
          tempi = wr*data[j] + wi*data[j-1]
          data[j-1] = data[i-1] - tempr
          data[j] = data[i] - tempi
          data[i-1] = data[i-1] + tempr
          data[i] = data[i] + tempi
        wtemp = wr
        wr = wtemp*wpr - wi*wpi + wr
        wi = wi*wpr + wtemp*wpi + wi
      mmax = istep
    if isign < 0:
      for i in range(2*n):
        data[i] = data[i] / n
    return data

def convolution(A,B):
  fft(A,1.)
  fft(B,1.)
  N = len(A) >> 1
  conv = np.zeros(2*N)
  for i in range(N):
    conv[2*i] = A[2*i]*B[2*i] - A[2*i+1]*B[2*i+1]
    conv[2*i+1] = A[2*i+1]*B[2*i] + A[2*i]*B[2*i+1]
  fft(conv,-1.)
  return conv

def convolution_slow(A,B):
  A = fft_slow(A,1.)
  B = fft_slow(B,1.)
  N = len(A) >> 1
  conv = np.zeros(2*N)
  for i in range(N):
    conv[2*i] = A[2*i]*B[2*i] - A[2*i+1]*B[2*i+1]
    conv[2*i+1] = A[2*i+1]*B[2*i] + A[2*i]*B[2*i+1]
  conv = fft_slow(conv,-1.)
  return conv

def pow2(n):
  n = n - 1
  n |= n >> 1
  n |= n >> 2
  n |= n >> 4
  n |= n >> 8
  n |= n >> 16
  n = n + 1
  return n
