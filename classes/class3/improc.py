# improc.py (image processing module)
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

# read module
def read(fname):
  im = img.imread(fname)
  im = im.astype(float) / 255
  return im

# show image
def show(image):
  plt.imshow(image)
  plt.show()

# show n images
def show_n(image):
  n = len(image)
  for i in range(n):
    plt.figure(i)
    plt.imshow(image[i])
  plt.show()

# remove all red
def rm_red(image):
  tmp = np.array(image)
  [Nx, Ny, C] = tmp.shape
  for i in range(Nx):
    for j in range(Ny):
      tmp[i,j,0] = 0
  return tmp

# convert to grayscale (average)
def rgb_to_gray_avg(image):
  tmp = np.array(image)
  [Nx, Ny, C] = tmp.shape
  for i in range(Nx):
    for j in range(Ny):
      avg = (tmp[i,j,0] + tmp[i,j,1] + tmp[i,j,2]) / 3
      tmp[i,j,0] = avg
      tmp[i,j,1] = avg
      tmp[i,j,2] = avg
  return tmp

# convert to grayscale (luminosity)
def rgb_to_gray_lum(image):
  tmp = np.array(image)
  [Nx, Ny, C] = tmp.shape
  for i in range(Nx):
    for j in range(Ny):
      avg = 0.21*tmp[i,j,0] + 0.72*tmp[i,j,1] + 0.07*tmp[i,j,2]
      tmp[i,j,0] = avg
      tmp[i,j,1] = avg
      tmp[i,j,2] = avg
  return tmp

# scale (linear interpolation)
def scale(image, s):
  tmp = np.array(image)
  [Nx, Ny, C] = tmp.shape
  sNx = round(s * Nx)
  sNy = round(s * Ny)
  scaled = np.empty([sNx,sNy,3])
  ddx = 1 / Nx
  ddy = 1 / Ny
  for si in range(sNx):
    for sj in range(sNy):
      x = si / sNx
      y = sj / sNy
      i = round(x * Nx)
      j = round(y * Ny)
      if i == 0:
        dRdx = (tmp[i+1,j,0] - tmp[i,j,0]) * ddx
        dGdx = (tmp[i+1,j,1] - tmp[i,j,1]) * ddx
        dBdx = (tmp[i+1,j,2] - tmp[i,j,2]) * ddx
      elif i == Nx-1:
        dRdx = (tmp[i,j,0] - tmp[i-1,j,0]) * ddx
        dGdx = (tmp[i,j,1] - tmp[i-1,j,1]) * ddx
        dBdx = (tmp[i,j,2] - tmp[i-1,j,2]) * ddx
      else:
        dRdx = 0.5 * (tmp[i+1,j,0] - tmp[i-1,j,0]) * ddx
        dGdx = 0.5 * (tmp[i+1,j,1] - tmp[i-1,j,1]) * ddx
        dBdx = 0.5 * (tmp[i+1,j,2] - tmp[i-1,j,2]) * ddx
      if j == 0:
        dRdy = (tmp[i,j+1,0] - tmp[i,j,0]) * ddy
        dGdy = (tmp[i,j+1,1] - tmp[i,j,1]) * ddy
        dBdy = (tmp[i,j+1,2] - tmp[i,j,2]) * ddy
      elif j == Ny-1:
        dRdy = (tmp[i,j,0] - tmp[i,j-1,0]) * ddy
        dGdy = (tmp[i,j,1] - tmp[i,j-1,1]) * ddy
        dBdy = (tmp[i,j,2] - tmp[i,j-1,2]) * ddy
      else:
        dRdy = 0.5 * (tmp[i,j+1,0] - tmp[i,j-1,0]) * ddy
        dGdy = 0.5 * (tmp[i,j+1,1] - tmp[i,j-1,1]) * ddy
        dBdy = 0.5 * (tmp[i,j+1,2] - tmp[i,j-1,2]) * ddy
      scaled[si,sj,0] = tmp[i,j,0] + dRdx * (x - i*ddx) + dRdy * (y - j*ddy)
      scaled[si,sj,1] = tmp[i,j,1] + dGdx * (x - i*ddx) + dGdy * (y - j*ddy)
      scaled[si,sj,2] = tmp[i,j,2] + dBdx * (x - i*ddx) + dBdy * (y - j*ddy)
  return scaled

# difference
def difference(image1, image2):
  tmp = np.array(image1)
  [Nx, Ny, C] = tmp.shape
  for i in range(Nx):
    for j in range(Ny):
      tmp[i,j,0] = image2[i,j,0] - image1[i,j,0]
      tmp[i,j,1] = image2[i,j,1] - image1[i,j,1]
      tmp[i,j,2] = image2[i,j,2] - image1[i,j,2]
  return tmp

# maximum (unsorted)
def maximum(image, rgb):
  M = 0
  if rgb < 0 or rgb > 2:
    print("0 <= rgb <= 2")
  else:
    [Nx, Ny, C] = image.shape
    for i in range(Nx):
      for j in range(Ny):
        if image[i,j,rgb] > M:
          M = image[i,j,rgb]
  return M

# selection sort
def selectionsort(A, n):
  for i in range(n):
    m = A[i]
    mj = i
    for j in range(i,n):
      if A[j] < m:
        m = A[j]
        mj = j
    A[mj] = A[i]
    A[i] = m

# merge sort (entry point)
def mergesort(A,n):
  if n > 1:
    m = round(0.5 * n)
    mergesort(A[0:m],m)
    mergesort(A[m:], n-m)
    merge(A,n,m)

def merge(A,n,m):
  B = np.empty(n)
  i = 0  # first half index
  j = m  # second half index
  for k in range(n):
    if j == n:
      B[k] = A[i]
      i = i+1
    elif i == m:
      B[k] = A[j]
      j = j+1
    elif A[j] < A[i]:
      B[k] = A[j]
      j = j+1
    else:
      B[k] = A[i]
      i = i+1
  for i in range(n):
    A[i] = B[i]

# search (unsorted)
def search_unsorted(A, val):
  n = len(A)
  imatch = -1
  for i in range(n):
    if A[i] == val:
      imatch = i
  return imatch

# search (binary)
def binary_search(A, s, e, val):
  if e < s:
    return -1
  else:
    m = round(0.5*(s+e))
    if A[m] > val:
      return binary_search(A,s,m-1,val)
    elif A[m] < val:
      return binary_search(A,m+1,e,val)
    else:
      return m

# recursive factorial
def factorial(n):
  if n == 1:
    return 1
  else:
    print(n)
    return n * factorial(n-1)

# convolution
def convolution(image, kernel):
  tmp = np.array(image)
  [Nx, Ny, C] = tmp.shape
  kernel = np.array(kernel)
  [kNx, kNy] = kernel.shape
  for i in range(Nx):
    for j in range(Ny):
      print(i)
      Rval = 0
      Gval = 0
      Bval = 0
      for p in range(kNx):
        for q in range(kNy):
          ii = p - np.floor(0.5*kNx)
          jj = q - np.floor(0.5*kNy)
          if i+ii >= 0 and i+ii < Nx-1 and j+jj >= 0 and j+jj <= Ny-1:
            Rval = Rval + image[i+ii,j+jj,0]*kernel[p,q]
            Gval = Gval + image[i+ii,j+jj,1]*kernel[p,q]
            Bval = Bval + image[i+ii,j+jj,2]*kernel[p,q]
      tmp[i,j,0] = Rval
      tmp[i,j,1] = Gval
      tmp[i,j,2] = Bval
  return tmp
