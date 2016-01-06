#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

# this is a comment
n = 100
s = 0
e = 2*np.pi
dx = (e - s) / (n-1)
x = np.zeros(n)
y = np.zeros(n)

for i in range(n):
  x[i] = s + i * dx
  y[i] = np.sin(x[i])

plt.plot(x,y)
plt.show()
