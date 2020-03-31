import numpy as np


def std_Normalize(x):
  z = np.max(x) - np.min(x)
  x_norm = (x - np.min(x)) / z
  return x_norm

def l2_Normalize(x):
  x = x - np.mean(x)
  z = np.sqrt(np.sum(np.multiply(x, x)))
  x_norm = x / z
  return x_norm