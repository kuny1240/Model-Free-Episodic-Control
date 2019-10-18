import numpy as np

def inverse_distance(h, h_i, epsilon=1e-3):
  return 1 / (np.linalg.norm(h, h_i) + epsilon)