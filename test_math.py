import numpy as np
from src.layers import manual_convolution_2d, relu, max_pooling, flatten

# 1. THE INPUT (Image)
image = np.array([
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0]
])

# 2. THE FILTER (Vertical Edge)
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

print("\n--- PHASE 1: PERCEPTION (The Eye) ---")
conv_out = manual_convolution_2d(image, kernel)
relu_out = relu(conv_out)
pool_out = max_pooling(relu_out, pool_size=2, stride=1)

print("Final Image Shape (2D):", pool_out.shape)
print(pool_out)

print("\n--- PHASE 2: ADAPTATION (The Bridge) ---")
flat_out = flatten(pool_out)

print("Final Vector Shape (1D):", flat_out.shape)
print(flat_out)
print("\nSTATUS: Ready for the Brain (Dense Layer).")

