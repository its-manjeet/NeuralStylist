import numpy as np
from src.layers import manual_convolution_2d, relu, max_pooling

# 1. THE INPUT (5x5 Image with a vertical line)
image = np.array([
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0]
])

# 2. THE FILTER (Vertical Edge Detector)
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

print("\n--- STEP 1: CONVOLUTION (The Scan) ---")
conv_output = manual_convolution_2d(image, kernel)
print(conv_output)
# Expect: 30s (match), 0s (ignore), -30s (mismatch)

print("\n--- STEP 2: RELU (The Bouncer) ---")
relu_output = relu(conv_output)
print(relu_output)
# Expect: Negatives becomes 0. Only 30s remain.

print("\n--- STEP 3: MAX POOLING (The Compressor) ---")
# We use stride=1 here just because our test image is tiny (3x3).
# In real life, we usually use stride=2.
pooled_output = max_pooling(relu_output, pool_size=2, stride=1)
print(pooled_output)
# Expect: The grid gets smaller, but the 30s survive.

