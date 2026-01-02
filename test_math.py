import numpy as np
from src.layers import manual_convolution_2d

# 1. CREATE THE VIBE (The Image)
# Imagine a dark room (0s) with a bright vertical light beam (10s) in the middle.
image = np.array([
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0]
])

# 2. CREATE THE DETECTOR (The Kernel)
# A Vertical Edge Detector.
# Left side negative, Right side positive. It LOVES vertical contrast.
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# 3. RUN THE SYSTEM
print("--- INITIALIZING NEURAL STYLIST ---")
result = manual_convolution_2d(image, kernel)

print("\nInput Image (The Data):")
print(image)

print("\nFilter (The Detector):")
print(kernel)

print("\nFeature Map (The Result):")
print(result)