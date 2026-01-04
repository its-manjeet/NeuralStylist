import numpy as np
from src.layers import manual_convolution_2d, relu, max_pooling, flatten, dense

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

from src.layers import dense # <--- Make sure you add 'dense' to the import line at the top!

# ... (Previous code remains the same) ...

print("\n--- PHASE 3: DECISION (The Brain) ---")

# 1. SETUP THE BRAIN CONNECTIONS
# Input Size = 4 (Because our vector is [30, 0, 0, 0])
# Output Size = 2 (We want 2 scores: Is it Class A or Class B?)
input_size = 4
output_size = 2

# Create Random Weights (The Synapses)
# In a real AI, 'Training' finds the perfect numbers here.
# For now, we use random numbers to test the math.
np.random.seed(42) # This makes sure we both get the same "random" numbers
weights = np.random.randn(input_size, output_size) 
bias = np.random.randn(output_size)

# 2. RUN THE DENSE LAYER
scores = dense(flat_out, weights, bias)

print(f"Weights Shape: {weights.shape}")
print(f"Final Scores (Logits): {scores}")

if scores[0] > scores[1]:
    print("Prediction: Class A (Vertical)")
else:
    print("Prediction: Class B (Horizontal)")

