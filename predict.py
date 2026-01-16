import numpy as np
import pickle
from src.layers import manual_convolution_2d, relu, max_pooling, flatten, dense
from src.activations import softmax

# 1. LOAD THE SAVED WEIGHTS
print("--- LOADING BRAIN ---")
try:
    with open('brain.pkl','rb') as f:
        brain = pickle.load(f)
        print("Brain Loaded Successfully.")
except FileNotFoundError:
    print("Error: brain.pkl not found . Run train.py as first")
    exit()

# 2. CREATE A NEW IMAGE (The Test)
# A vertical line on the far right (Index 4)
input_image = np.array([
    [0, 0, 0, 0, 10],
    [0, 0, 0, 0, 10],
    [0, 0, 0, 0, 10],
    [0, 0, 0, 0, 10],
    [0, 0, 0, 0, 10]
])

print("\n---NEW INPUT---")
print(input_image)

# 3. THE FORWARD PASS (The Brain Thinking)
# Step A: The Eye (Convolution)
conv_out = manual_convolution_2d(input_image, brain['filter'])
relu_out = relu(conv_out)
pool_out = max_pooling(relu_out,2,1)

#Step:B : The Bridge (Flatten)
flat_out = flatten(pool_out)

# Step C: The Decision (Dense Layer)
# This is the "Brain" part you were asking about
logits = dense(flat_out, brain['weights'], brain['bias'])

# Step D: The Confidence (Softmax)
probs = softmax(logits.flatten())

# 4. REPORT
print("\n--- BRAIN DIAGNOSIS ---")
print(f"Vertical Confidence:   {probs[0]*100:.2f}%")
print(f"Horizontal Confidence: {probs[1]*100:.2f}%")

if probs[0] > probs[1]:
    print("RESULT: VERTICAL (CORRECT) ")
else:
    print("RESULT: HORIZONTAL (INCORRECT) ")