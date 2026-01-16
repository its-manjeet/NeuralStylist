import numpy as np
# Import all the tools we built
from src.layers import manual_convolution_2d, relu, max_pooling, flatten, dense
from src.activations import softmax
from src.loss import categorical_cross_entropy
from src.backprops import get_loss_gradient, dense_backward
from src.optimizer import update_parameters
import pickle

# ==========================================
# 1. PREPARE THE DATA
# ==========================================

# Image 1: VERTICAL LINE (Class 0)
img_vertical = np.array([
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0]
])
label_vertical = 0

# Image 2: HORIZONTAL LINE (Class 1)
img_horizontal = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [10, 10, 10, 10, 10],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
label_horizontal = 1

dataset = [
    (img_vertical, label_vertical),
    (img_horizontal, label_horizontal)
]

# Fixed Filter (Vertical Edge Detector)
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# ==========================================
# 2. INITIALIZE THE BRAIN
# ==========================================
print("--- INITIALIZING NEURAL STYLIST ---")
np.random.seed(42)

input_size = 4 
output_size = 2 

weights = np.random.randn(input_size, output_size)
bias = np.random.randn(output_size)

print(f"Initial Random Weights (Top Row):\n{weights[0]}")

# ==========================================
# 3. THE TRAINING LOOP (The Gym)
# ==========================================
epochs = 20
learning_rate = 0.1

print("\n--- STARTING TRAINING ---")

for epoch in range(epochs):
    total_loss = 0
    
    for image, target in dataset:
        # --- FORWARD PASS ---
        conv_out = manual_convolution_2d(image, kernel)
        relu_out = relu(conv_out)
        pool_out = max_pooling(relu_out, pool_size=2, stride=1)
        
        flat_out = flatten(pool_out)
        flat_out_matrix = flat_out.reshape(1, -1)
        
        scores = dense(flat_out, weights, bias)
        
        # FIX IS HERE: .flatten() forces the matrix into a simple list
        # This ensures Softmax compares Vertical vs Horizontal correctly
        probs = softmax(scores.flatten())
        
        # --- LOSS ---
        loss = categorical_cross_entropy(probs, target)
        total_loss += loss
        
        # --- BACKWARD PASS ---
        # We reshape back to matrix for the math to work
        probs_matrix = probs.reshape(1, -1)
        
        d_scores = get_loss_gradient(probs_matrix, target)
        d_weights, d_biases = dense_backward(d_scores, flat_out_matrix)
        
        # --- OPTIMIZE ---
        weights, bias = update_parameters(weights, bias, d_weights, d_biases, learning_rate)

    if epoch % 2 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

print("\n--- TRAINING COMPLETE ---")
print(f"Final Loss: {total_loss:.4f}")
print(f"Smart Weights (Top Row):\n{weights[0]}")

# ==========================================
# 4. THE FINAL TEST
# ==========================================
print("\n--- TESTING THE SMARTER BRAIN ---")

# Test 1: Vertical
print("Test 1: Vertical Image")
conv_out = manual_convolution_2d(img_vertical, kernel)
pool_out = max_pooling(relu(conv_out), 2, 1)
flat_out = flatten(pool_out)
scores = dense(flat_out, weights, bias)
# FIX HERE TOO:
probs = softmax(scores.flatten())
print(f"Probabilities: Vertical={probs[0]:.4f}, Horizontal={probs[1]:.4f}")

# Test 2: Horizontal
print("\nTest 2: Horizontal Image")
conv_out = manual_convolution_2d(img_horizontal, kernel)
pool_out = max_pooling(relu(conv_out), 2, 1)
flat_out = flatten(pool_out)
scores = dense(flat_out, weights, bias)
# FIX HERE TOO:
probs = softmax(scores.flatten())
print(f"Probabilities: Vertical={probs[0]:.4f}, Horizontal={probs[1]:.4f}")

# 5. SAVE THE BRAIN (Persistence)
# ==========================================
print("\n--- SAVING BRAIN ---")
brain_memory = {
    "filter": kernel,   # The Eye (Convolution)
    "weights": weights, # The Brain (Dense)
    "bias": bias        # The Bias
}

with open("brain.pkl", "wb") as f:
    pickle.dump(brain_memory, f)

print(" Brain saved to 'brain.pkl'. You can now run predict.py!")