import numpy as np
from src.layers import manual_convolution_2d, relu, max_pooling, flatten, dense
from src.activations import softmax
from src.loss import categorical_cross_entropy
from src.backprops import get_loss_gradient, dense_backward

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

print("\n---PHASE:4 ACTIVATION (The Translator)---")

probs = softmax(scores)

print(f"Probabiliities : {probs}")
print(f"Sum of Probs: {np.sum(probs)} (Should be 1.0)")

# Improved Prediction Logic
class_names = ["Vertical", "Horizontal"]
winner_index = np.argmax(probs) # Finds the index of the highest number
winner_name = class_names[winner_index]
confidence = probs[winner_index]*100

print(f"\nFINAL PREDICTION: {winner_name} ({confidence:.2f}% confidence)")

print(f"\n--- PHASE 5: THE JUDGE(Loss Calcualtion) ---")
# Let's say the TRUE label is Class 0 (Vertical)
target_class = 1
print(f"True Target: Class {target_class}(Horizontal)")

#Calculate Loss:
loss = categorical_cross_entropy(probs, target_class)
print(f"Loss (Error Score): {loss:.8f}")

if loss<0.1:
    print("Verdict:  Great Job Low Error")
else:
    print("Verdict: High Error, Punishment Needed")

print("\n--- PHASE 6: BACKPROPAGATION (The Blame Game) ---")

# 1. FIX: Reshape 'probs' to be a 2D matrix (1 Row, 2 Columns)
# Before: [0.99, 0.01]  (Shape: (2,))
# After: [[0.99, 0.01]] (Shape: (1, 2))
probs_matrix = probs.reshape(1, -1)

# 2. Get the initial error signal (d_scores)
# Now passing the matrix, so the function won't crash
d_scores = get_loss_gradient(probs_matrix, target_class)
print(f"1. Error Signal (d_scores):\n {d_scores}")

# 3. FIX: Reshape 'flat_out' to be a 2D matrix (1 Row, 4 Columns)
flat_out_matrix = flat_out.reshape(1, -1)

# 4. Calculate the Weight Gradients
d_weights, d_biases = dense_backward(d_scores, flat_out_matrix)

print(f"\n2. Weight Gradients (Shape: {d_weights.shape}):")
print(d_weights)
print(f"\n3. Bias Gradients (Shape: {d_biases.shape}):")
print(d_biases)

print("\nSUCCESS: We have the map to fix the brain!")

# ... (Keep all imports and Phase 1-6 code exactly as it is) ...

# NEW IMPORT
from src.optimizer import update_parameters

print("\n--- PHASE 7: OPTIMIZATION (The Fix) ---")

# 1. Print the "Old" Brain
print("1. Old Weights (Top Row Only):")
print(weights[0]) 
# Example: [0.49, -0.13] (Random numbers)

# 2. Define the Learning Rate
learning_rate = 0.1
print(f"Learning Rate: {learning_rate}")

# 3. Run the Optimizer
# We pass the current weights and the gradients we just calculated in Phase 6
new_weights, new_biases = update_parameters(
    weights, bias, d_weights, d_biases, learning_rate
)

# 4. Print the "New" Brain
print("\n2. New Weights (Top Row Only):")
print(new_weights[0])

# 5. Verify the Math manually
print("\n3. Manual Check:")
# Formula: Old - (Rate * Gradient)
gradient_value = d_weights[0][0] # Should be around 30.0
old_value = weights[0][0]
manual_calc = old_value - (learning_rate * gradient_value)
print(f"Old Weight: {old_value:.4f}")
print(f"Gradient: {gradient_value:.4f}")
print(f"Math: {old_value:.4f} - (0.1 * {gradient_value:.4f}) = {manual_calc:.4f}")
print(f"New Weight: {new_weights[0][0]:.4f}")

if new_weights[0][0] == manual_calc:
    print("\n VERDICT: Optimization logic is PERFECT.")
else:
    print("\n VERDICT: Math mismatch.")


