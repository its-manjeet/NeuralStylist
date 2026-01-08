import numpy as np

def get_loss_gradient(probabilities, target_index):
    """
    PHASE 6: The "Blame" Signal
    Calculates the derivative of the Loss Function.
    
    Math Shortcut: For Softmax + Cross-Entropy, the gradient is simply:
    (Predicted Probability - Actual Target)
    
    Args:
        probabilities (np.array): Output from Softmax (e.g., [[0.99, 0.01]])
        target_index (int): The correct answer (e.g., 0 for Vertical)
    
    Returns:
        d_scores: The gradient to send back to the Dense Layer.
    """
    # 1. Copy the probabilities so we don't modify the original
    d_scores = probabilities.copy()
    
    # 2. Subtract 1 from the correct class
    # Logic: If Prob is 0.99 and Target is 1.0 -> Error is -0.01 (Tiny error)
    # Logic: If Prob is 0.01 and Target is 1.0 -> Error is -0.99 (Huge error)
    d_scores[0, target_index] -= 1.0
    
    return d_scores

def dense_backward(d_scores, input_vector):
    """
    PHASE 6: The Weight Update Calculator
    Calculates how much each weight contributed to the error.
    
    Args:
        d_scores (np.array): The error signal from the loss. Shape: (1, 2)
        input_vector (np.array): The input the layer received. Shape: (1, 9)
        
    Returns:
        d_weights: Gradient for weights. Shape: (9, 2)
        d_biases: Gradient for biases. Shape: (1, 2)
    """
    # 1. Calculate Weight Gradient (dL/dW)
    # Math: Input Transpose dot Error
    # Why? If an Input pixel was bright (10) and Error is high, 
    # that specific weight gets a massive blame.
    d_weights = np.dot(input_vector.T, d_scores)
    
    # 2. Calculate Bias Gradient (dL/db)
    # The bias gradient is just the sum of errors.
    d_biases = np.sum(d_scores, axis=0, keepdims=True)
    
    return d_weights, d_biases
