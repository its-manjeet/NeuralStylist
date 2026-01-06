import numpy as np

def categorical_cross_entropy(probs, target_index):
    import numpy as np

def categorical_cross_entropy(probs, target_index):
    """
    Calculates the loss (error) for a single example.
    
    Args:
        probs (np.array): The output probabilities from Softmax (e.g., [0.1, 0.9])
        target_index (int): The correct class index (0 or 1).
        
    Returns:
        float: The loss value (Lower is better).
    """
    # 1. Get the probability the AI assigned to the CORRECT class
    correct_prob = probs[target_index]
    
    # 2. Safety Net: Log(0) is negative infinity (Computer explodes).
    # We clip the value to be slightly above 0.
    epsilon = 1e-15
    correct_prob = np.clip(correct_prob, epsilon, 1.0 - epsilon)
    
    # 3. Calculate Negative Log Likelihood
    loss = -np.log(correct_prob)
    
    return loss