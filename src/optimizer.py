import numpy as np

def update_parameters(weights, bias, d_weights, d_biases, learning_rate):
    """
    PHASE 7: THE OPTIMIZER (The Fixer)
    
    This function applies the 'Learning Step'.
    It subtracts the Error Gradient from the current weights.
    
    The Formula:
    New_Weight = Old_Weight - (Learning_Rate * Gradient)
    
    Args:
        weights (np.array): The current brain connections.
        biases (np.array): The current baseline preferences.
        d_weights (np.array): The 'Blame Report' from Phase 6.
        d_biases (np.array): The 'Bias Blame' from Phase 6.
        learning_rate (float): How big of a step to take (Usually 0.01 or 0.1).
        
    Returns:
        new_weights, new_biases
    """

    # 1. Update Weights
    # If Gradient is POSITIVE (meaning we were too high), we SUBTRACT to go lower.
    # If Gradient is NEGATIVE (meaning we were too low), we ADD (subtracting a negative) to go higher.

    new_weights = weights - (learning_rate * d_weights)
    new_biases = bias - (learning_rate * d_biases)

    return new_weights, new_biases