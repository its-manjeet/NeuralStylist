import numpy as np

def softmax(x):
    """
    Computes the softmax values for each set of scores in x.
    
    Formula: exp(x) / sum(exp(x))
    
    Trick: We subtract the max(x) for 'Numerical Stability'. 
    If we don't, e^1000 causes an overflow (computer crash). 
    Subtracting the max keeps the exponents negative or zero, which is safe.
    """
    # 1. Shift values for stability (The Engineering Fix)
    # If x is [14.43, -3.60], max is 14.43.
    # We essentially do: [0, -18.03]. Much safer for the computer.
    e_x = np.exp(x -np.max(x))

    # 2. Divide by the total sum to get percentages
    return e_x /e_x.sum(axis = 0)