import numpy as np

def manual_convolution_2d(input_image, kernel):
    """
    The Foundation: A manual 2D convolution.
    Input: A grayscale image (2D Matrix).
    Kernel: A filter (Small 2D Matrix) that hunts for patterns.
    """
    
    # 1. GET DIMENSIONS
    # We need to know how big the playground is.
    image_h, image_w = input_image.shape
    kernel_h, kernel_w = kernel.shape
    
    # 2. CALCULATE OUTPUT SIZE (The "Valid" Formula)
    # Why +1? Because if the filter sits on the last pixel, we stop.
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1
    
    # 3. PREPARE THE CANVAS
    # A blank matrix of zeros where we will store the "activations" (matches).
    output_map = np.zeros((output_h, output_w))
    
    # 4. THE SLIDING WINDOW (The Loop)
    # We slide the filter over the image, row by row, col by col.
    for i in range(output_h):
        for j in range(output_w):
            
            # 5. THE CROP (Region of Interest)
            # We slice a small chunk of the image exactly the size of our kernel.
            region = input_image[i : i + kernel_h, j : j + kernel_w]
            
            # 6. THE VIBE CHECK (Convolution Operation)
            # Element-wise multiplication followed by a Sum.
            # High number = Strong match. Zero = No match.
            output_map[i, j] = np.sum(region * kernel)
            
    return output_map

def relu(feature_map):
    """
    The 'Bouncer' Function.
    Anything negative becomes 0.
    Anything positive stays the same.
    """
    return np.maximum(0, feature_map)

def max_pooling(feature_map, pool_size = 2, stride = 2):
    """
    Downsamples the image by keeping only the strongest feature.
    
    Args:
        feature_map: The input matrix (from ReLU).
        pool_size: Size of the window (usually 2x2).
        stride: How much we jump (usually 2).
    
    Returns:
        The compressed version of the map (Half the size).
    """
    # 1. Get Dimensions
    h_prev, w_prev = feature_map.shape

    # 2. Calculate Output Dimensions
    # Formula: (Input - Pool) / Stride + 1
    h = int((h_prev - pool_size)/ stride + 1)
    w = int((w_prev - pool_size)/stride + 1)

    downsampled_map = np.zeros((h,w))

    # 4. The Loop (Jumping, not Sliding)
    for i in range(h):
        for j in range(w):
            # Find the starting points in the original map
            start_row = i * stride
            start_col = j * stride

            #Extract the 2*2 Patch:
            region = feature_map[start_row : start_row + pool_size,
                                 start_col : start_col + pool_size]
            
            # 5. The Selection: Keep ONLY the max value
            downsampled_map[i,j] = np.max(region)

        return downsampled_map
    

