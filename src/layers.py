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