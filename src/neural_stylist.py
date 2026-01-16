import numpy as np
import pickle
import os
from src.layers import manual_convolution_2d, relu, max_pooling, flatten, dense
from src.activations import softmax

class NeuralStylist:
    def __init__(self, model_path= 'brain.pkl'):
        """
        The Agent Constructor.
        It tries to load a brain immediately when you create the agent.
        """
        self.brain = None
        self.model_path = model_path
        self.load_brain()

    def load_brain(self):
        """Loads the weights (The Brain) from the disk."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.brain = pickle.load(f)
            print("Brain loaded successfully")
        else:
            print(f"Warning: {self.model_path} not found. Agent is untrained.")

    def predict(self, image):
        """
        The Vision System.
        Input: 5x5 Numpy Array
        Output: Prediction Dictionary
        """
        if self.brain is None:
            return {"error": "Brain is empty. Train the model first."}
        
        # 1. The Forward Pass (The "Thinking")
        # We use Stride=1 for pooling to match your specific training setup
        conv_out = manual_convolution_2d(image, self.brain['filter'])
        relu_out = relu(conv_out)
        pool_out = max_pooling(relu_out, pool_size=2, stride=1)
        flat_out = flatten(pool_out)

        # 2. The Decision
        logits = dense(flat_out, self.brain['weights'], self.brain['bias'])
        probs = softmax(logits.flatten())

        # 3. The Report (Formatting for Humans)
        vertical_conf = probs[0]* 100
        horizontal_conf = probs[1]* 100

        prediction = "VERTICAL" if probs[0]>probs[1] else 'HORIZONTAL'

        return{
            "prediction": prediction,
            "confidence_score": f"{max(vertical_conf, horizontal_conf):.2f}%",
            "details": {
                "vertical_prob": vertical_conf,
                "horizontal_prob": horizontal_conf
        }
        }




