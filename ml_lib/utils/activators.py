import numpy as np
import torch as pt

class Linear():
    def activate(self, input_tensor):
        return input_tensor

class Sigmoid(Linear):
    def activate(self, input_tensor):
        activated_tensor = 1 / (1 + pt.exp(-input_tensor))
        return activated_tensor

class ReLU(Linear):
    def activate(self, input_tensor):
        activated_tensor = pt.clamp(input_tensor, 0, np.inf)
        return activated_tensor