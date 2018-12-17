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
    def __init__(self, leak = 0):
        self.leak = leak
        
    def activate(self, input_tensor):
        activated_tensor = pt.max(input_tensor, input_tensor * self.leak)
        return activated_tensor
    
class SoftMax(Linear):
    def activate(self, input_tensor):
        activated_tensor = pt.softmax(input_tensor, dim = 1)
        return activated_tensor
    
class SoftPlus(Linear):
    def activate(self, input_tensor):
        activated_tensor = pt.log(1 + pt.exp(input_tensor))
        return activated_tensor