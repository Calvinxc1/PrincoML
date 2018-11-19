import numpy as np
import torch as pt

class Constant():
    def __init__(self, constant):
        self.constant = constant
        
    def init(self, tensor_shape):
        init_tensor = pt.from_numpy(np.full(tensor_shape, self.constant)).type(pt.Tensor)
        return init_tensor

class Uniform(Constant):
    def __init__(self, init_range):
        self.init_range = init_range
        
    def init(self, tensor_shape):
        range_val = self.init_range[1] - self.init_range[0]
        init_tensor = (pt.rand(tensor_shape) * range_val) + self.init_range[0]
        return init_tensor
    
class Normal(Constant):
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev
        
    def init(self, tensor_shape):
        init_tensor = (pt.randn(tensor_shape) * self.stdev) + self.mean
        return init_tensor