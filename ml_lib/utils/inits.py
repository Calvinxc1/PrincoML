import numpy as np
import torch as pt

class Constant():
    def __init__(self, constant = 0):
        self.constant = constant
        
    def init(self, tensor_shape):
        init_tensor = pt.from_numpy(np.full(tensor_shape, self.constant)).type(pt.Tensor)
        return init_tensor

class Uniform():
    def __init__(self, init_range = (-1, 1)):
        self.init_range = init_range
        
    def init(self, tensor_shape):
        range_val = self.init_range[1] - self.init_range[0]
        init_tensor = (pt.rand(tensor_shape) * range_val) + self.init_range[0]
        return init_tensor
    
class Normal():
    def __init__(self, mean = 0, stdev = 1):
        self.mean = mean
        self.stdev = stdev
        
    def init(self, tensor_shape):
        init_tensor = (pt.randn(tensor_shape) * self.stdev) + self.mean
        return init_tensor
    
class Xavier():
    def __init__(self, clamp = False):
        self.clamp = clamp
                 
    def init(self, tensor_shape):
        scalar = 2 / np.sum(tensor_shape)
        init_tensor = pt.randn(tensor_shape) * scalar
        if self.clamp: init_tensor = pt.clamp(init_tensor, -3 * scalar, 3 * scalar)
        return init_tensor
    
class Orthogonal():
    def __init__(self, xavier = False):
        self.xavier = xavier
        
    def init(self, tensor_shape):
        seed = np.random.randn(*tensor_shape)
        u, _, v = np.linalg.svd(seed, full_matrices = False)
        init_tensor = (u * (2 / np.sum(tensor_shape))) @ v if self.xavier else u @ v
        init_tensor = pt.from_numpy(init_tensor).type(pt.Tensor)
        return init_tensor