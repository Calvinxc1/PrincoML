import torch as pt
import numpy as np

from .RootLearnNoise import RootLearnNoise as Root

class StochasticLearnNoise(Root):
    defaults = {
        **Root.defaults,
        'scale': 0.3,
        'decay': 0.55
    }
    
    def __init__(self, path_name = None, verbose = None, seed_learn = None, scale = None, decay = None):
        self.scale = self.defaults['scale'] if scale is None else scale
        self.decay = self.defaults['decay'] if decay is None else decay
        self.iters = 0
        
    def gen_noise(self, noise_shape):
        noise_scale = self.scale / ((1+self.iters) ** self.decay)
        noise_tensor = pt.randn(noise_shape) * noise_scale
        return noise_tensor