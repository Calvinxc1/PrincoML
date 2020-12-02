import numpy as np
import torch as pt

from .RootMerger import RootMerger as Root

class SurvivalMerger(Root):
    defaults = {
        **Root.defaults,
        'clamp': 32,
        'output_space': 'prob'
    }
    
    inputs = 3
    
    def __init__(self, path_name = None, verbose = None,
                 clamp = None, output_space = None
                ):
        super().__init__(path_name = path_name, verbose = verbose)
        self.clamp = self.defaults['clamp'] if clamp is None else clamp
        self.output_space = self.defaults['output_space'] if output_space is None else output_space
    
    def merge_process(self, input_tensor):
        time, alpha, beta = [input_tensor[:,:,idx] for idx in range(self.inputs)]
        time_alpha = time + alpha
        
        if self.output_space == 'prob':
            merge_tensor = time_alpha / (time_alpha + beta)
        elif self.output_space == 'sig':
            merge_tensor = -pt.log(beta / time_alpha)
        else:
            raise Exception ('output_space set to "%s", but must be either "prob" or "sig".' % self.output_space)
        
        return merge_tensor