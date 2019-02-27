import numpy as np
import torch as pt

from ml_lib.utils.mergers.RootMerger import RootMerger as Root

class SurvivalMerger(Root):
    defaults = {
        **Root.defaults,
        'inputs': 3,
        'clamp': 32
    }
    
    def __init__(self, path_name = None, verbose = None,
                 clamp = None
                ):
        super().__init__(path_name = path_name, verbose = verbose)
        self.clamp = self.defaults['clamp'] if clamp is None else clamp
    
    def merge_process(self, input_tensor):
        time = input_tensor[:,:,0]
        
        alpha = input_tensor[:,:,1]
        alpha_mask = pt.abs(alpha) <= self.clamp
        for col_idx in range(alpha_mask.size()[1]):
            mask = alpha_mask[:,col_idx]
            alpha[mask,col_idx] = pt.log(1 + pt.exp(alpha[mask,col_idx]))
        
        beta = input_tensor[:,:,2]
        beta_mask = pt.abs(beta) <= self.clamp
        for col_idx in range(beta_mask.size()[1]):
            mask = beta_mask[:,col_idx]
            beta[mask,col_idx] = pt.log(1 + pt.exp(beta[mask,col_idx]))
        
        time_alpha = time + alpha
        
        merge_tensor = time_alpha / (time_alpha + beta)
        
        return merge_tensor