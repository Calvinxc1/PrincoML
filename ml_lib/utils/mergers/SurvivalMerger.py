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
        time, alpha, beta = [input_tensor[:,:,idx] for idx in range(self.defaults['inputs'])]
        time_alpha = time + alpha
        merge_tensor = time_alpha / (time_alpha + beta)
        
        return merge_tensor