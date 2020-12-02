import numpy as np
from datetime import datetime as dt

from .RootSplit import RootSplit as Root

class BalanceSplit(Root):
    defaults = {
        **Root.defaults,
        'ratios': {'train': 0.8, 'holdout': 0.2}
    }
    
    def __init__(self, split_tensor, path_name = None, verbose = None,
                 ratios = None, balance_cols = np.array([])
                ):
        ratios = self.defaults['ratios'] if ratios is None else ratios
        self.set_ratios(ratios)
        
        self.balance_cols = balance_cols
        
        super().__init__(split_tensor, path_name = path_name, verbose = verbose)
    
    def set_ratios(self, ratios):
        if sum(ratios.values()) != 1: raise Exception('Ratio values do not sum to 1.')
        
        self.ratios = ratios
        
    def gen_split(self, split_tensor):
        split_array = split_tensor[:, self.balance_cols].detach().cpu().numpy()
        
        extra = (split_array.sum(axis = 1, keepdims = True) == 0).astype(int)
        if extra.sum() > 0:
            split_array = np.concatenate((split_array, extra), axis = 1)
            
        self.splits = {}
        for col_idx in range(split_array.shape[1]):
            split_idx = split_array[:, col_idx].nonzero()[0]
            np.random.shuffle(split_idx)
            anchor_len = split_idx.size
            anchor_idx = 0
            for split_type, split_ratio in self.ratios.items():
                span_idx = int(split_ratio * anchor_len) + anchor_idx
                sample = split_idx[anchor_idx:span_idx]
                if split_type not in self.splits.keys(): self.splits[split_type] = []
                self.splits[split_type].append(sample)
                anchor_idx = span_idx