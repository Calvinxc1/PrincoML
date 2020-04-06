import numpy as np
from datetime import datetime as dt

from .RootSplit import RootSplit as Root

class BaseSplit(Root):
    defaults = {
        **Root.defaults,
        'ratios': {'train': 0.8, 'holdout': 0.2}
    }
    
    def __init__(self, split_tensor, path_name = None, verbose = None,
                 ratios = {}
                ):
        ratios = {**self.defaults['ratios'], **ratios}
        self.set_ratios(ratios)
        
        super().__init__(split_tensor, path_name = path_name, verbose = verbose)
    
    def set_ratios(self, ratios):
        if sum(ratios.values()) != 1: raise Exception('Ratio values do not sum to 1.')
        
        self.ratios = ratios
    
    def gen_split(self, split_tensor):
        
        def gen_breaks(split_len, ratios):
            break_idx = 0
            breaks = []
            
            for ratio in ratios[:-1]:
                break_idx += int(split_len * ratio)
                breaks.append(break_idx)
            
            return breaks
        
        split_len = split_tensor.size()[0]
        split_base = np.arange(split_len)
        np.random.shuffle(split_base)
        
        breaks = gen_breaks(split_len, list(self.ratios.values()))
        split_items = np.split(split_base, breaks)
        
        self.splits = {}
        for split_name, split_item in zip(self.ratios.keys(), split_items):
            self.splits[split_name] = [split_item]
        
        self._v_msg('Splits generated: %s. %s total obs' % (
            ', '.join(['%s %s obs' % (key, np.concatenate(value).size) for key, value in self.splits.items()]),
            split_len
        ))