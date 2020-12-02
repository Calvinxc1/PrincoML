import numpy as np

from .RootBatch import RootBatch as Root

class LowestBatch(Root):
    defaults = {
        **Root.defaults,
        'replace': False
    }
    
    def __init__(self, path_name = None, verbose = None,
                 proportion = None, replace = None
                ):
        self.replace = self.defaults['replace'] if replace is None else replace
        
        super().__init__(path_name = path_name, verbose = verbose)
        
    def batch_obs(self, obs_splits):
        batch_splits = []
        
        for key, value in obs_splits.items():
            lowest_size = min([idx_list.size for idx_list in value])
            overall_batch = np.concatenate([np.random.choice(idx_list, lowest_size, replace = False) for idx_list in value])
            
            batch_splits.append({
                'name': key,
                'index': overall_batch
            })
            
        self._v_msg('Batch generated: %s.' % (
            ', '.join(['%s %s obs' % (batch['name'], batch['index'].size) for batch in batch_splits]),
        ))
            
        return batch_splits