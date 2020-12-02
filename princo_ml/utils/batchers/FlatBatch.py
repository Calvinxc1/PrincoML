import numpy as np

from .RootBatch import RootBatch as Root

class FlatBatch(Root):
    defaults = {
        **Root.defaults,
        'proportion': 1.0,
        'replace': False
    }
    
    def __init__(self, path_name = None, verbose = None,
                 proportion = None, replace = None
                ):
        self.prop = self.defaults['proportion'] if proportion is None else proportion
        self.replace = self.defaults['replace'] if replace is None else replace
        
        super().__init__(path_name = path_name, verbose = verbose)
        
    def batch_obs(self, obs_splits):
        batch_splits = []
        
        for key, value in obs_splits.items():
            working_split = np.concatenate(value)
            batch_count = int(working_split.size * self.prop) if type(self.prop) is float else self.prop
            overall_batch = np.random.choice(working_split, size = batch_count, replace = self.replace)
            
            batch_splits.append({
                'name': key,
                'index': overall_batch
            })
            
        self._v_msg('Batch generated: %s.' % (
            ', '.join(['%s %s obs' % (batch['name'], batch['index'].size) for batch in batch_splits]),
        ))
            
        return batch_splits