import numpy as np

from ml_lib.clusters.data_cluster.batchers.RootBatch import RootBatch as Root

class FlatBatch(Root):
    defaults = {
        **Root.defaults,
        'proportion': 1.0
    }
    
    def __init__(self, proportion = None, path_name = None, verbose = None):
        self.prop = self.defaults['proportion'] if proportion is None else proportion
        
        super().__init__(path_name = path_name, verbose = verbose)
        
    def batch_obs(self, obs_splits):
        batch_splits = {}
        for key, value in obs_splits.items():
            batch_count = int(value.size * self.prop) if type(self.prop) is float else self.prop
            split_batch = np.random.choice(value, size = batch_count)
            batch_splits[key] = split_batch
            
        self._v_msg('Batch generated: %s.' % (
            ', '.join(['%s %s obs' % (key, value.size) for key, value in batch_splits.items()]),
        ))
            
        return batch_splits