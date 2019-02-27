import torch as pt

from ml_lib.clusters.RootCluster import RootCluster as Root
from ml_lib.utils.mergers.AddMerger import AddMerger
from ml_lib.utils.learn_modules.activators.LinearActivate import LinearActivate

class MergeCluster(Root):
    defaults = {
        **Root.defaults,
        'merger': AddMerger, 'merger_kwargs': {},
        'activator': LinearActivate, 'activator_kwargs': {}
    }
    
    def __init__(self, cluster_name, path_name = None, verbose = None,
                 merger = None, merger_kwargs = None,
                 activator = None, activator_kwargs = None
                ):
        merger = self.defaults['merger'] if merger is None else merger
        merger_kwargs = self.defaults['merger_kwargs'] if merger_kwargs is None else merger_kwargs
        
        activator = self.defaults['activator'] if activator is None else activator
        activator_kwargs = self.defaults['activator_kwargs'] if activator_kwargs is None else activator_kwargs
        
        super().__init__(cluster_name, path_name = path_name, verbose = verbose)
        
        self.Merger = merger(path_name = '%s:%s' % (self.path_name, self.name), **merger_kwargs)
        self.Activator = activator(path_name = '%s:%s' % (self.path_name, self.name), **activator_kwargs)
        
    @property
    def input_count(self):
        input_count = max([
            link['cluster'].get_output_count(self.name)
            for link in self.links['input']
        ])
        return input_count
    
    @property
    def input_tensor(self):
        input_tensor = pt.stack([
            link['cluster'].get_output_tensor(self.name)
            for link in self.links['input']
        ], dim = 2)
        return input_tensor
    
    def get_output_count(self, req_cluster_name):
        return self.input_count
    
    def get_output_tensor(self, req_cluster_name):
        if self.enable is False: raise Exception('Cluster is not enabled!')
        
        if self.buffer is None:
            merged_tensor = self.Merger.merge(self.input_tensor)
            output_tensor = self.Activator.activate(merged_tensor)
            self.buffer = output_tensor.clone()
        else:
            output_tensor = self.buffer
            self._v_msg('Retrieving output tensor from buffer.')
            
        return output_tensor