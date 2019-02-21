from ml_lib.clusters.root_cluster.RootCluster import RootCluster as Root
from ml_lib.clusters.learn_cluster.modules.DenseModule import DenseModule

class LearnCluster(Root):
    defaults = {
        **Root.defaults,
        'module': DenseModule, 'module_kwargs': {}
    }
    
    def __init__(self, cluster_name, path_name = None, verbose = None,
                 module = None, module_kwargs = {}
                ):
        module = self.defaults['module'] if module is None else module
        module_kwargs = {**self.defaults['module_kwargs'], **module_kwargs}
        
        super().__init__(cluster_name, path_name = path_name, verbose = verbose)
        
        self.Module = module(path_name = '%s:%s' % (self.path_name, self.name), **module_kwargs)
        
    def enable(self, override = False):
        super().enable(override = override)
        self.Module.enable(self.input_count, override = override)
        
    def get_output_count(self, req_cluster_name):
        return self.Module.output_count
        
    def get_output_tensor(self, req_cluster_name):
        if self.enable is False: raise Exception('Cluster is not enabled!')
            
        if self.buffer is None:
            output_tensor = self.Module.process_tensor(self.input_tensor)
            self.buffer = output_tensor.clone()
        else:
            output_tensor = self.buffer
            self._v_msg('Retrieving output tensor from buffer.')
        
        return output_tensor
    
    def learn(self, loss):
        self.Module.learn(loss)