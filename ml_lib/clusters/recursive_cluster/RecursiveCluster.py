from ml_lib.clusters.root_cluster.RootCluster import RootCluster as Root

class RecursiveCluster(Root):
    defaults = {
        **Root.defaults
    }
    
    def __init__(self, cluster_name, path_name = None, verbose = None
                ):
        super().__init__(cluster_name, path_name = path_name, verbose = verbose)
        
    def enable(self, override = False):
        super().enable(override = override)
        
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
        
    def coefs(self, exempt_bias = False):
        return self.Module.get_coefs(exempt_bias = exempt_bias)