from ml_lib.clusters.learn_cluster.modules.combiners.RootCombine import RootCombine as Root

class SimpleCombine(Root):
    def combine(self, input_tensor, coefs):
        combined_tensor = (input_tensor @ coefs[1:, :]) + coefs[0:1, :] if self.bias_active else input_tensor @ coefs
        
        self._v_msg('Combined to produce %s shape tensor' % (tuple([dim for dim in combined_tensor.size()]),))
        
        return combined_tensor