from .RootCluster import RootCluster as Root
from ..utils.learn_modules import DenseModule

class LearnCluster(Root):
    defaults = {
        **Root.defaults,
        'module': DenseModule, 'module_kwargs': {}
    }

    def __init__(self, cluster_name, path_name=None, verbose=None, reshape=None,
                 module=None, module_kwargs=None
                 ):
        module = self.defaults['module'] if module is None else module
        module_kwargs = self.defaults['module_kwargs'] if module_kwargs is None else module_kwargs

        super().__init__(cluster_name, path_name=path_name, verbose=verbose, reshape=reshape)

        self.Module = module(path_name='%s:%s' %
                             (self.path_name, self.name), **module_kwargs)

    def enable(self, override=False):
        super().enable(override=override)
        self.Module.enable(self.input_count, override=override)

    def get_output_count(self, req_cluster_name):
        return self.Module.output_count

    def load_output_tensor(self, req_cluster_name):
        if self.enable is False:
            raise Exception('Cluster is not enabled!')

        if self.buffer is None:
            output_tensor = self.Module.process_tensor(self.input_tensor)
            self.buffer = output_tensor.clone()
        else:
            output_tensor = self.buffer
            self._v_msg('Retrieving output tensor from buffer.')

        return output_tensor

    def learn(self, loss, best_iter=False):
        self.Module.learn(loss, best_iter=best_iter)

    def lock_coefs(self):
        self.Module.lock_coefs()

    def coefs(self, exempt_bias=False):
        return self.Module.get_coefs(exempt_bias=exempt_bias)

    def predict(self):
        predict_array = self.get_output_tensor(None).detach().cpu().numpy()
        return predict_array

    @property
    def learn_rate(self):
        return self.Module.learn_rate
