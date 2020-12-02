import torch as pt

from .RootCluster import RootCluster as Root
from ..utils.mergers import AddMerger
from ..utils.learn_modules.activators import LinearActivator

class MergeCluster(Root):
    defaults = {
        **Root.defaults,
        'merger': AddMerger, 'merger_kwargs': {},
        'activator': LinearActivator, 'activator_kwargs': {}
    }

    def __init__(self, cluster_name, path_name=None, verbose=None, reshape=None,
                 merger=None, merger_kwargs=None,
                 activator=None, activator_kwargs=None
                 ):
        merger = self.defaults['merger'] if merger is None else merger
        merger_kwargs = self.defaults['merger_kwargs'] if merger_kwargs is None else merger_kwargs

        activator = self.defaults['activator'] if activator is None else activator
        activator_kwargs = self.defaults['activator_kwargs'] if activator_kwargs is None else activator_kwargs

        super().__init__(cluster_name, path_name=path_name, verbose=verbose, reshape=reshape)

        self.Merger = merger(path_name='%s:%s' %
                             (self.path_name, self.name), **merger_kwargs)
        self.Activator = activator(path_name='%s:%s' % (
            self.path_name, self.name), **activator_kwargs)

    @property
    def input_count(self):
        input_count = max([
            link['cluster'].get_output_count(self.name)
            for link in self.links['input']
        ])
        return input_count

    @property
    def input_tensor(self):
        input_tensor = []
        for link in self.links['input']:
            link_tensor = link['cluster'].get_output_tensor(self.name)
            link_len = len(link_tensor.size())
            if link_len == 2:
                input_tensor.append(link_tensor.view(*link_tensor.size(), 1))
            elif link_len == 3:
                input_tensor.append(link_tensor)
            else:
                raise Exception(
                    'link_tensor has %s dimensions, can only support 2 or 3 dimension inputs' % link_len)

        input_tensor = pt.cat(input_tensor, dim=2)

        return input_tensor

    def get_output_count(self, req_cluster_name):
        return self.input_count

    def load_output_tensor(self, req_cluster_name):
        if self.enable is False:
            raise Exception('Cluster is not enabled!')

        if self.buffer is None:
            merged_tensor = self.Merger.merge(self.input_tensor)
            output_tensor = self.Activator.activate(merged_tensor)
            self.buffer = output_tensor.clone()
        else:
            output_tensor = self.buffer
            self._v_msg('Retrieving output tensor from buffer.')

        return output_tensor

    def predict(self):
        predict_array = self.get_output_tensor(None).detach().cpu().numpy()
        return predict_array
