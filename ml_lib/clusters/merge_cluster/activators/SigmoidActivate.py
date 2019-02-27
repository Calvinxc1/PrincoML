import torch as pt

from ml_lib.clusters.learn_cluster.modules.activators.RootActivate import RootActivate as Root

class SigmoidActivate(Root):
    def activate(self, input_tensor):
        activated_tensor = pt.sigmoid(input_tensor)
        self._v_msg('Sigmoid Activation on %s shape tensor' % (tuple([dim for dim in activated_tensor.size()]),))
        return activated_tensor