import torch as pt

from .RootActivator import RootActivator as Root

class SigmoidActivator(Root):
    def activate(self, input_tensor):
        activated_tensor = pt.sigmoid(input_tensor)
        self._v_msg('Sigmoid Activation on %s shape tensor' % (tuple([dim for dim in activated_tensor.size()]),))
        return activated_tensor