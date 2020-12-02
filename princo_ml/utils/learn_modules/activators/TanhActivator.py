import torch as pt

from .RootActivator import RootActivator as Root

class TanhActivator(Root):
    def activate(self, input_tensor):
        activated_tensor = pt.tanh(input_tensor)
        self._v_msg('TanH Activation on %s shape tensor' % (tuple([dim for dim in activated_tensor.size()]),))
        return activated_tensor