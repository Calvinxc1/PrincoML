import numpy as np
import torch as pt

from .RootActivator import RootActivator as Root

class SoftMaxActivator(Root):
    def activate(self, input_tensor):
        activated_tensor = pt.softmax(input_tensor, dim = 1)
        self._v_msg('SoftMax Activation on %s shape tensor' % (tuple([dim for dim in activated_tensor.size()]),))
        return activated_tensor