from .RootActivator import RootActivator as Root

class LinearActivator(Root):
    def activate(self, input_tensor):
        self._v_msg('Linear Activation on %s shape tensor' % (tuple([dim for dim in input_tensor.size()]),))
        return input_tensor