from PrincoML.utils.learn_modules.activators.RootActivate import RootActivate as Root

class LinearActivate(Root):
    def activate(self, input_tensor):
        self._v_msg('Linear Activation on %s shape tensor' % (tuple([dim for dim in input_tensor.size()]),))
        return input_tensor