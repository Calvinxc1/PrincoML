from PrincoML.utils.loss_combiners.RootLossCombine import RootLossCombine as Root

class MeanLossCombine(Root):
    def loss_combine(self, loss_tensor):
        combined_loss = loss_tensor.mean()
        
        self._v_msg('Combined %s losses.' % loss_tensor.numel())
        
        return combined_loss