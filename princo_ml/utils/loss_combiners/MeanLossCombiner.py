from .RootLossCombiner import RootLossCombiner as Root

class MeanLossCombiner(Root):
    def loss_combine(self, loss_tensor):
        combined_loss = loss_tensor.mean()
        
        self._v_msg('Combined %s losses.' % loss_tensor.numel())
        
        return combined_loss