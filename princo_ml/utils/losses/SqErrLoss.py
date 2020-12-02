import torch as pt

from .RootLoss import RootLoss as Root

class SqErrLoss(Root):
    defaults = {
        **Root.defaults,
        'mean': True,
        'sqrt': True
    }
    
    def __init__(self, path_name = None, verbose = None,
                 mean = None, sqrt = None
                ):
        super().__init__(path_name = path_name, verbose = verbose)
        self.mean = self.defaults['mean'] if mean is None else mean
        self.sqrt = self.defaults['sqrt'] if sqrt is None else sqrt
        
    def loss(self, target_tensor, predict_tensor):
        loss_tensor = (target_tensor - predict_tensor) ** 2
        loss_tensor = loss_tensor.mean(dim = 0) if self.mean else loss_tensor.sum(dim = 0)
        
        if self.sqrt: loss_tensor = pt.sqrt(loss_tensor)
        
        self._v_msg('%s loss values generated.' % loss_tensor.numel())
        
        return loss_tensor