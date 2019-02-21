import torch as pt

from ml_lib.clusters.data_cluster.losses.RootLoss import RootLoss as Root

class CrossEntLoss(Root):
    defaults = {
        **Root.defaults,
        'mean': False,
        'clamper': 1e-15
    }
    
    def __init__(self, path_name = None, verbose = None,
                 mean = None, clamper = None
                ):
        super().__init__(path_name = path_name, verbose = verbose)
        self.mean = self.defaults['mean'] if mean is None else mean
        self.clamper = self.defaults['clamper'] if clamper is None else clamper
        
    def loss(self, target_tensor, predict_tensor):
        loss_tensor = -(
            target_tensor * pt.log(pt.clamp(predict_tensor, self.clamper, 1 - self.clamper))
        ) - (
            (1 - target_tensor) * pt.log(pt.clamp(1 - predict_tensor, self.clamper, 1 - self.clamper))
        )

        loss_tensor = loss_tensor.mean(dim = 0) if self.mean else loss_tensor.sum(dim = 0)
        
        self._v_msg('%s Negative Cross Entropy loss values generated.' % loss_tensor.numel())
        
        return loss_tensor