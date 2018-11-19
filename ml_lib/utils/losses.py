import torch as pt

class SqErr():
    def __init__(self, mean = True, root = True):
        self.mean = mean,
        self.root = root
        
    def loss(self, predict_tensor, target_tensor):
        loss_tensor = pt.mean((target_tensor - predict_tensor) ** 2, 0) if self.mean else pt.sum((target_tensor - predict_tensor) ** 2, 0)
        
        if self.root: loss_tensor = pt.sqrt(loss_tensor)
        
        return loss_tensor