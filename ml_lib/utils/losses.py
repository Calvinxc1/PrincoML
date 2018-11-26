import numpy as np
import torch as pt

class SqErr():
    def __init__(self, root = True, percent = False,
                 collapser = pt.mean, collapser_kwargs = {'dim': 0}
                ):
        self.Collapser = collapser
        self.collapser_kwargs = collapser_kwargs
        self.percent = percent
        self.root = root
        
    def losses(self, predict_tensor, target_tensor, splits):
        loss_tensor = (target_tensor - predict_tensor) ** 2
        if self.percent: loss_tensor = loss_tensor / target_tensor
        
        loss_tensors = [
            (split_name, self.Collapser(loss_tensor[split_idx, :], **self.collapser_kwargs))
            for split_name, split_idx in splits
        ]
        
        if self.root: loss_tensors = [
            (split_name, pt.sqrt(split_data))
            for split_name, split_data in loss_tensors
        ]
        
        return loss_tensors
    
class AbsErr():
    def __init__(self, percent = False,
                 collapser = pt.mean, collapser_kwargs = {'dim': 0}
                ):
        self.precent = percent
        self.Collapser = collapser
        self.collapser_kwargs = collapser_kwargs
        
    def losses(self, predict_tensor, target_tensor, splits):
        loss_tensor = pt.abs(target_tensor - predict_tensor)
        if self.precent: loss_tensor = loss_tensor / target_tensor
        
        loss_tensors = [
            (split_name, self.Collapser(loss_tensor[split_idx, :], **self.collapser_kwargs))
            for split_name, split_idx in splits
        ]
        
        return loss_tensors
    
class LogLoss():
    def __init__(self, collapser = pt.mean, collapser_kwargs = {'dim': 0}, clamper = 1e-15):
        self.clamper = clamper
        self.Collapser = collapser
        self.collapser_kwargs = collapser_kwargs
        
    def losses(self, predict_tensor, target_tensor, splits):
        loss_tensor = pt.clamp((
            predict_tensor ** target_tensor
        ) * (
            (1 - predict_tensor) ** (1 - target_tensor)
        ), self.clamper, 1 - self.clamper)
        loss_tensor = -pt.log(loss_tensor)
        
        loss_tensors = [
            (split_name, self.Collapser(loss_tensor[split_idx, :], **self.collapser_kwargs))
            for split_name, split_idx in splits
        ]
        
        return loss_tensors