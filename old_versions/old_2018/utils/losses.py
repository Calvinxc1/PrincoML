import numpy as np
import torch as pt


class SqErr():
    def __init__(self, root=True, percent=False,
                 collapser=pt.mean, collapser_kwargs={'dim': 0}
                 ):
        self.Collapser = collapser
        self.collapser_kwargs = collapser_kwargs
        self.percent = percent
        self.root = root

    def losses(self, predict_tensor, target_tensor, splits):
        loss_tensor = (target_tensor - predict_tensor) ** 2
        if self.percent:
            loss_tensor = loss_tensor / target_tensor

        loss_tensors = self.split_losses(loss_tensor, splits)

        if self.root:
            loss_tensors = [
                (split_name, pt.sqrt(split_data))
                for split_name, split_data in loss_tensors
            ]

        return loss_tensors

    def split_losses(self, loss_tensor, splits):
        split_losses = [
            (split_name, self.Collapser(
                loss_tensor[split_idx, :], **self.collapser_kwargs))
            for split_name, split_idx in splits
        ]
        return split_losses


class AbsErr(SqErr):
    def __init__(self, percent=False,
                 collapser=pt.mean, collapser_kwargs={'dim': 0}
                 ):
        self.precent = percent
        self.Collapser = collapser
        self.collapser_kwargs = collapser_kwargs

    def losses(self, predict_tensor, target_tensor, splits):
        loss_tensor = pt.abs(target_tensor - predict_tensor)
        if self.precent:
            loss_tensor = loss_tensor / target_tensor

        loss_tensors = self.split_losses(loss_tensor, splits)

        return loss_tensors


class LogLoss(SqErr):
    def __init__(self, clamper=1e-15,
                 collapser=pt.mean, collapser_kwargs={'dim': 0}
                 ):
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

        loss_tensors = self.split_losses(loss_tensor, splits)

        return loss_tensors


class Huber(SqErr):
    def __init__(self, steepness=1,
                 collapser=pt.mean, collapser_kwargs={'dim': 0}
                 ):
        self.steepness = steepness
        self.Collapser = collapser
        self.collapser_kwargs = collapser_kwargs

    def losses(self, predict_tensor, target_tensor, splits):
        loss_tensor = predict_tensor - target_tensor
        loss_tensor = (self.steepness ** 2) * \
            (pt.sqrt(1 + ((loss_tensor / self.steepness) ** 2)) - 1)

        loss_tensors = self.split_losses(loss_tensor, splits)
        return loss_tensors
