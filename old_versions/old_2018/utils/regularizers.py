import torch as pt


class Lasso:
    def __init__(self, l1=1e-2):
        self.l1 = l1

    def regularize(self, coefs):
        reg_coefs = self.l1 * pt.sum(pt.abs(coefs))
        return reg_coefs


class Ridge:
    def __init__(self, l2=1e-2):
        self.l1 = l2

    def regularize(self, coefs):
        reg_coefs = self.l2 * pt.sum(coefs ** 2)
        return reg_coefs


class LassoRidge:
    def __init__(self, l1=1e-2, l2=1e-2):
        self.l1 = l1
        self.l2 = l2

    def regularize(self, coefs):
        reg_coefs = (self.l1 * pt.sum(pt.abs(coefs))) + \
            (self.l2 * pt.sum(coefs ** 2))
        return reg_coefs
