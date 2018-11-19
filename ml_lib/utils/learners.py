import torch as pt

class Gradient():
    def __init__(self, learn_rate = 1):
        self.learn_rate = learn_rate
        
    def learn(self, loss, coefs):
        gradients = pt.autograd.grad(loss, coefs, retain_graph = True)
        with pt.no_grad():
            new_coefs = [coefs - (grads * self.learn_rate) for coefs, grads in zip(coefs, gradients)]
        return new_coefs