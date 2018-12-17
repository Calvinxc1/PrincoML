import torch as pt

class Gradient():
    def __init__(self, learn_rate = 1, scale_features = False):
        self.learn_rate = learn_rate
        self.scale_features = scale_features
        
    def set_learn_rate(self, learn_rate):
        self.learn_rate = learn_rate
        
    def learn(self, loss, coefs):
        gradient = [self._clean_grad(grad) for grad in pt.autograd.grad(loss, coefs, retain_graph = True)]
        with pt.no_grad():
            new_coefs = [
                coef - (steps * self.learn_rate / coef.size()[0])
                for coef, steps in zip(coefs, gradient)
            ] if self.scale_features else [
                coef - (steps * self.learn_rate)
                for coef, steps in zip(coefs, gradient)
            ]
        return new_coefs
    
    def _clean_grad(self, grad):
        grad[grad != grad] = 0
        return grad
    
class Momentum(Gradient):
    def __init__(self, learn_rate = 1, scale_features = False, momentum = 0.8):
        super().__init__(learn_rate, scale_features = scale_features)
        self.momentum = momentum
        self.prior_steps = [0, 0]
        
    def learn(self, loss, coefs):
        gradient = [self._clean_grad(grad) for grad in pt.autograd.grad(loss, coefs, retain_graph = True)]
        with pt.no_grad():
            step_size = [
                (self.momentum * step) + ((1-self.momentum) * prior)
                for step, prior in zip(gradient, self.prior_steps)
            ]
            self.prior_steps = step_size
            new_coefs = [
                coef - (steps * self.learn_rate / coef.size()[0])
                for coef, steps in zip(coefs, step_size)
            ] if self.scale_features else [
                coef - (steps * self.learn_rate)
                for coef, steps in zip(coefs, step_size)
            ]
        return new_coefs
    
class AdaGrad(Gradient):
    def __init__(self, learn_rate = 1, scale_features = False, bump = 1e-16):
        super().__init__(learn_rate, scale_features = scale_features)
        self.bump = bump
        self.sq_grad_integral = None
        
    def learn(self, loss, coefs):
        gradient = [self._clean_grad(grad) for grad in pt.autograd.grad(loss, coefs, retain_graph = True)]
        with pt.no_grad():
            if self.sq_grad_integral is None: self.sq_grad_integral = [pt.zeros(grad.size()) for grad in gradient]
            self.sq_grad_integral = [
                prior + (grad ** 2)
                for prior, grad in zip(self.sq_grad_integral, gradient)
            ]
            
            step_size = [
                (self.learn_rate * grad) / pt.sqrt(grad_int + self.bump)
                for grad, grad_int in zip(gradient, self.sq_grad_integral)
            ]
            new_coefs = [
                coef - (step / coef.size()[0])
                for coef, step in zip(coefs, step_size)
            ] if self.scale_features else [
                coef - step
                for coef, step in zip(coefs, step_size)
            ]
        return new_coefs
    
class AdaDelta(Gradient):
    def __init__(self, learn_rate = 1, scale_features = False, bump = 1e-16, window_rate = 0.9):
        super().__init__(learn_rate, scale_features = scale_features)
        self.bump = bump
        self.sq_grad_integral = None
        self.sq_window_rate = window_rate
        
    def learn(self, loss, coefs):
        gradient = [self._clean_grad(grad) for grad in pt.autograd.grad(loss, coefs, retain_graph = True)]
        with pt.no_grad():
            if self.sq_grad_integral is None: self.sq_grad_integral = [pt.zeros(grad.size()) for grad in gradient]
            self.sq_grad_integral = [
                (self.window_rate * prior) + ((1-self.window_rate) * (grad ** 2))
                for prior, grad in zip(self.sq_grad_integral, gradient)
            ]
            
            step_size = [
                (self.learn_rate * grad) / pt.sqrt(grad_int + self.bump)
                for grad, grad_int in zip(gradient, self.sq_grad_integral)
            ]
            new_coefs = [
                coef - (step / coef.size()[0])
                for coef, step in zip(coefs, step_size)
            ] if self.scale_features else [
                coef - step
                for coef, step in zip(coefs, step_size)
            ]
        return new_coefs
    
class ADAM(Gradient):
    def __init__(self, learn_rate = 1, scale_features = False,
                 window_rate = 1e-1, sq_window_rate = 1e-3, bump = 1e-16,
                 noise_rate = 0, noise_coef = 0.55
                ):
        super().__init__(learn_rate, scale_features = scale_features)
        self.grad_integral = None
        self.sq_grad_integral = None
        self.iter_count = 1
        self.bump = bump
        self.window_rate = window_rate
        self.sq_window_rate = sq_window_rate
        self.noise_rate = noise_rate
        self.noise_coef = noise_coef
        
    def learn(self, loss, coefs):
        gradient = pt.autograd.grad(loss, coefs, retain_graph = True)
        gradient = [self._clean_grad(grad) for grad in gradient]
        with pt.no_grad():
            if self.noise_rate != 0:
                stdev = self.noise_rate / ((1 + self.iter_count) ** self.noise_coef)
                gradient = [grad + (pt.randn(grad.size()) * stdev) for grad in gradient]
            
            if self.grad_integral is None: self.grad_integral = [pt.zeros(grad.size()) for grad in gradient]
            self.grad_integral = [
                (self.window_rate * prior) + ((1-self.window_rate) * grad)
                for prior, grad in zip(self.grad_integral, gradient)
            ]
            
            if self.sq_grad_integral is None: self.sq_grad_integral = [pt.zeros(grad.size()) for grad in gradient]
            self.sq_grad_integral = [
                (self.window_rate * prior) + ((1-self.window_rate) * (grad ** 2))
                for prior, grad in zip(self.sq_grad_integral, gradient)
            ]
            
            step_size = [
                (
                    self.learn_rate * (grad_int / (1 - (self.window_rate ** self.iter_count)))
                ) / (
                    pt.sqrt(sq_grad_int / (1 - (self.sq_window_rate ** self.iter_count))) + self.bump
                )
                for grad_int, sq_grad_int in zip(self.grad_integral, self.sq_grad_integral)
            ]
            new_coefs = [
                coef - (step / coef.size()[0])
                for coef, step in zip(coefs, step_size)
            ] if self.scale_features else [
                coef - step
                for coef, step in zip(coefs, step_size)
            ]
            
            self.iter_count += 1
        return new_coefs