import torch as pt

from ml_lib.clusters.learn_cluster.modules.learners.RootLearner import RootLearner as Root

class GradientLearner(Root):
    def learn(self, loss, coefs):
        (gradient,) = pt.autograd.grad(loss, coefs)
        with pt.no_grad():
            learn_step = gradient * self.learn_rate
            if self.coef_scale: learn_step /= coefs.size()[0]
            
        self._v_msg('Gradient Descent updated %s coefs, learn_rate %s.' % (new_coefs.numel(), self.learn_rate))
            
        return learn_step