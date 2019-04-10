from copy import deepcopy as copy
import torch as pt

from ml_lib.clusters.root_cluster import RootCluster as Root
from ml_lib.utils.inits import Constant
from ml_lib.utils.normalisers import MeanVar
from ml_lib.utils.learners import Gradient


class NormCluster(Root):
    def __init__(self, cluster_name,
                 weight_init=Constant, weight_init_params={'constant': 1},
                 bias_init=Constant, bias_init_params={'constant': 0},
                 normaliser=MeanVar, normaliser_params={},
                 learner=Gradient, learner_params={}
                 ):
        super().__init__(cluster_name)

        self.Inits = {
            'weight': weight_init(**weight_init_params),
            'bias': bias_init(**bias_init_params)
        }
        self.Normaliser = normaliser(**normaliser_params)
        self.Learner = learner(**learner_params)
        self.coefs = None

    def init_cluster(self, reinit=False, overwrite=False):
        super().init_cluster(reinit=reinit)
        self._init_coefs(overwrite)

    def _init_coefs(self, overwrite):
        if (self.coefs is not None) & (not overwrite):
            raise Exception(
                '%s: Attempting to overwrite existing coefficents when overwrite is False' % self.name)

        init_coefs = {
            'weight': self.Inits['weight'].init((1,)),
            'bias': self.Inits['bias'].init((1,))
        }
        self.coefs = init_coefs
        self.best_coefs = copy(self.coefs)

    def deinit_cluster(self):
        self._deinit_coefs()
        super().deinit_cluster()

    def _deinit_coefs(self):
        self.coefs = copy(self.best_coefs)

    def prime_cluster(self, reprime=False, **kwargs):
        super().prime_cluster(reprime=reprime)
        self._prime_coefs()

    def _prime_coefs(self):
        self.coefs['weight'].requires_grad = True
        self.coefs['bias'].requires_grad = True

    def get_output_count(self, req_cluster):
        return self.get_input_count()

    def get_output_tensor(self, req_cluster, coef_override=None):
        coefs = self.coefs if coef_override is None else coef_override

        input_tensor = self.get_input_tensor()
        normalised_tensor = self.Normaliser.norm(input_tensor, self.coefs)
        return normalised_tensor

    def learn(self, loss, best_iter=False, coef_override=None):
        if best_iter:
            with pt.no_grad():
                self.best_coefs = copy(self.coefs)

        coefs = (self.coefs['weight'], self.coefs['bias']
                 ) if coef_override is None else coef_override
        self.update_coefs(loss, coefs)

    def update_coefs(self, loss, coefs):
        new_coefs = self.Learner.learn(loss, coefs)
        self.coefs['weight'] = new_coefs[0]
        self.coefs['bias'] = new_coefs[1]

    def get_coefs(self):
        return self.coefs
