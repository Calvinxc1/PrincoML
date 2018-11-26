from copy import deepcopy as copy
import torch as pt

from ml_lib.clusters.root_cluster import RootCluster as Root
from ml_lib.utils.inits import Constant, Normal
from ml_lib.utils.combiners import Simple
from ml_lib.utils.activators import Linear
from ml_lib.utils.learners import Gradient

class DenseCluster(Root):
    def __init__(self, cluster_name, nodes,
                 weight_init = Normal, weight_init_params = {'mean': 0, 'stdev': 1},
                 bias_init = Constant, bias_init_params = {'constant': 0},
                 combiner = Simple, combiner_params = {},
                 activator = Linear, activator_params = {},
                 learner = Gradient, learner_params = {}
                ):
        super().__init__(cluster_name)
        
        self.nodes = nodes
        self.Inits = {
            'weights': weight_init(**weight_init_params),
            'bias': bias_init(**bias_init_params)
        }
        self.Combiner = combiner(**combiner_params)
        self.Activator = activator(**activator_params)
        self.Learner = learner(**learner_params)
        self.coefs = None
        
    def init_cluster(self, reinit = False, overwrite = False):
        super().init_cluster(reinit = reinit)
        self._init_coefs(overwrite)
        
    def _init_coefs(self, overwrite):
        if (self.coefs is not None) & (not overwrite):
            raise Exception('%s: Attempting to overwrite existing coefficents when overwrite is False' % self.name)
        
        init_coefs = {
            'weights': self.Inits['weights'].init((self.get_input_count(), self.nodes)),
            'bias': self.Inits['bias'].init((1, self.nodes))
        }
        self.coefs = init_coefs
        self.best_coefs = copy(self.coefs)
        
    def deinit_cluster(self):
        self._deinit_coefs()
        
    def _deinit_coefs(self):
        self.coefs = copy(self.best_coefs)
        
    def prime_cluster(self, reprime = False, **kwargs):
        super().prime_cluster(reprime = reprime)
        self._prime_coefs()
        
    def _prime_coefs(self):
        self.coefs['weights'].requires_grad = True
        self.coefs['bias'].requires_grad = True
        
    def get_output_count(self, req_cluster):
        return self.nodes
        
    def get_output_tensor(self, req_cluster, coef_override = None):
        coefs = self.coefs if coef_override is None else coef_override
        
        input_tensor = self.get_input_tensor()
        combined_tensor = self.Combiner.combine(input_tensor, coefs)
        activated_tensor = self.Activator.activate(combined_tensor)
        return activated_tensor
    
    def learn(self, loss, best_iter = False, coef_override = None):
        if best_iter:
            with pt.no_grad():
                self.best_coefs = copy(self.coefs)
        
        coefs = (self.coefs['weights'], self.coefs['bias']) if coef_override is None else coef_override
        self.update_coefs(loss, coefs)
    
    def update_coefs(self, loss, coefs):
        new_coefs = self.Learner.learn(loss, coefs)
        self.coefs['weights'] = new_coefs[0]
        self.coefs['bias'] = new_coefs[1]