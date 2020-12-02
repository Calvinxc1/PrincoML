import torch as pt
from copy import deepcopy as copy

from .RootModule import RootModule as Root
from ..initialisers import FlatInit
from ..initialisers import NormalInit
from .combiners import SimpleCombiner
from .activators import LinearActivator
from .learners import GradientLearner
from .learn_rates import FlatLearnRate
from .learn_noise.RootLearnNoise import RootLearnNoise

class DenseModule(Root):
    name = 'dense_module'
    defaults = {
        **Root.defaults,
        'bias_active': True,
        'learn_rate': FlatLearnRate,
        'learn_noise': RootLearnNoise,
        'bias_init': FlatInit,
        'weight_init': NormalInit,
        'combiner': SimpleCombiner,
        'activator': LinearActivator,
        'learner': GradientLearner,
        'nesterov': False
    }
    
    def __init__(self, path_name = None, verbose = None,
                 nodes = None, nesterov = None,
                 bias_active = None,
                 learn_rate = None, learn_rate_kwargs = {},
                 learn_noise = None, learn_noise_kwargs = {},
                 bias_init = None, bias_init_kwargs = {},
                 weight_init = None, weight_init_kwargs = {},
                 combiner = None, combiner_kwargs = {},
                 activator = None, activator_kwargs = {},
                 learner = None, learner_kwargs = {}
                ):
        if type(nodes) is not int:
            raise Exception('Kwarg nodes must be integer number greater than 0.')
        elif nodes <= 0:
            raise Exception('Kwarg nodes must be integer number greater than 0.')
        
        learn_rate = self.defaults['learn_rate'] if learn_rate is None else learn_rate
        learn_noise = self.defaults['learn_noise'] if learn_noise is None else learn_noise
        bias_init = self.defaults['bias_init'] if bias_init is None else bias_init
        weight_init = self.defaults['weight_init'] if weight_init is None else weight_init
        combiner = self.defaults['combiner'] if combiner is None else combiner
        activator = self.defaults['activator'] if activator is None else activator
        learner = self.defaults['learner'] if learner is None else learner
        super().__init__(path_name = path_name, verbose = verbose)
        
        self.nodes = nodes
        self.best_coefs = {}
        self.prior_coefs = {}
        self.nesterov = self.defaults['nesterov'] if nesterov is None else nesterov
        self.bias_active = self.defaults['bias_active'] if bias_active is None else bias_active
        
        self.LearnRate = learn_rate(path_name = '%s:%s' % (self.path_name, self.name), **learn_rate_kwargs)
        self.LearnNoise = learn_noise(path_name = '%s:%s' % (self.path_name, self.name), **learn_noise_kwargs)
        self.Inits = {
            'bias': bias_init(path_name = '%s:%s' % (self.path_name, self.name), **bias_init_kwargs),
            'weight': weight_init(path_name = '%s:%s' % (self.path_name, self.name), **weight_init_kwargs)
        }
        self.Combiner = combiner(path_name = '%s:%s' % (self.path_name, self.name), bias_active = self.bias_active, **combiner_kwargs)
        self.Activator = activator(path_name = '%s:%s' % (self.path_name, self.name), **activator_kwargs)
        self.Learner = learner(path_name = '%s:%s' % (self.path_name, self.name), **learner_kwargs)
    
    @property
    def output_count(self):
        return self.nodes
        
    def enable(self, input_count, override = False):
        self.coefs = {
            'weight': self.Inits['weight'].init((input_count, self.nodes))
        }
        if self.bias_active: self.coefs['bias'] = self.Inits['bias'].init((self.nodes,))
        
        self.save_coefs(self.coefs)
        
        if self.nesterov:
            for key, value in self.coefs.items():
                self.prior_coefs[key] = value.clone()
        
        self.activate_grad()
        
        #self._v_msg('Generated coefficient tensor of shape %s' % (tuple([dim for dim in self.coefs.size()]),))
        
    def save_coefs(self, coefs):
        for key, value in coefs.items():
            self.best_coefs[key] = value.detach().clone()
            
    def activate_grad(self):
        for coef in self.coefs.values():
            coef.requires_grad = True
        
    def process_tensor(self, input_tensor):
        if self.enable is False: raise Exception('Module is not enabled!')
        
        combined_tensor = self.Combiner.combine(input_tensor, self.coefs)
        activated_tensor = self.Activator.activate(combined_tensor)
        return activated_tensor
    
    def learn(self, loss, best_iter = False):
        if best_iter: self.save_coefs(self.coefs)
        
        learn_step = self.Learner.learn(loss, self.coefs)
        with pt.no_grad():
            for coef_name, coef_step in learn_step.items():
                noise = self.LearnNoise.gen_noise(coef_step.size())
                if self.nesterov:
                    self.coefs[coef_name] = self.prior_coefs[coef_name] - ((coef_step + noise) * self.learn_rate)
                    self.prior_coefs[coef_name] = copy(self.coefs[coef_name])
                    self.coefs[coef_name] = self.coefs[coef_name] - (coef_step * self.learn_rate)
                else:
                    self.coefs[coef_name] = self.coefs[coef_name] - ((coef_step + noise) * self.learn_rate)
        
        self.activate_grad()
        
    def get_coefs(self, exempt_bias = False):
        if exempt_bias & self.bias_active:
            coefs = {'weight': self.coefs['weight']}
        else:
            coefs = self.coefs
        return coefs
    
    def lock_coefs(self):
        for coef_name, coef in self.best_coefs.items():
            self.coefs[coef_name] = coef.clone()
        self.activate_grad()
        
    @property
    def learn_rate(self):
        return self.LearnRate.learn_rate