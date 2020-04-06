import torch as pt
from copy import deepcopy as copy

from .RootModule import RootModule as Root
from ..initialisers import FlatInit
from ..initialisers import NormalInit
from ..initialisers import HingeInit
from .combiners import HingeCombiner
from .activators import LinearActivator
from .learners import GradientLearner
from .learn_rates import FlatLearnRate
from .learn_noise.RootLearnNoise import RootLearnNoise

class DenseHingeModule(Root):
    name = 'dense_hinge_module'
    defaults = {
        **Root.defaults,
        'bias_active': True,
        'learn_rate': FlatLearnRate,
        'learn_noise': RootLearnNoise,
        'hinge_init': HingeInit,
        'bias_init': FlatInit,
        'weight_init': NormalInit,
        'combiner': HingeCombiner,
        'activator': LinearActivator,
        'learner': GradientLearner,
        'nesterov': False
    }
    
    def __init__(self, nodes, hinges, path_name = None, verbose = None,
                 nesterov = None, bias_active = None,
                 learn_rate = None, learn_rate_kwargs = {},
                 learn_noise = None, learn_noise_kwargs = {},
                 hinge_init = None, hinge_init_kwargs = {},
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
        hinge_init = self.defaults['hinge_init'] if hinge_init is None else hinge_init
        bias_init = self.defaults['bias_init'] if bias_init is None else bias_init
        weight_init = self.defaults['weight_init'] if weight_init is None else weight_init
        combiner = self.defaults['combiner'] if combiner is None else combiner
        activator = self.defaults['activator'] if activator is None else activator
        learner = self.defaults['learner'] if learner is None else learner
        
        super().__init__(path_name = path_name, verbose = verbose)
        
        self.nodes = nodes
        self.hinges = hinges
        self.coefs = None
        self.best_coefs = {}
        self.prior_coefs = {}
        self.nesterov = self.defaults['nesterov'] if nesterov is None else nesterov
        self.bias_active = self.defaults['bias_active'] if bias_active is None else bias_active
        
        self.LearnRate = learn_rate(path_name = '%s:%s' % (self.path_name, self.name), **learn_rate_kwargs)
        self.LearnNoise = learn_noise(path_name = '%s:%s' % (self.path_name, self.name), **learn_noise_kwargs)
        self.Inits = {
            'hinge': hinge_init(path_name = '%s:%s' % (self.path_name, self.name), **hinge_init_kwargs),
            'bias': bias_init(path_name = '%s:%s' % (self.path_name, self.name), **bias_init_kwargs),
            'weight': weight_init(path_name = '%s:%s' % (self.path_name, self.name), **weight_init_kwargs)
        }
        self.Combiner = combiner(path_name = '%s:%s' % (self.path_name, self.name), **combiner_kwargs)
        self.Activator = activator(path_name = '%s:%s' % (self.path_name, self.name), **activator_kwargs)
        self.Learner = learner(path_name = '%s:%s' % (self.path_name, self.name), **learner_kwargs)
        
    @property
    def output_count(self):
        return self.nodes
        
    def save_coefs(self, coefs):
        for key, value in coefs.items():
            self.best_coefs[key] = value.detach().clone()
            
    def activate_grad(self):
        for coef in self.coefs.values():
            coef.requires_grad = True
        
    def process_tensor(self, input_tensor):
        if self.enable is False: raise Exception('Module is not enabled!')
            
        if self.coefs is None: self.enable_coefs(input_tensor)
        
        combined_tensor = self.Combiner.combine(input_tensor, self.coefs)
        activated_tensor = self.Activator.activate(combined_tensor)
        return activated_tensor
    
    def enable_coefs(self, input_tensor):
        input_count = input_tensor.size()[1]
        
        self.coefs = {
            'bias': self.Inits['bias'].init((self.nodes,)),
            'weights': self.Inits['weight'].init((input_count, self.nodes)),
            'hinge': self.Inits['hinge'].init(input_tensor, self.hinges),
            'hinge_bias': pt.stack([
                self.Inits['bias'].init((input_count, self.nodes))
                for _ in range(self.hinges+1)
            ], dim = 2),
            'hinge_weights': pt.stack([
                self.Inits['weight'].init((input_count, self.nodes))
                for _ in range(self.hinges+1)
            ], dim = 2)
        }
        
        self.save_coefs(self.coefs)
        
        if self.nesterov:
            for key, value in self.coefs.items():
                self.prior_coefs[key] = value.clone()
        
        self.activate_grad()
    
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
            coefs = {}
            for key, value in self.coefs.items():
                if key == 'bias': continue
                coefs[key] = value
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