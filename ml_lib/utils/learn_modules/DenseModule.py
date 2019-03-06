import torch as pt
from copy import deepcopy as copy

from ml_lib.utils.learn_modules.RootModule import RootModule as Root
from ml_lib.utils.initialisers.FlatInit import FlatInit
from ml_lib.utils.initialisers.NormalInit import NormalInit
from ml_lib.utils.learn_modules.combiners.SimpleCombine import SimpleCombine
from ml_lib.utils.learn_modules.activators.LinearActivate import LinearActivate
from ml_lib.utils.learn_modules.learners.GradientLearner import GradientLearner
from ml_lib.utils.learn_modules.learn_rates.FlatLearnRate import FlatLearnRate
from ml_lib.utils.learn_modules.learn_noise.RootLearnNoise import RootLearnNoise

class DenseModule(Root):
    name = 'dense_module'
    defaults = {
        **Root.defaults,
        'bias_active': True,
        'learn_rate': FlatLearnRate,
        'learn_noise': RootLearnNoise,
        'bias_init': FlatInit,
        'weight_init': NormalInit,
        'combiner': SimpleCombine,
        'activator': LinearActivate,
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
        self.coefs = pt.cat((
            self.Inits['bias'].init((1, self.nodes)),
            self.Inits['weight'].init((input_count, self.nodes))
        ), dim = 0) if self.bias_active else self.Inits['weight'].init((input_count, self.nodes))
        
        self.best_coefs = self.coefs.detach().clone()
        
        if self.nesterov: self.prior_coefs = copy(self.coefs)
        
        self.coefs.requires_grad = True
        
        self._v_msg('Generated coefficient tensor of shape %s' % (tuple([dim for dim in self.coefs.size()]),))
        
    def process_tensor(self, input_tensor):
        if self.enable is False: raise Exception('Module is not enabled!')
        
        combined_tensor = self.Combiner.combine(input_tensor, self.coefs)
        activated_tensor = self.Activator.activate(combined_tensor)
        return activated_tensor
    
    def learn(self, loss, best_iter = False):
        if best_iter:
            self.best_coefs = self.coefs.detach().clone()
        
        learn_step = self.Learner.learn(loss, self.coefs)
        with pt.no_grad():
            learn_noise = self.LearnNoise.gen_noise(self.coefs.size())
            if self.nesterov:
                self.coefs = self.prior_coefs - ((learn_step + learn_noise) * self.learn_rate)
                self.prior_coefs = copy(self.coefs)
                self.coefs = self.coefs - (learn_step * self.learn_rate)
            else:
                self.coefs = self.coefs - ((learn_step + learn_noise) * self.learn_rate)
        
        self.coefs.requires_grad = True
        
    def get_coefs(self, exempt_bias = False):
        if exempt_bias & self.bias_active:
            coefs = self.coefs[1:, :]
        else:
            coefs = self.coefs
        return coefs
    
    def lock_coefs(self):
        self.coefs = self.best_coefs.clone()
        self.coefs.requires_grad = True
        
    @property
    def learn_rate(self):
        return self.LearnRate.learn_rate