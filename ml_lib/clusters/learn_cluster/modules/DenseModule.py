import torch as pt

from ml_lib.clusters.learn_cluster.modules.RootModule import RootModule as Root
from ml_lib.utils.initialisers.FlatInit import FlatInit
from ml_lib.utils.initialisers.NormalInit import NormalInit
from ml_lib.clusters.learn_cluster.modules.combiners.SimpleCombine import SimpleCombine
from ml_lib.clusters.learn_cluster.modules.activators.LinearActivate import LinearActivate
from ml_lib.clusters.learn_cluster.modules.learners.GradientLearner import GradientLearner

class DenseModule(Root):
    name = 'dense_module'
    defaults = {
        **Root.defaults,
        'bias_init': FlatInit, 'bias_init_kwargs': {},
        'weight_init': NormalInit, 'weight_init_kwargs': {},
        'combiner': SimpleCombine, 'combiner_kwargs': {},
        'activator': LinearActivate, 'activator_kwargs': {},
        'learner': GradientLearner, 'learner_kwargs': {}
    }
    
    def __init__(self, path_name = None, verbose = None,
                 nodes = None,
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
        
        bias_init = self.defaults['bias_init'] if bias_init is None else bias_init
        bias_init_kwargs = {**self.defaults['bias_init_kwargs'], **bias_init_kwargs}
        
        weight_init = self.defaults['weight_init'] if weight_init is None else weight_init
        weight_init_kwargs = {**self.defaults['weight_init_kwargs'], **weight_init_kwargs}
        
        combiner = self.defaults['combiner'] if combiner is None else combiner
        combiner_kwargs = {**self.defaults['combiner_kwargs'], **combiner_kwargs}
        
        activator = self.defaults['activator'] if activator is None else activator
        activator_kwargs = {**self.defaults['activator_kwargs'], **activator_kwargs}
        
        learner = self.defaults['learner'] if learner is None else learner
        learner_kwargs = {**self.defaults['learner_kwargs'], **learner_kwargs}
        
        super().__init__(path_name = path_name, verbose = verbose)
        
        self.nodes = nodes
        
        self.Inits = {
            'bias': bias_init(path_name = '%s:%s' % (self.path_name, self.name), **bias_init_kwargs),
            'weight': weight_init(path_name = '%s:%s' % (self.path_name, self.name), **weight_init_kwargs)
        }
        self.Combiner = combiner(path_name = '%s:%s' % (self.path_name, self.name), **combiner_kwargs)
        self.Activator = activator(path_name = '%s:%s' % (self.path_name, self.name), **activator_kwargs)
        self.Learner = learner(path_name = '%s:%s' % (self.path_name, self.name), **learner_kwargs)
        
    def enable(self, input_count, override = False):
        self.coefs = pt.cat((
            self.Inits['bias'].init((1, self.nodes)),
            self.Inits['weight'].init((input_count, self.nodes))
        ), dim = 0)
        
        self.coefs.requires_grad = True
        
        self._v_msg('Generated coefficient tensor of shape %s' % (tuple([dim for dim in self.coefs.size()]),))
        
    def process_tensor(self, input_tensor):
        if self.enable is False: raise Exception('Module is not enabled!')
        
        combined_tensor = self.Combiner.combine(input_tensor, self.coefs)
        activated_tensor = self.Activator.activate(combined_tensor)
        return activated_tensor
    
    def learn(self, loss):
        new_coefs = self.Learner.learn(loss, self.coefs)
        self.coefs = new_coefs
        self.coefs.requires_grad = True