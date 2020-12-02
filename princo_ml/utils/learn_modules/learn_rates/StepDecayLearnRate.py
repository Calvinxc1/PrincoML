import numpy as np

from .RootLearnRate import RootLearnRate as Root

class StepDecayLearnRate(Root):
    defaults = {
        **Root.defaults,
        'drop': 1e-1,
        'decay': 10
    }
    
    def __init__(self, path_name = None, verbose = None, seed_learn = None, decay = None, drop = None):
        self.iters = 0
        self.decay = self.defaults['decay'] if decay is None else decay
        self.drop = self.defaults['drop'] if drop is None else drop
        super().__init__(path_name = path_name, verbose = verbose, seed_learn = seed_learn)
    
    @property
    def learn_rate(self):
        learn_rate = self.seed_learn * (self.drop ** np.floor(self.iters / self.decay))
        self.iters += 1
        return learn_rate