import numpy as np

from .RootLearnRate import RootLearnRate as Root

class ExpDecayLearnRate(Root):
    defaults = {
        **Root.defaults,
        'decay': 100
    }
    
    def __init__(self, path_name = None, verbose = None, seed_learn = None, decay = None):
        self.iters = 0
        self.decay = self.defaults['decay'] if decay is None else decay
        super().__init__(path_name = path_name, verbose = verbose, seed_learn = seed_learn)
    
    @property
    def learn_rate(self):
        learn_rate = self.seed_learn * np.exp(-self.decay * self.iters)
        self.iters += 1
        return learn_rate