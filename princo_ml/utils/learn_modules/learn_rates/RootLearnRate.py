from datetime import datetime as dt

class RootLearnRate:
    defaults = {
        'path_name': 'N/A',
        'verbose': False,
        'seed_learn': 1e-3
    }
    
    def __init__(self, path_name = None, verbose = None, seed_learn = None):
        self.path_name = self.defaults['path_name'] if path_name is None else path_name
        self.verbose = self.defaults['verbose'] if verbose is None else verbose
        self.seed_learn = self.defaults['seed_learn'] if seed_learn is None else seed_learn
        
    def _v_msg(self, message):
        if self.verbose: print('%s | %s:learner (%s) - %s' % (
            dt.utcnow().isoformat(sep = ' '),
            self.path_name,
            type(self).__name__,
            message
        ))
    
    @property
    def learn_rate(self):
        ## Define in child classes
        pass