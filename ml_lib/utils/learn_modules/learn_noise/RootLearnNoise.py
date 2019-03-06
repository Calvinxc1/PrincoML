from datetime import datetime as dt

class RootLearnNoise:
    defaults = {
        'path_name': 'N/A',
        'verbose': False
    }
    
    def __init__(self, path_name = None, verbose = None, seed_learn = None):
        self.path_name = self.defaults['path_name'] if path_name is None else path_name
        self.verbose = self.defaults['verbose'] if verbose is None else verbose
        
    def _v_msg(self, message):
        if self.verbose: print('%s | %s:learner (%s) - %s' % (
            dt.utcnow().isoformat(sep = ' '),
            self.path_name,
            type(self).__name__,
            message
        ))
    
    def gen_noise(self, noise_shape):
        ## Define in child classes
        return 0