from datetime import datetime as dt

class RootLearner:
    defaults = {
        'path_name': 'N/A',
        'verbose': False,
        'learn_rate': 1e-3,
        'coef_scale': True
    }
    
    def __init__(self, path_name = None, verbose = None, coef_scale = None):
        self.path_name = self.defaults['path_name'] if path_name is None else path_name
        self.verbose = self.defaults['verbose'] if verbose is None else verbose
        self.coef_scale = self.defaults['coef_scale'] if coef_scale is None else coef_scale
        
    def _v_msg(self, message):
        if self.verbose: print('%s | %s:learner (%s) - %s' % (
            dt.utcnow().isoformat(sep = ' '),
            self.path_name,
            type(self).__name__,
            message
        ))
            
    def learn(self, loss, coefs):
        ## Define in child classes
        pass