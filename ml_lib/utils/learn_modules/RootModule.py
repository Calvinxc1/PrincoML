from datetime import datetime as dt

class RootModule:
    defaults = {
        'path_name': 'N/A',
        'verbose': False,
    }
    
    def __init__(self, path_name = None, verbose = None):
        self.path_name = self.defaults['path_name'] if path_name is None else path_name
        self.verbose = self.defaults['verbose'] if verbose is None else verbose
        
        self.enabled = False
        
    def _v_msg(self, message):
        if self.verbose: print('%s | %s:module (%s) - %s' % (
            dt.utcnow().isoformat(sep = ' '),
            self.path_name,
            type(self).__name__,
            message
        ))
            
    def enable(self, input_count, override = False):
        if self.enabled & (override is False):
            raise Exception('Module is already enabled and override is False.')
        
        self.enabled = True
        
    def learn(self, loss, best_iter = False):
        ## Define in child classes
        pass
    
    def get_coefs(self, exempt_bias = False):
        ## Define in child classes
        pass
    
    def lock_coefs(self):
        ## Define in child classes
        pass
    
    @property
    def learn_rate(self):
        return None