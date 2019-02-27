from datetime import datetime as dt

class RootCombine:
    defaults = {
        'path_name': 'N/A',
        'verbose': False,
        'bias_active': True
    }
    
    def __init__(self, path_name = None, verbose = None, bias_active = None):
        self.path_name = self.defaults['path_name'] if path_name is None else path_name
        self.verbose = self.defaults['verbose'] if verbose is None else verbose
        self.bias_active = self.defaults['bias_active'] if bias_active is None else bias_active
        
    def _v_msg(self, message):
        if self.verbose: print('%s | %s:combiner (%s) - %s' % (
            dt.utcnow().isoformat(sep = ' '),
            self.path_name,
            type(self).__name__,
            message
        ))
            
    def combine(self, input_tensor, coefs):
        ## Define in child classes
        pass