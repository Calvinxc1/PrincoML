import numpy as np

class RootMerger:
    defaults = {
        'path_name': 'N/A',
        'verbose': False,
        'inputs': np.inf
    }
    
    def __init__(self, path_name = None, verbose = None):
        self.path_name = self.defaults['path_name'] if path_name is None else path_name
        self.verbose = self.defaults['verbose'] if verbose is None else verbose
        
    def _v_msg(self, message):
        if self.verbose: print('%s | %s:activator (%s) - %s' % (
            dt.utcnow().isoformat(sep = ' '),
            self.path_name,
            type(self).__name__,
            message
        ))
        
    def merge(self, input_tensor):
        if self.defaults['inputs'] != np.inf:
            if input_tensor.size()[2] != self.defaults['inputs']:
                raise Exception('Merger has a requirement of %s inputs. %s inputs provided' % (input_tensor.size()[2], self.defaults['inputs']))
                
        merge_tensor = self.merge_process(input_tensor)
        return merge_tensor
    
    def merge_process(self, input_tensor):
        ## define in child classes
        pass