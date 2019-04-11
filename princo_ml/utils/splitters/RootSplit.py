from datetime import datetime as dt

class RootSplit:
    defaults = {
        'path_name': 'N/A',
        'verbose': False
    }
    
    def __init__(self, split_tensor, path_name = None, verbose = None):
        self.path_name = self.defaults['path_name'] if path_name is None else path_name
        self.verbose = self.defaults['verbose'] if verbose is None else verbose
        
        self.gen_split(split_tensor)
        
    def _v_msg(self, message):
        if self.verbose: print('%s | %s:splitter (%s) - %s' % (
            dt.utcnow().isoformat(sep = ' '),
            self.path_name,
            type(self).__name__,
            message
        ))
        
    def gen_split(self, split_tensor):
        ## Define in child classes
        pass