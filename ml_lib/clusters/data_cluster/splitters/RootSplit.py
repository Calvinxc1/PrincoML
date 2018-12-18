from datetime import datetime as dt

class RootSplit:
    defaults = {
        'path_name': 'N/A',
        'verbose': False
    }
    
    def __init__(self, split_idx, path_name = None, verbose = None):
        self.path_name = self.defaults['path_name'] if path_name is None else path_name
        self.verbose = self.defaults['verbose'] if verbose is None else verbose
        
        self.gen_split(split_idx)
        
    def _v_msg(self, message):
        if self.verbose: print('%s | %s:splitter (%s) - %s' % (
            dt.utcnow().isoformat(sep = ' '),
            self.path_name,
            type(self).__name__,
            message
        ))
        
    def gen_split(self, split_idx):
        ## Define in child classes
        pass