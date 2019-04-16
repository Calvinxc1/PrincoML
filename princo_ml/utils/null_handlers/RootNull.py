from datetime import datetime as dt

class RootNull:
    defaults = {
        'path_name': 'N/A',
        'verbose': False,
        'ignore_cols': []
    }
    
    def __init__(self, path_name = None, verbose = None):
        self.path_name = self.defaults['path_name'] if path_name is None else path_name
        self.verbose = self.defaults['verbose'] if verbose is None else verbose
        
    def _v_msg(self, message):
        if self.verbose: print('%s | %s:batcher (%s) - %s' % (
            dt.utcnow().isoformat(sep = ' '),
            self.path_name,
            type(self).__name__,
            message
        ))
        
    def init_nulls(self, dataframe):
        ## Define in child classes
        pass
    
    def col_mapper(self, col):
        ## Define in child classes
        pass