from datetime import datetime as dt

class RootNorm:
    defaults = {
        'path_name': 'N/A',
        'verbose': False,
        'ignore_cols': []
    }
    
    def __init__(self, path_name = None, verbose = None, ignore_cols = None):
        self.path_name = self.defaults['path_name'] if path_name is None else path_name
        self.verbose = self.defaults['verbose'] if verbose is None else verbose
        self.ignore_cols = self.defaults['ignore_cols'] if ignore_cols is None else ignore_cols
        
    def _v_msg(self, message):
        if self.verbose: print('%s | %s:batcher (%s) - %s' % (
            dt.utcnow().isoformat(sep = ' '),
            self.path_name,
            type(self).__name__,
            message
        ))
        
    def norm_data(self, data_frame):
        ## Define in child classes
        #norm_cols = self._norm_cols(data_frame)
        print(data_frame.head())
        return data_frame
        
    def _norm_cols(self, data_frame):
        norm_list = (data_frame.min() < 0) | (data_frame.max() > 1)
        norm_cols = norm_list.index[norm_list]
        norm_cols = [col for col in norm_cols if col not in self.ignore_cols]
        return norm_cols