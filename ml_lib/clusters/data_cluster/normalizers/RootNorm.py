from datetime import datetime as dt

class RootNorm:
    defaults = {
        'path_name': 'N/A',
        'verbose': False
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
        
    def norm_data(self, data_frame):
        ## Define in child classes
        #norm_cols = self._norm_cols(data_frame)
        print(data_frame.head())
        return data_frame
        
    def _norm_cols(self, data_frame):
        norm_list = (data_frame.min() < 0) | (data_frame.max() > 1)
        norm_cols = norm_list.index[norm_list]
        return norm_cols