from .RootNorm import RootNorm as Root

class NormalNorm(Root):
    def __init__(self, path_name = None, verbose = None, ignore_cols = None):
        super().__init__(path_name = path_name, verbose = verbose, ignore_cols = ignore_cols)
        self.mean = None
        self.stdev = None
        self.cols = None
    
    def norm_data(self, data_frame):
        if (self.mean is not None) | (self.stdev is not None) | (self.cols is not None):
            raise Exception('Normalizer already set, cannot override')
        
        frame_copy = data_frame.copy()
        
        self.cols = self._norm_cols(data_frame)
        
        self.mean = data_frame[self.cols].mean()
        self.stdev = data_frame[self.cols].std()
        
        frame_copy[self.cols] = (data_frame[self.cols] - self.mean) / self.stdev
        return frame_copy
    
    def denorm_data(self, data_frame):
        frame_copy = data_frame.copy()
        
        frame_copy[self.cols] = (data_frame[self.cols] * self.stdev) + self.mean
        return frame_copy
    
    def norm_new_data(self, data_frame):
        frame_copy = data_frame.copy()
        frame_copy[self.cols] = (data_frame[self.cols] - self.mean) / self.stdev
        return frame_copy