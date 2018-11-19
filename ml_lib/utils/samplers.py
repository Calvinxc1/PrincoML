import numpy as np

class All():
    def __init__(self, index_len):
        self.update_splits(index_len)
        
    def update_splits(self, index_len):
        self.index_len = index_len
    
    def sample(self, sample_type):
        sample = np.arange(self.index_len)
        return sample
    
class Holdout(All):
    def update_splits(self, index_len, split = 0.2):
        super().update_splits(index_len)
        
        split_index = np.split(
            np.random.permutation(np.arange(self.index_len)),
            [index_len * (1 - split)]
        )
        
        self.splits = {
            'train': split_index[0],
            'holdout': split_index[1]
        }
        
    def sample(self, sample_type):
        sample = self.splits[sample_type]
        return sample
    
class Validation(All):
    def update_splits(self, index_len, splits = (0.2, 0.2)):
        super().update_splits(index_len)
        
        valid_obs = int(self.index_len * splits[0])
        holdout_obs = int(self.index_len * splits[1])
        split_index = np.split(
            np.random.permutation(np.arange(self.index_len)),
            [self.index_len - valid_obs - holdout_obs, self.index_len - holdout_obs]
        )
        
        self.splits = {
            'train': split_index[0],
            'valid': split_index[1],
            'holdout': split_index[2]
        }
        
    def sample(self, sample_type):
        sample = self.splits[sample_type]
        return sample
    
class CrossValid(All):
    def update_splits(self, index_len, splits):
        super().update_splits(index_len)
        
        split_size = int(self.index_len * splits) + 1
        split_index = np.split(
            np.random.permutation(np.arange(self.index_len)),
            [split * split_size for split in range(splits)]
        )
        
        self.splits = {
            'cross_valid': split_index[:-1],
            'holdout': split_index[-1]
        }
        
        self.valid_split = 0
        
    def sample(self, sample_type, increment = False):
        if sample_type == 'train':
            sample = np.concatenate([
                self.splits['cross_valid'][split]
                for split in range(len(self.splits['cross_valid']))
                if split != self.valid_split
            ])
        elif sample_type == 'valid':
            sample = self.splits['cross_valid'][self.valid_split]
        elif sample_type == 'holdout':
            sample = self.splits['holdout']
        else:
            raise Exception('%s is an invalid sample type' % sample_type)
        
        if increment: self.valid_split = (self.valid_split + 1) % self.splits['cross_valid']
        
        return sample