import numpy as np


class All():
    split_names = ('train',)

    def __init__(self, index_len):
        self.update_splits(index_len)

    def update_splits(self, index_len):
        self.split_idxs = [np.arange(index_len)]

    @property
    def splits(self):
        splits = [
            (split_name, split_idx)
            for split_name, split_idx
            in zip(self.split_names, self.split_idxs)
        ]
        return splits


class Holdout(All):
    split_names = ('train', 'holdout')

    def update_splits(self, index_len, split=0.2):
        self.split_idxs = np.split(
            np.random.permutation(np.arange(index_len)),
            [int(index_len * (1 - split))]
        )


class Validation(All):
    split_names = ('train', 'valid', 'holdout')

    def update_splits(self, index_len, splits=(0.2, 0.2)):
        valid_obs = int(index_len * splits[0])
        holdout_obs = int(index_len * splits[1])
        self.split_idxs = np.split(
            np.random.permutation(np.arange(index_len)),
            [index_len - valid_obs - holdout_obs, index_len - holdout_obs]
        )


class CrossValid(All):
    def update_splits(self, index_len, splits, valid_seed=0):
        split_size = int(index_len * splits) + 1
        split_index = np.split(
            np.random.permutation(np.arange(index_len)),
            [split * split_size for split in range(splits)]
        )

        self.splits = {
            'cross_valid': split_index[:-1],
            'holdout': split_index[-1]
        }

        self.valid_split = valid_seed

    @property
    def splits(self):
        splits = [
            ('train', self.splits['cross_valid'][:self.valid_split] +
             self.splits['cross_valid'][self.valid_split+1:]),
            ('valid', self.splits['cross_valid'][self.valid_split]),
            ('holdout', self.splits['holdout'])
        ]
        return splits

    def increment():
        self.valid_split = (self.valid_split +
                            1) % len(self.splits['cross_valid'])
