import torch as pt
import numpy as np

from ml_lib.clusters.root_cluster.RootCluster import RootCluster as Root
from ml_lib.clusters.data_cluster.normalizers.NormalNorm import NormalNorm
from ml_lib.clusters.data_cluster.splitters.BaseSplit import BaseSplit
from ml_lib.clusters.data_cluster.batchers.FlatBatch import FlatBatch
from ml_lib.clusters.data_cluster.losses.SqErrLoss import SqErrLoss
from ml_lib.utils.loss_combiners.MeanLossCombine import MeanLossCombine

class DataCluster(Root):
    defaults = {
        **Root.defaults,
        'normalizer': NormalNorm,
        'splitter': BaseSplit,
        'batcher': FlatBatch,
        'loss': SqErrLoss,
        'loss_combiner': MeanLossCombine,
    }
    
    def __init__(self, cluster_name, data_frame, path_name = None, verbose = None,
                 normalizer = None, normalizer_kwargs = {},
                 splitter = None, splitter_kwargs = {},
                 batcher = None, batcher_kwargs = {},
                 loss = None, loss_kwargs = {},
                 loss_combiner = None, loss_combiner_kwargs = {}
                ):
        normalizer = self.defaults['normalizer'] if normalizer is None else normalizer
        splitter = self.defaults['splitter'] if splitter is None else splitter
        batcher = self.defaults['batcher'] if batcher is None else batcher
        loss = self.defaults['loss'] if loss is None else loss
        loss_combiner = self.defaults['loss_combiner'] if loss_combiner is None else loss_combiner
        
        super().__init__(cluster_name, path_name = path_name, verbose = verbose)
        self.data = None
        
        self.Normalizer = normalizer(path_name = '%s:%s' % (self.path_name, self.name), **normalizer_kwargs)
        self.add_data(data_frame)
        
        self.Splitter = splitter(self.data['tensor'], path_name = '%s:%s' % (self.path_name, self.name), **splitter_kwargs)
        self.Batcher = batcher(path_name = '%s:%s' % (self.path_name, self.name), **batcher_kwargs)
        self.Loss = loss(path_name = '%s:%s' % (self.path_name, self.name), **loss_kwargs)
        self.LossCombiner = loss_combiner(path_name = '%s:%s' % (self.path_name, self.name), **loss_combiner_kwargs)
        
    def _link_cols(self, cluster_name, link_type):
        link_idx = self._link_idx(cluster_name, link_type)
        if link_idx is None:
            raise Exception('%s: Cluster %s not present in %s links.' % (self.name, cluster_name, link_type))
            
        link_cols = self.links[link_type][link_idx]['params']['columns']
        return link_cols
        
    def add_data(self, data_frame, overwrite = False):
        
        def convert_frame(data_frame):
            data_dict = {
                'tensor': pt.from_numpy(data_frame.values).type(pt.Tensor),
                'columns': data_frame.columns,
                'index': data_frame.index
            }

            return data_dict
        
        if (self.data is not None) & (not overwrite):
            raise Exception('%s: Attempting to overwrite existing data_frame when overwrite is False' % self.name)
            
        normed_data = self.Normalizer.norm_data(data_frame)
        self.data = convert_frame(normed_data)
        
        self._v_msg('Data frame added, overwrite %s.' % overwrite)
        
    def add_link(self, cluster, link_type, data_cols = None, **kwargs):
        if data_cols is None: raise Exception('data_cols kwarg cannot be empty.')        
        
        col_idx = np.array([self.data['columns'].get_loc(col) for col in data_cols])
        super().add_link(cluster, link_type)
        
        self.links[link_type][-1]['params']['columns'] = col_idx
        
    def get_output_count(self, req_cluster_name):
        link_cols = self._link_cols(req_cluster_name, 'output')
        output_count = len(link_cols)
        return output_count
    
    def build_batch_splits(self):
        self.batch_splits = self.Batcher.batch_obs(self.Splitter.splits)
        
    def get_output_tensor(self, req_cluster_name):
        link_cols = self._link_cols(req_cluster_name, 'output')
        output_tensor = self.data['tensor'][self.batch_tensor_idx, :][:, link_cols]
        
        self._v_msg('Output tensor shape %s provided.' % (tuple([dim for dim in output_tensor.size()]),))
        
        return output_tensor
    
    @property
    def batch_tensor_idx(self):
        batch_tensor_idx = np.concatenate([
            batch['index']
            for batch in self.batch_splits
        ])
        return batch_tensor_idx
    
    @property
    def target_tensor(self):
        link_cols = np.concatenate([link['params']['columns'] for link in self.links['input']])
        target_tensor = self.data['tensor'][:, link_cols]
        return target_tensor
    
    @property
    def loss(self):
        if self.buffer is None:
            loss_vals = {}
            run_idx = 0
            predict_tensor = self.input_tensor
            target_tensor = self.target_tensor
            for batch_split in self.batch_splits:
                self._v_msg('Building loss for %s.' % batch_split['name'])
                loss_tensor = self.Loss.loss(
                    target_tensor[batch_split['index'], :],
                    predict_tensor[run_idx:run_idx + batch_split['index'].size, :]
                )
                loss_val = self.LossCombiner.loss_combine(loss_tensor)
                loss_vals[batch_split['name']] = loss_val
                
                run_idx += batch_split['index'].size
            
            self.buffer = loss_vals
        else:
            loss_vals = self.buffer
            self._v_msg('Retrieving loss tensor from buffer.')
        
        return loss_vals