import numpy as np
import torch as pt

from ml_lib.clusters.root_cluster import RootCluster as Root
from ml_lib.utils.samplers import All
from ml_lib.utils.losses import SqErr

class DataCluster(Root):
    def __init__(self, cluster_name, data_frame,
                 sampler = All, sampler_params = {},
                 loss = SqErr, loss_params = {}
                ):
        super().__init__(cluster_name)
        self.data = None
        self.add_data(data_frame)
        self.Sampler = sampler(self.data['index'].size, **sampler_params)
        self.Loss = loss(**loss_params)
        
    def add_data(self, data_frame, overwrite = False):
        if (self.data is not None) & (not overwrite):
            raise Exception('%s: Attempting to overwrite existing data_frame when overwrite is False' % self.name)
            
        self.data = {
            'tensor': pt.from_numpy(data_frame.values).type(pt.Tensor),
            'columns': data_frame.columns,
            'index': data_frame.index
        }
        
    def add_link(self, cluster, link_type, data_cols = None, **kwargs):
        col_idx = np.array([self.data['columns'].get_loc(col) for col in data_cols])
        super().add_link(cluster, link_type)
        
        self.links[link_type][-1]['params']['columns'] = col_idx
        
    def get_link_cols(self, cluster_name, link_type):
        link_idx = self.get_link_idx(cluster_name, link_type)
        if link_idx is None:
            raise Exception('%s: Cluster %s not present in %s link' % (self.name, cluster_name, link_type))
            
        link_cols = self.links[link_type][link_idx]['params']['columns']
        return link_cols
        
    def get_output_count(self, req_cluster):
        output_count = len(self.get_link_cols(req_cluster.name, 'output'))
        return output_count
    
    def get_output_tensor(self, req_cluster):
        link_cols = self.get_link_cols(req_cluster.name, 'output')
        output_data = self.data['tensor'][:, link_cols]
        return output_data
    
    def get_loss(self):
        predict_tensor = self.get_input_tensor()
        target_tensor = self.get_target_tensor()
        loss_vals = self.Loss.loss(predict_tensor, target_tensor)
        return loss_vals
    
    def get_target_tensor(self):
        col_idx = []
        for link_item in self.links['input']:
            col_idx.extend(link_item['params']['columns'])
            
        target_tensor = self.data['tensor'][:, col_idx]
        return target_tensor