import numpy as np
import torch as pt
import tqdm
import matplotlib.pyplot as plt

from ml_lib.utils.loss_combiners.MeanLossCombine import MeanLossCombine

class Controller:
    defaults = {
        'train_split': 'train',
        'use_tqdm': True,
        'loss_combiner': MeanLossCombine, 'loss_combiner_kwargs': {},
    }
    
    def __init__(self, control_name,
                 train_split = None, use_tqdm = None,
                 loss_combiner = None, loss_combiner_kwargs = {}
                ):
        loss_combiner = self.defaults['loss_combiner'] if loss_combiner is None else loss_combiner
        loss_combiner_kwargs = {**self.defaults['loss_combiner_kwargs'], **loss_combiner_kwargs}
        
        self.name = control_name
        self.clusters = {}
        self.train_split = self.defaults['train_split'] if train_split is None else train_split
        self.use_tqdm = self.defaults['use_tqdm'] if use_tqdm is None else use_tqdm
        self.enabled = False
        
        self.LossCombiner = loss_combiner(self.name, **loss_combiner_kwargs)
        
    def add_cluster(self, cluster):
        if cluster.name in self.clusters.keys():
            raise Exception('Cluster %s already in controller.' % cluster.name)
            
        cluster.path_name = self.name
        self.clusters[cluster.name] = cluster
        
    def link_clusters(self, from_cluster_name, to_cluster_name, **kwargs):
        self.clusters[from_cluster_name].add_link(self.clusters[to_cluster_name], 'output', **kwargs)
        self.clusters[to_cluster_name].add_link(self.clusters[from_cluster_name], 'input', **kwargs)
        
    def link_add(self, cluster, link_cluster_name, link_type, **kwargs):
        cluster_name = cluster.name
        self.add_cluster(cluster)
        
        if link_type == 'input':
            self.link_clusters(link_cluster_name, cluster_name, **kwargs)
        elif link_type == 'output':
            self.link_clusters(cluster_name, link_cluster_name, **kwargs)
        else:
            raise Exception('%s is not a valid link_type, must be either input or output.' % link_type)
            
    def enable_network(self):
        for cluster in self.clusters.values(): cluster.enable()
        self.loss_record = {}
            
    def build_batch_splits(self):
        for cluster in self.clusters.values(): cluster.build_batch_splits()

    @property
    def network_loss(self):
        network_loss = {}
        for cluster in self.clusters.values():
            loss_val = cluster.loss
            if loss_val is None: continue
            for key, value in loss_val.items():
                if key not in network_loss.keys():
                    network_loss[key] = []
                network_loss[key].append(value)
        
        for key in network_loss.keys():
            network_loss[key] = self.LossCombiner.loss_combine(pt.stack(network_loss[key]))
        
        return network_loss
    
    def clear_buffers(self):
        for cluster in self.clusters.values(): cluster.clear_buffer()
            
    def network_learn(self):
        network_loss = self.network_loss[self.train_split]
        for cluster in self.clusters.values(): cluster.learn(network_loss)
            
    def train_model(self, epocs):
        t = tqdm.tnrange(epocs) if self.use_tqdm else range(epocs)
        for epoc in t:
            self.build_batch_splits()
            
            postfix = {}
            for key, value in self.network_loss.items():
                if key not in self.loss_record: self.loss_record[key] = []
                self.loss_record[key].append(np.asscalar(value.detach().cpu().numpy()))
                postfix[key] = self.loss_record[key][-1]
            
            if self.use_tqdm: t.set_postfix(postfix)
            
            self.network_learn()
            self.clear_buffers()
            
    def plot_losses(self, figsize = (16, 10)):
        plt.figure(figsize = figsize)
        for key, value in self.loss_record.items():
            plt.plot(value, label = key)
        plt.legend()