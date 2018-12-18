import torch as pt

from ml_lib.utils.loss_combiners.MeanLossCombine import MeanLossCombine

class Controller:
    defaults = {
        'loss_combiner': MeanLossCombine, 'loss_combiner_kwargs': {}
    }
    
    def __init__(self, control_name,
                 loss_combiner = None, loss_combiner_kwargs = {}
                ):
        loss_combiner = self.defaults['loss_combiner'] if loss_combiner is None else loss_combiner
        loss_combiner_kwargs = {**self.defaults['loss_combiner_kwargs'], **loss_combiner_kwargs}
        
        self.name = control_name
        self.clusters = {}
        
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
            
    def build_batch_splits(self):
        for cluster in self.clusters.values(): cluster.build_batch_splits()
    
    @property
    def network_loss(self):
        cluster_losses = []
        for cluster in self.clusters.values():
            loss_val = cluster.loss
            if loss_val is None: continue
            cluster_losses.append(loss_val)
        cluster_losses = pt.stack(cluster_losses)
        
        network_loss = self.LossCombiner.loss_combine(cluster_losses)
        return network_loss
    
    def clear_buffers(self):
        for cluster in self.clusters.values(): cluster.clear_buffer()
            
    def network_learn(self, network_loss):
        for cluster in self.clusters.values(): cluster.learn(network_loss)