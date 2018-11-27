import numpy as np
import tqdm
import torch as pt
import matplotlib.pyplot as plt

from ml_lib.utils.loss_collapser import Linear

class Controller():
    def __init__(self, use_tqdm = True,
                 train_split = 'train', valid_split = 'train',
                 collapser = Linear, collapser_params = {}
                ):
        self.clusters = {}
        self.Collapser = collapser(**collapser_params)
        self.loss_record = {}
        self.train_split = train_split
        self.valid_split = valid_split
        self.best_loss = np.inf
        self.best_iter = None
        self.use_tqdm = use_tqdm
        
    def add_cluster(self, cluster):
        if cluster.name in self.clusters.keys():
            raise Exception('Cluster names %s already in model' % cluster.name)
            
        self.clusters[cluster.name] = cluster
        
    def link_clusters(self, from_cluster_name, to_cluster_name, **kwargs):
        self.clusters[from_cluster_name].add_link(self.clusters[to_cluster_name], 'output', **kwargs)
        self.clusters[to_cluster_name].add_link(self.clusters[from_cluster_name], 'input', **kwargs)
        
    def add_link(self, cluster, link_cluster_name, link_type, **kwargs):
        self.add_cluster(cluster)
        if link_type == 'output':
            self.link_clusters(cluster.name, link_cluster_name, **kwargs)
        elif link_type == 'input':
            self.link_clusters(link_cluster_name, cluster.name, **kwargs)
        else:
            raise Exception('%s is not a valid link type, use "input" or "output"' % link_type)
        
    def init_clusters(self, reinit = False):
        for cluster in self.clusters.values(): cluster.init_cluster(reinit = reinit)
            
    def train_model(self, iters):
        t = tqdm.tnrange(iters) if self.use_tqdm else range(iters)
        for epoc in t:
            self.learning_iter(epoc, t)
        
    def deinit_clusters(self):
        for cluster in self.clusters.values(): cluster.deinit_cluster()
            
    def learning_iter(self, epoc, t):
        losses = self.get_losses()
        
        if self.best_loss > losses[self.valid_split]:
            self.best_loss = losses[self.valid_split]
            self.best_iter = epoc
            best_iter = True
        else:
            best_iter = False
        
        for cluster in self.clusters.values():
            cluster.learn(losses[self.train_split], best_iter = best_iter)
            
        for loss_name, loss_value in losses.items():
            self.loss_record[loss_name] = self.loss_record.get(loss_name, [])
            self.loss_record[loss_name].append(loss_value.detach().cpu().numpy())
        
        if self.use_tqdm:
            postfix = {'best_iter': self.best_iter}
            for key, value in self.loss_record.items():
                postfix['loss_%s' % key] = value[-1]
            t.set_postfix(postfix)
            
    def get_losses(self):
        self.prime_clusters()
        
        losses = {}
        for cluster in self.clusters.values():
            cluster_losses = cluster.get_losses()
            if cluster_losses is not None:
                for loss_name, loss_val in cluster.get_losses():
                    losses[loss_name] = losses.get(loss_name, [])
                    losses[loss_name].append(
                        self.Collapser.collapse(loss_val)
                    )
        
        self.deprime_clusters()
        
        for key in losses.keys():
            losses[key] = self.Collapser.collapse(pt.stack(losses[key]))

        return losses
            
    def prime_clusters(self, reprime = False):
        for cluster in self.clusters.values(): cluster.prime_cluster(reprime = reprime)
            
    def deprime_clusters(self):
        for cluster in self.clusters.values(): cluster.deprime_cluster()
            
    def plot_losses(self, figsize = (16, 10)):
        plt.figure(figsize = figsize)
        
        for key, value in self.loss_record.items():
            plt.plot(value, label = key)
            
        plt.axvline(x = self.best_iter, label = 'best validation loss', color = 'red')
        
        plt.legend()
        
    def get_outputs(self, data_override = None):
        if data_override is not None:
            for cluster_name, cluster in self.clusters.items():
                cluster.prime_cluster(data_override = data_override.get(cluster_name, None))
        else:
            self.prime_clusters()
        
        outputs = {}
        for cluster_name, cluster in self.clusters.items():
            if type(cluster).__name__ != 'DataCluster':
                outputs[cluster_name] = cluster.get_output_tensor(self).detach().cpu().numpy()
        
        self.deprime_clusters()
        
        return outputs