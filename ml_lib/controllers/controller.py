import tqdm
import torch as pt

from ml_lib.utils.loss_collapser import Linear

class Controller():
    def __init__(self, use_tqdm = True,
                 collapser = Linear, collapser_params = {}
                ):
        self.clusters = {}
        self.Collapser = collapser(**collapser_params)
        self.loss_record = []
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
            self.learning_iter(t)
            
    def learning_iter(self, t):
        loss = self.get_loss()
        for cluster in self.clusters.values():
            cluster.learn(loss)
            
        self.loss_record.append(loss.detach().cpu().numpy())
        if self.use_tqdm:
            postfix = {'loss': self.loss_record[-1]}
            if len(self.loss_record) > 1: postfix['loss_delta'] = self.loss_record[-1] - self.loss_record[-2]
            t.set_postfix(postfix)
            
    def get_loss(self):
        self.prime_clusters()
        losses = [cluster.get_loss() for cluster in self.clusters.values()]
        self.deprime_clusters()
        loss = self.Collapser.collapse(pt.stack([
            self.Collapser.collapse(loss_val)
            for loss_val in losses
            if loss_val is not None
        ]))
        return loss
            
    def prime_clusters(self, reprime = False):
        for cluster in self.clusters.values(): cluster.prime_cluster(reprime = reprime)
            
    def deprime_clusters(self):
        for cluster in self.clusters.values(): cluster.deprime_cluster()