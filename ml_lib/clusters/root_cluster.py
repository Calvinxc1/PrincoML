import torch as pt

class RootCluster():
    def __init__(self, cluster_name):
        self.name = cluster_name
        self.init = False
        self.primed = False
        self.links = {
            'output': [],
            'input': []
        }
        self.buffers = {}
    
    def get_link_idx(self, cluster_name, link_type):
        link_names = [link_clusters['cluster'].name for link_clusters in self.links[link_type]]
        
        try:
            link_index = link_names.index(cluster_name)
        except:
            link_index = None
        
        return link_index
        
    def add_link(self, cluster, link_type, **kwargs):        
        if link_type not in self.links.keys():
            raise Exception('%s: Link type %s not a valid link type' % (self.name, link_type))
        
        if self.get_link_idx(cluster.name, link_type) is not None:
            raise Exception('%s: Cluster %s already present in %s link' % (self.name, cluster.name, link_type))
            
        link_item = {
            'cluster': cluster,
            'params': {}
        }
        self.links[link_type].append(link_item)
        
    def remove_link(self, cluster_name, link_type):
        if link_type not in self.links.keys():
            raise Exception('%s: Link type %s not a valid link type' % (self.name, link_type))
        
        link_idx = self.get_link_idx(cluster_name, link_type)
        if link_idx is None:
            raise Exception('%s: Cluster %s not present in %s link' % (self.name, cluster_name, link_type))
            
        del self.links[link_type][link_idx]
        
    def init_cluster(self, reinit = False, **kwargs):
        ## Placeholder method, define in child classes
        if self.init & (not reinit):
            raise Exception('%s: Cluster already initialized and reinit is False' % self.name)
        
        self.init = True
        
    def deinit_cluster(self):
        ## Placeholder, define in child
        pass
        
    def prime_cluster(self, reprime = False, **kwargs):
        ## Placeholder method, define in child classes
        if not self.init:
            raise Exception('%s: Cluster not initialized' % self.name)
        
        if self.primed & (not reprime):
            raise Exception('%s: Cluster already primed and reprime is False' % self.name)
        
        self.primed = True
        
    def deprime_cluster(self):
        if not self.primed:
            raise Exception('%s: Cluster not initialized' % self.name)
        
        self._deprime_buffers()
        self.primed = False
        
    def _deprime_buffers(self):
        self.buffers = {}
    
    def get_output_count(self, req_cluster):
        ## Placeholder method, define in child classes
        return 0
        
    def get_output_tensor(self, req_cluster):
        ## Placeholder method, define in child classes
        pass
    
    def get_input_count(self):
        input_count = sum([link_item['cluster'].get_output_count(self) for link_item in self.links['input']])
        return input_count
    
    def get_input_tensor(self):
        self.buffers['input_tensor'] = self.buffers.get(
            'input_tensor',
            pt.cat([link_item['cluster'].get_output_tensor(self) for link_item in self.links['input']], dim = 1)
        )
        
        return self.buffers['input_tensor']
    
    def get_losses(self):
        ## Placeholder method, define in child classes
        return None
    
    def learn(self, loss, best_iter = False, **kwargs):
        ## Placeholder method, define in child classes
        pass