from datetime import datetime as dt
import torch as pt

class RootCluster:
    defaults = {
        'path_name': 'N/A',
        'verbose': False
    }

    def __init__(self, cluster_name, path_name = None, verbose = None):
        self.path_name = self.defaults['path_name'] if path_name is None else path_name
        self.verbose = self.defaults['verbose'] if verbose is None else verbose
        
        self.name = cluster_name
        self.links = {
            'input': [],
            'output': []
        }
        self.enabled = False
        self.buffer = None
        
    def _v_msg(self, message):
        if self.verbose: print('%s | %s:%s (%s) - %s' % (
            dt.utcnow().isoformat(sep = ' '),
            self.path_name,
            self.name,
            type(self).__name__,
            message
        ))
    
    def _link_names(self, link_type):
        cluster_names = [
            link_item['cluster'].name
            for link_item in self.links[link_type]
        ]
        return cluster_names
    
    def _link_idx(self, cluster_name, link_type):
        link_names = self._link_names(link_type)
        try: link_idx = link_names.index(cluster_name)
        except: link_idx = None
        return link_idx
    
    def add_link(self, cluster, link_type, **kwargs):
        if self._link_idx(cluster.name, link_type) is not None:
            raise Exception('Cluster %s already in %s links.' % (cluster.name, link_type))
            
        link_item = {
            'cluster': cluster,
            'params': {}
        }
        
        self.links[link_type].append(link_item)
        
        self._v_msg('Cluster %s added to %s links.' % (cluster.name, link_type))
        
    def del_link(self, cluster_name, link_type, confirm = False):
        if not confirm: raise Exception('Kwarg confirm is False.')
        
        cluster_idx = self._link_idx(cluster_name, link_type)
        if cluster_idx is None:
            raise Exception('Cluster %s not in %s links.' % (cluster_name, link_type))
        
        del self.links[link_type][cluster_idx]
        
        self._v_msg('Cluster %s deleted from %s links.' % (cluster_name, link_type))
        
    def get_output_count(self, req_cluster_name):
        ## Define in child classes
        return 0
    
    def get_output_tensor(self, req_cluster_name):
        ## Define in child classes
        pass
    
    @property
    def input_count(self):
        ## Define in child classes
        input_count = sum([
            link['cluster'].get_output_count(self.name)
            for link in self.links['input']
        ])
        return input_count
    
    @property
    def input_tensor(self):
        input_tensor = pt.cat([
            link['cluster'].get_output_tensor(self.name)
            for link in self.links['input']
        ], dim = 1)
        return input_tensor        
    
    def enable(self, override = False):
        if self.enabled & (override is False):
            raise Exception('Cluster is already enabled and override is False.')
        
        self.enabled = True
        self._v_msg('Cluster enabled.')
        
    def build_batch_splits(self):
        ## Define in child classes
        pass
    
    @property
    def loss(self):
        ## Define in child classes
        return None
    
    def clear_buffer(self):
        self.buffer = None
        
    def learn(self, loss):
        ## Define in child classes
        pass