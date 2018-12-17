import torch as pt
from datetime import datetime as dt

class RootCluster:
    def __init__(self, cluster_name, path_name = 'N/A', verbose = False):
        self.name = cluster_name
        self.path_name = path_name
        self.verbose = verbose
        self.connects = {
            'input': [],
            'output': []
        }
        
    def _v_mess(self, message):
        if self.verbose: print('%s | %s:%s (%s) - %s' % (
            dt.utcnow().isoformat(sep = ' '),
            self.path_name,
            self.name,
            type(self).__name__, message
        ))
    
    def _cluster_names(self, conn_type):
        cluster_names = [
            conn_item['cluster'].name
            for conn_item in self.connects[conn_type]
        ]
        return cluster_names
    
    def _cluster_idx(self, cluster_name, conn_type):
        cluster_names = self._cluster_names(conn_type)
        cluster_idx = cluster_names.index(cluster_name)
        return cluster_idx
    
    def add_conn(self, cluster, conn_type):
        if cluster.name in self._cluster_names(conn_type):
            raise Exception('Cluster %s already in connection %s.' % (cluster.name, conn_type))
            
        conn_item = {
            'cluster': cluster,
            'params': {}
        }
        
        self.connects[conn_type].append(conn_item)
        
        self._v_mess('Cluster %s added to %s connections.' % (cluster.name, conn_type))
        
    def del_conn(self, cluster_name, conn_type, confirm = False):
        if not confirm: raise Exception('Kwarg confirm is False.')
        
        if cluster_name not in self._cluster_names(conn_type):
            raise Exception('Cluster %s not in connection %s.' % (cluster_name, conn_type))
            
        cluster_idx = self._cluster_idx(cluster_name, conn_type)
        del self.connects[conn_type][cluster_idx]
        
        self._v_mess('Cluster %s deleted from %s connections.' % (cluster_name, conn_type))
        
class DataCluster(RootCluster):
    def __init__(self, cluster_name, data_frame, path_name = 'N/A', verbose = False):
        super().__init__(cluster_name, path_name = path_name, verbose = verbose)
        self.data = None
        self.add_data(data_frame)
        
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
            
        self.data = convert_frame(data_frame)
        
        self._v_mess('Data frame added, overwrite %s' % overwrite)
        
    def add_conn(self, cluster, conn_type, data_cols = None, **kwargs):
        col_idx = np.array([self.data['columns'].get_loc(col) for col in data_cols])
        super().add_link(cluster, conn_type)
        
        self.connects[link_type][-1]['params']['columns'] = col_idx
        
    def get_conn_cols(self, cluster_name, link_type):
        link_idx = self.get_link_idx(cluster_name, link_type)
        if link_idx is None:
            raise Exception('%s: Cluster %s not present in %s link' % (self.name, cluster_name, link_type))
            
        link_cols = self.links[link_type][link_idx]['params']['columns']
        return link_cols