from ml_lib.clusters.root_cluster import RootCluster as Root
from ml_lib.utils.mergers import Sum
from ml_lib.utils.learners import Gradient

class MergeCluster(Root):
    def __init__(self, cluster_name,
                 merger = Sum, merger_params = {},
                 learner = Gradient, learner_params = {}
                ):
        super().__init__(cluster_name)
        self.Merger = merger(**merger_params)
        self.Learner = learner(**learner_params)
    
    def init_cluster(self, reinit = False, **kwargs):
        super().init_cluster(reinit = reinit)
        self.Merger.init_merger(*self.get_input_count(), reinit = reinit)
        
    def deinit_cluster(self):
        self.Merger.deinit_merger()
        
    def prime_cluster(self, reprime = False, **kwargs):
        super().prime_cluster(reprime = reprime)
        self.Merger.prime_merger(reprime = reprime)
        
    def deprime_cluster(self):
        super().deprime_cluster()
        self.Merger.deprime_merger()
    
    def get_input_count(self):
        input_items = [link_item['cluster'].get_output_count(self) for link_item in self.links['input']]
        col_count = max(input_items)
        input_count = len(input_items)
        return (col_count, input_count)
    
    def get_input_tensor(self):
        self.buffers['input_tensor'] = self.buffers.get(
            'input_tensor',
            pt.stack([link_item['cluster'].get_output_tensor(self) for link_item in self.links['input']], dim = 2)
        )
        
        return self.buffers['input_tensor']
    
    def get_output_count(self, req_cluster):
        return self.get_input_count()[0]
        
    def get_output_tensor(self, req_cluster):
        input_tensor = self.get_input_tensor()
        output_tensor = self.Merger.merge(input_tensor)
        return output_tensor
        
    def learn(self, loss, best_iter = False, coef_override = None):
        coefs = self.coefs if coef_override is None else coef_override
        new_coefs = self.update_coefs(loss, coefs)
        self.Merger.update_coefs(new_coefs, best_iter = best_iter)
    
    def update_coefs(self, loss, coefs):
        new_coefs = self.Learner.learn(loss, [coef_value for _, coef_value in coefs])
        new_coefs = zip([coef_label for coef_label, _ in coefs], new_coefs)
        return new_coefs
    
    @property
    def coefs(self):
        coefs = self.Merger.get_coefs()
        return coefs