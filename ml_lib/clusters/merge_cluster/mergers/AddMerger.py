from ml_lib.clusters.merge_cluster.mergers.RootMerger import RootMerger as Root

class AddMerger(Root):
    defaults = {
        **Root.defaults
    }
    
    def merge(self, input_tensor):
        merged_tensor = input_tensor.sum(dim = 2)
        return merged_tensor