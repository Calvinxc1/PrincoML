from .RootMerger import RootMerger as Root

class AddMerger(Root):
    defaults = {
        **Root.defaults
    }
    
    def merge_process(self, input_tensor):
        merged_tensor = input_tensor.sum(dim = 2)
        return merged_tensor