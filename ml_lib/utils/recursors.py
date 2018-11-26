import torch as pt

class RecursorRoot():
    def __init__(self, return_types = [], return_horizons = [1]):
        self.return_types = return_types
        self.return_horizons = return_horizons
        self.init = False
        self.primed = False
        self.coefs = {}
        self.buffer = {}
        
    def init_recurse(self, col_count, reinit = False):
        if self.init & (not reinit):
            raise Exception('Recursor is initialized and reinit is False')
    
    @property
    def col_mod(self):
        col_mod = len(self.return_types) * len(self.return_horizons)
        return col_mod
        
    def prime_recurse(self, reprime = False):
        if self.primed & (not reprime):
            raise Exception('Recursor is primed and reprime is False')
        
        for coef in self.coefs.values():
            coef.requires_grad = True
            
    def deprime_recurse(self):
        self.buffer = {}
        
    def recurse(self, input_tensor):
        for row in input_tensor:
            self.recurse_row(row)
        recurse_tensor = self.build_recurse_tensor()
        return recurse_tensor
    
    def recurse_row(self, row_tensor):
        ## Placeholder, define in child
        pass
        
    def get_coefs(self):
        coefs = [(coef_label, coef_value) for coef_label, coef_value in self.coefs.items()]
        return coefs
    
    def update_coefs(self, new_coefs):
        for coef_label, coef_value in new_coefs:
            self.coefs[coef_label] = coef_value
            
    def build_recurse_tensor(self):
        ## Placeholder, define in child
        pass
            

class SingleSmooth(RecursorRoot):
    def __init__(self, return_types = ['level'], return_horizons = [1]):
        super().__init__(return_types = return_types, return_horizons = return_horizons)
    
    def init_recurse(self, col_count, reinit = False):
        super().init_recurse(col_count, reinit = reinit)
        self.coefs['level_balance'] = pt.zeros(col_count)
        self.coefs['level_seed'] = pt.zeros(col_count)
        
    def prime_recurse(self, reprime = False):
        super().prime_recurse(reprime = reprime)
        self.buffer['level_balance'] = 1 / (1 + pt.exp(-self.coefs['level_balance']))
        self.buffer['level'] = [self.coefs['level_seed']]
    
    def recurse_row(self, row_tensor):
        smooth_level = self.buffer['level'][-1] + (self.buffer['level_balance'] * (row_tensor - self.buffer['level'][-1]))
        self.buffer['level'].append(smooth_level)
    
    def build_recurse_tensor(self):
        recurse_tensor = pt.cat([
            pt.stack(self.buffer[return_type][1:], dim = 0)
            for return_type in self.return_types
        ], dim = 1)
        return recurse_tensor
    
class DoubleSmooth(RecursorRoot):
    def __init__(self, return_types = ['level'], return_horizons = [1]):
        super().__init__(return_types = return_types, return_horizons = return_horizons)
    
    def init_recurse(self, col_count, reinit = False):
        super().init_recurse(col_count, reinit = reinit)
        self.coefs['level_balance'] = pt.zeros(col_count)
        self.coefs['trend_balance'] = pt.zeros(col_count)
        
        self.coefs['level_seed'] = pt.zeros(col_count)
        self.coefs['trend_seed'] = pt.zeros(col_count)
        
    def prime_recurse(self, reprime = False):
        super().prime_recurse(reprime = reprime)
        self.buffer['level_balance'] = 1 / (1 + pt.exp(-self.coefs['level_balance']))
        self.buffer['trend_balance'] = 1 / (1 + pt.exp(-self.coefs['trend_balance']))
        
        self.buffer['level'] = [self.coefs['level_seed']]
        self.buffer['trend'] = [self.coefs['trend_seed']]
        
    def recurse_row(self, row_tensor):
        smooth_level = self.buffer['level'][-1] + self.buffer['trend'][-1] + (
            self.buffer['level_balance'] * (row_tensor - self.buffer['level'][-1] - self.buffer['trend'][-1])
        )
        smooth_trend = self.buffer['trend'][-1] + (
            self.buffer['trend_balance'] * (smooth_level - self.buffer['level'][-1] - self.buffer['trend'][-1])
        )
        
        self.buffer['level'].append(smooth_level)
        self.buffer['trend'].append(smooth_trend)
    
    def build_recurse_tensor(self):
        recurse_tensor = pt.cat([
            pt.cat([
                pt.stack(self.buffer[return_type][1:], dim = 0) + (horizon * pt.stack(self.buffer[return_type][1:], dim = 0))
                for horizon in self.return_horizons
            ], dim = 1)
            for return_type in self.return_types
        ], dim = 1)
        return recurse_tensor