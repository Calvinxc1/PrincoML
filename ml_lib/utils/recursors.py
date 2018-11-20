import torch as pt

class RecursorRoot():
    def __init__(self):
        self.init = False
        self.primed = False
        self.coefs = {}
        self.buffer = {}
        
    def init_recurse(self, col_count, reinit = False):
        if self.init & (not reinit):
            raise Exception('Recursor is initialized and reinit is False')
        
    def prime_recurse(self, reprime = False):
        if self.primed & (not reprime):
            raise Exception('Recursor is primed and reprime is False')
        
        for coef in self.coefs.values():
            coef.requires_grad = True
            
    def deprime_recurse(self):
        self.buffer = {}
        
    def get_coefs(self):
        coefs = [(coef_label, coef_value) for coef_label, coef_value in self.coefs.items()]
        return coefs
    
    def update_coefs(self, new_coefs):
        for coef_label, coef_value in new_coefs:
            self.coefs[coef_label] = coef_value

class SingleSmooth(RecursorRoot):        
    def init_recurse(self, col_count, reinit = False):
        super().init_recurse(col_count, reinit = reinit)
        self.coefs['level_balance'] = pt.zeros(col_count)
        self.coefs['level_seed'] = pt.zeros(col_count)
        
    def prime_recurse(self, reprime = False):
        super().prime_recurse(reprime = reprime)
        self.buffer['level_balance'] = 1 / (1 + pt.exp(-self.coefs['level_balance']))
        self.buffer['level'] = [self.coefs['level_seed']]
        
    def recurse(self, input_tensor):
        smooth_trend = self.buffer['level'][-1] + (self.buffer['level_balance'] * (input_tensor - self.buffer['level'][-1]))
        self.buffer['level'].append(smooth_trend)
        return smooth_trend
    