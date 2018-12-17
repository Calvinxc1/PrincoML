class MeanVar():
    def __init__(self, balancer = 1e-8):
        self.balancer = balancer
        
    def norm(self, input_tensor, coefs):
        whited_tensor = (
            input_tensor - input_tensor.mean(dim = 0, keepdim = True)
        ) / (
            input_tensor.std(dim = 0, keepdim = True) + self.balancer
        )
        normed_tensor = (whited_tensor * coefs['weight']) + coefs['bias']
        return normed_tensor