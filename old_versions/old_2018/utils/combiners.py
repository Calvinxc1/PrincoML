class Simple():
    def combine(self, input_tensor, coefs):
        combined_tensor = (input_tensor @ coefs['weights']) + coefs['bias']
        return combined_tensor
