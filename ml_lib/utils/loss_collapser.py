class Linear():
    def __init__(self, mean = True):
        self.mean = mean
        
    def collapse(self, loss_data):
        collapsed_loss = loss_data.mean() if self.mean else loss_data.sum()
        return collapsed_loss