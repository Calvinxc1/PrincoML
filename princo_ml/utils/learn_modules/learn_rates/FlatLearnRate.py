from .RootLearnRate import RootLearnRate as Root

class FlatLearnRate(Root):
    @property
    def learn_rate(self):
        return self.seed_learn