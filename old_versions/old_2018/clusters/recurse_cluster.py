import torch as pt

from ml_lib.clusters.root_cluster import RootCluster as Root
from ml_lib.utils.learners import Gradient
from ml_lib.utils import recursors


class RecurseCluster(Root):
    def __init__(self, cluster_name,
                 recursor=recursors.SingleSmooth, recursor_params={},
                 learner=Gradient, learner_params={}
                 ):
        super().__init__(cluster_name)
        self.Recursor = recursor(**recursor_params)
        self.Learner = learner(**learner_params)

    def init_cluster(self, reinit=False, **kwargs):
        super().init_cluster(reinit=reinit)
        self.Recursor.init_recurse(self.get_input_count(), reinit=reinit)

    def deinit_cluster(self):
        self.Recursor.deinit_recurse()
        super().deinit_cluster()

    def prime_cluster(self, reprime=False, **kwargs):
        super().prime_cluster(reprime=reprime)
        self.Recursor.prime_recurse(reprime=reprime)

    def deprime_cluster(self):
        super().deprime_cluster()
        self.Recursor.deprime_recurse()

    def get_output_count(self, req_cluster):
        output_count = self.get_input_count() * self.Recursor.col_mod
        return output_count

    def get_output_tensor(self, req_cluster):
        input_tensor = self.get_input_tensor()
        output_tensor = self.Recursor.recurse(input_tensor)
        return output_tensor

    def learn(self, loss, best_iter=False, coef_override=None):
        coefs = self.coefs if coef_override is None else coef_override
        new_coefs = self.update_coefs(loss, coefs)
        self.Recursor.update_coefs(new_coefs, best_iter=best_iter)

    def update_coefs(self, loss, coefs):
        new_coefs = self.Learner.learn(
            loss, [coef_value for _, coef_value in coefs])
        new_coefs = zip([coef_label for coef_label, _ in coefs], new_coefs)
        return new_coefs

    @property
    def coefs(self):
        coefs = self.Recursor.get_coefs()
        return coefs
