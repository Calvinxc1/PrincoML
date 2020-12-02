import pandas as pd
import numpy as np
import torch as pt
import tqdm
import matplotlib.pyplot as plt

from ..utils.loss_combiners import MeanLossCombiner
from ..utils.regularizers import NormRegularizer

class Controller:
    defaults = {
        'train_split': 'train', 'coef_lock_split': 'holdout',
        'use_tqdm': True,
        'loss_combiner': MeanLossCombiner, 'loss_combiner_kwargs': {},
        'regularizer': NormRegularizer, 'regularizer_kwargs': {'l1': 0, 'l2': 0},
        'loss_smooth_coefs': np.array([0.9, 10])
    }

    def __init__(self, control_name, coef_lock_split=None,
                 train_split=None, use_tqdm=None, loss_smooth_coefs=None,
                 loss_combiner=None, loss_combiner_kwargs={},
                 regularizer=None, regularizer_kwargs={}
                 ):
        loss_combiner = self.defaults['loss_combiner'] if loss_combiner is None else loss_combiner
        loss_combiner_kwargs = {
            **self.defaults['loss_combiner_kwargs'], **loss_combiner_kwargs}
        regularizer = self.defaults['regularizer'] if regularizer is None else regularizer
        regularizer_kwargs = {
            **self.defaults['regularizer_kwargs'], **regularizer_kwargs}
        loss_smooth_coefs = np.array(
            self.defaults['loss_smooth_coefs'] if loss_smooth_coefs is None else loss_smooth_coefs)

        self.name = control_name
        self.clusters = {}
        self.train_split = self.defaults['train_split'] if train_split is None else train_split
        self.use_tqdm = self.defaults['use_tqdm'] if use_tqdm is None else use_tqdm
        self.coef_lock_split = self.defaults['coef_lock_split'] if coef_lock_split is None else coef_lock_split
        self.enabled = False
        self.loss_smooth = 1 - \
            ((1 - loss_smooth_coefs[0]) ** (1 / loss_smooth_coefs[1]))

        self.LossCombiner = loss_combiner(self.name, **loss_combiner_kwargs)
        self.Regularizer = regularizer(self.name, **regularizer_kwargs)

    def add_cluster(self, cluster):
        if cluster.name in self.clusters.keys():
            raise Exception('Cluster %s already in controller.' % cluster.name)

        cluster.path_name = self.name
        self.clusters[cluster.name] = cluster

    def link_clusters(self, from_cluster_name, to_cluster_name, **kwargs):
        self.clusters[from_cluster_name].add_link(
            self.clusters[to_cluster_name], 'output', **kwargs)
        self.clusters[to_cluster_name].add_link(
            self.clusters[from_cluster_name], 'input', **kwargs)

    def link_add(self, cluster, link_cluster_name, link_type, **kwargs):
        cluster_name = cluster.name
        self.add_cluster(cluster)

        if link_type == 'input':
            self.link_clusters(link_cluster_name, cluster_name, **kwargs)
        elif link_type == 'output':
            self.link_clusters(cluster_name, link_cluster_name, **kwargs)
        else:
            raise Exception(
                '%s is not a valid link_type, must be either input or output.' % link_type)

    def enable_network(self):
        self.best_epoc = 0
        for cluster in self.clusters.values():
            cluster.enable()
        self.loss_record = {'raw': {}, 'smooth': {}}
        self.learn_rate_record = {}
        self.lowest_loss = np.inf
        self.build_batch_splits()

    def build_batch_splits(self):
        for cluster in self.clusters.values():
            cluster.build_batch_splits()

    @property
    def network_loss(self):
        network_loss = {}
        for cluster in self.clusters.values():
            loss_val = cluster.loss
            if loss_val is None:
                continue
            for key, value in loss_val.items():
                if key not in network_loss.keys():
                    network_loss[key] = []
                network_loss[key].append(value)

        for key in network_loss.keys():
            network_loss[key] = self.LossCombiner.loss_combine(
                pt.stack(network_loss[key]))

        return network_loss

    def clear_buffers(self):
        for cluster in self.clusters.values():
            cluster.clear_buffer()

    def network_learn(self, best_iter):
        network_loss = self.network_loss[self.train_split]

        reg_coefs = []
        for coefs in self.network_coefs(True).values():
            for coef in coefs.values():
                reg_coefs.append(coef)
        reg_loss = self.Regularizer.regularize(reg_coefs)

        learn_loss = network_loss + reg_loss

        for cluster in self.clusters.values():
            cluster.learn(learn_loss, best_iter=best_iter)

    def train_model(self, epocs, lock_coefs=True):
        t = tqdm.tnrange(epocs) if self.use_tqdm else range(epocs)

        for epoc in t:
            best_iter = False
            self.build_batch_splits()

            learn_rates = self.network_learn_rates
            for key, value in learn_rates.items():
                if key not in self.learn_rate_record:
                    self.learn_rate_record[key] = []
                self.learn_rate_record[key].append(value)

            postfix = {'best_epoc': self.best_epoc,
                       'best_loss': self.lowest_loss}
            network_loss = self.network_loss
            for key, value in network_loss.items():
                if key not in self.loss_record['raw']:
                    self.loss_record['raw'][key] = []
                if key not in self.loss_record['smooth']:
                    self.loss_record['smooth'][key] = []

                loss_val = np.asscalar(value.detach().cpu().numpy())
                prior_loss = loss_val if len(
                    self.loss_record['smooth'][key]) == 0 else self.loss_record['smooth'][key][-1]
                smooth_loss = (self.loss_smooth * loss_val) + \
                    ((1 - self.loss_smooth) * prior_loss)

                self.loss_record['raw'][key].append(loss_val)
                self.loss_record['smooth'][key].append(smooth_loss)

                postfix[key] = self.loss_record['smooth'][key][-1]

            if self.use_tqdm:
                t.set_postfix(postfix)

            if (self.loss_record['smooth'][self.coef_lock_split][-1] < self.lowest_loss) & (self.loss_record['smooth'][self.coef_lock_split][-1] != -np.inf):
                best_iter = True
                self.lowest_loss = self.loss_record['smooth'][self.coef_lock_split][-1]
                self.best_epoc = epoc

            self.network_learn(best_iter)
            self.clear_buffers()

            if np.any([(pd.isnull(loss.detach().cpu().numpy()) | np.isinf(loss.detach().cpu().numpy())) for loss in network_loss.values()]):
                print('Null value appeared! Terminating learning!')
                break

        if lock_coefs:
            self.lock_coefs()

    def lock_coefs(self):
        for cluster in self.clusters.values():
            cluster.lock_coefs()

    def plot_losses(self, start_idx=0, end_idx=None, loss_type='smooth', figsize=(16, 10)):
        plt.figure(figsize=figsize)
        for key, value in self.loss_record[loss_type].items():
            plt.plot(value[start_idx:len(
                value) if end_idx is None else start_idx:end_idx], label=key)
        plt.axvline(self.best_epoc - start_idx, c='r', label='best_epoc')
        plt.legend()

    def network_coefs(self, exempt_bias=False):
        coefs = {}
        for cluster_name, cluster in self.clusters.items():
            cluster_coefs = cluster.coefs(exempt_bias=exempt_bias)
            if cluster_coefs is None:
                continue
            coefs[cluster_name] = cluster_coefs
        return coefs

    def predict(self, data_dict):
        for cluster, data_frame in data_dict.items():
            self.clusters[cluster].load_manual_data(data_frame)

        self.clear_buffers()

        predicts = {}
        for cluster_name, cluster in self.clusters.items():
            if cluster_name in data_dict.keys():
                continue

            predict = cluster.predict()
            if predict is None:
                continue

            predicts[cluster_name] = predict

        losses = self.network_loss['all'].detach().cpu().numpy()

        self.clear_buffers()

        for cluster in data_dict.keys():
            self.clusters[cluster].unload_manual_data()

        return {'outputs': predicts, 'loss': losses}

    @property
    def network_learn_rates(self):
        learn_rates = {}
        for cluster_name, cluster in self.clusters.items():
            learn_rate = cluster.learn_rate
            if learn_rate is None:
                continue
            learn_rates[cluster_name] = learn_rate

        return learn_rates

    def plot_learn_rates(self, figsize=(16, 10)):
        plt.figure(figsize=figsize)
        for cluster, learns in self.learn_rate_record.items():
            plt.plot(learns, label=cluster)
        plt.legend()
