import pandas as pd
import torch as pt

pt.set_default_tensor_type('torch.FloatTensor')

from princo_ml.controllers.Controller import Controller as Control
from princo_ml.clusters.DataCluster import DataCluster as Data
from princo_ml.clusters.LearnCluster import LearnCluster as Learn
from princo_ml.clusters.MergeCluster import MergeCluster as Merge
from princo_ml.utils.learn_modules.learners.GradientLearner import GradientLearner
from princo_ml.utils.learn_modules.learners.NewtonLearner import NewtonLearner
from princo_ml.utils.learn_modules.learners.MomentumLearner import MomentumLearner
from princo_ml.utils.learn_modules.learners.SmoothLearner import SmoothLearner
from princo_ml.utils.learn_modules.activators.LinearActivate import LinearActivate
from princo_ml.utils.learn_modules.activators.SigmoidActivate import SigmoidActivate
from princo_ml.utils.learn_modules.activators.TanhActivator import TanhActivator
from princo_ml.utils.learn_modules.activators.ReluActivator import ReluActivator
from princo_ml.utils.learn_modules.DenseHingeModule import DenseHingeModule

dataset = pd.read_csv(
    'data_files/kc_house_data.csv',
    index_col = 'id', parse_dates = ['date'], date_parser = lambda x: pd.datetime.strptime(x, '%Y%m%dT%H%M%S')
)
dataset = dataset[[col for col in dataset.columns if not col.endswith('15')]]
feature_cols = ['sqft_living', 'bedrooms', 'bathrooms']
target_cols = ['price']

verbose = False

learner = SmoothLearner

learn_rate_kwargs = {
    'seed_learn': 5e-1
}

loss_kwargs = {}

nesterov = True

hinger = DenseHingeModule
hinges = 5

activator = TanhActivator
activator_kwargs = {
    #'leak': 0.1
}

batcher_kwargs = {
    'proportion': 1.0
}

control = Control(
    'regression_controller',
    loss_smooth_coefs = [0.9, 100],
    use_tqdm=verbose,
)

control.add_cluster(
    Data(
        'data_cluster',
        dataset[feature_cols + target_cols],
        splitter_kwargs = {'verbose': verbose},
        batcher_kwargs = {'verbose': verbose, **batcher_kwargs},
        loss_kwargs = {'verbose': verbose, **loss_kwargs},
        loss_combiner_kwargs = {'verbose': verbose},
        verbose = verbose
    )
)

control.link_add(
    Learn(
        'learn_cluster_1',
        module = hinger,
        module_kwargs = {
            'nodes': 128,
            'hinges': hinges,
            'verbose': verbose, 'nesterov': nesterov,
            'bias_init_kwargs': {'verbose': verbose},
            'weight_init_kwargs': {'verbose': verbose},
            'combiner_kwargs': {'verbose': verbose},
            'activator': activator, 'activator_kwargs': {'verbose': verbose, **activator_kwargs},
            'learner': learner, 'learner_kwargs': {'verbose': verbose},
            'learn_rate_kwargs': {'verbose': verbose, **learn_rate_kwargs}
        },
        verbose = verbose
    ),
    'data_cluster', 'input', data_cols = feature_cols
)

control.link_add(
    Learn(
        'learn_cluster_2',
        module_kwargs = {
            'nodes': 32,
            'verbose': verbose, 'nesterov': nesterov,
            'bias_init_kwargs': {'verbose': verbose},
            'weight_init_kwargs': {'verbose': verbose},
            'combiner_kwargs': {'verbose': verbose},
            'activator': activator, 'activator_kwargs': {'verbose': verbose, **activator_kwargs},
            'learner': learner, 'learner_kwargs': {'verbose': verbose},
            'learn_rate_kwargs': {'verbose': verbose, **learn_rate_kwargs}
        },
        verbose = verbose
    ),
    'learn_cluster_1', 'input'
)

control.link_add(
    Learn(
        'learn_cluster_out',
        module_kwargs = {
            'nodes': len(target_cols),
            'verbose': verbose,
            'nesterov': nesterov,
            'learner': learner,
            'learn_rate_kwargs': {'verbose': verbose, **learn_rate_kwargs}
        },
        verbose = verbose
    ),
    'learn_cluster_2', 'input'
    #'data_cluster', 'input', data_cols = feature_cols
)

control.link_clusters('learn_cluster_out', 'data_cluster', data_cols = target_cols)

control.enable_network()

control.train_model(1000)

