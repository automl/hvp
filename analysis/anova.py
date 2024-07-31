import numpy as np

from fanova import fANOVA
import fanova.visualizer

import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import os

path = os.path.dirname(os.path.realpath(__file__))

# directory in which you can find all plots
plot_dir = 'example-plots'

n_samples = 100
# artificial dataset (here: features)
features = np.loadtxt(path + '/features.csv', delimiter=",").T[:n_samples, :]
responses = np.loadtxt(path + '/responses.csv', delimiter=",")[:n_samples]

# config space
pcs = list(zip(np.min(features, axis=0), np.max(features, axis=0)))
cs = ConfigSpace.ConfigurationSpace()
for i in range(len(pcs)):
    cs.add_hyperparameter(UniformFloatHyperparameter("%i" % i, pcs[i][0], pcs[i][1]))

# create an instance of fanova with trained forest and ConfigSpace
f = fANOVA(X=features, Y=responses, n_trees=32)

# marginal of particular parameter:
# dims = tuple(range(features.shape[1]))
# res = f.quantify_importance(dims)
# for k, v in res.items():
#     print(k, v)
print(f.quantify_importance((0,)))
print(f.quantify_importance((1,)))
print(f.quantify_importance((2,)))
print(f.quantify_importance((3,)))
print(f.quantify_importance((4,)))
print(f.quantify_importance((5,)))
print(f.quantify_importance((6,)))
print(f.quantify_importance((7,)))


# getting the 3 most important pairwise marginals sorted by importance
best_margs = f.get_most_important_pairwise_marginals(n=3)
print(best_margs)

# visualizations:
# first create an instance of the visualizer with fanova object and configspace
vis = fanova.visualizer.Visualizer(f, cs, 'example_output')
# creating the plot of pairwise marginal:
# vis.plot_marginal(0)
# vis.plot_marginal(1)
# vis.plot_marginal(2)
vis.plot_pairwise_marginal([0, 1])
# creating all plots in the directory
vis.create_all_plots(plot_dir)
