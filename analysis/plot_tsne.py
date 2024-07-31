import json
import functools
import operator
from typing import Iterable

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def flatten(items):
    """Yield items from any nested iterable"""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def flatten_params(x):
    a, b, c, d, e = x
    b = True if b else b

    return [*a, b, c, d, e]


def only_geometric_f_params(x):
    return x[0]


def pairs_params(dic):
    params = dic["params"][-1]
    lim = 16
    ncrops = 4
    out = []
    y = []
    x1 = [[flatten_params(item) for item in ll[:lim]] for ll in params]
    s1 = dic['selected'][-1]
    s1 = np.array([s1[0][:16], s1[1][:16]])
    for n in range(lim):
        a, b = s1[:, n]
        for i in range(ncrops):
            for j in range(i + 1, ncrops):
                out.append(x1[i][n] + x1[j][n])
                if a == i and j == b:
                    y.append(1)
                else:
                    y.append(0)
    return out, y


def single_point_params(dic, only_geometric=False):
    fn = only_geometric_f_params if only_geometric else flatten_params
    params = dic["params"][-1]
    selected = dic["selected"][-1]
    selected = np.array(selected)
    bs = selected.shape[-1]
    x = [fn(item) for crop in params for item in crop]
    y = []

    for n in range(len(x)):
        a = n % bs
        b = n // bs
        if b in selected[:, a]:
            y.append(1)
        else:
            y.append(0)

    return x, y


def plot_single_point_params(metrics):
    for e, line in enumerate(metrics):
        X, y = single_point_params(line)
        X = np.array(X)

        tsne = PCA(2)
        tsne_result = tsne.fit_transform(X)

        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
        fig, ax = plt.subplots()
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df)
        lim = (tsne_result.min() - 5, tsne_result.max() + 5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        ax.set_title(f"Epoch {e}")
        plt.savefig(f"plots/tsne-single/epoch_{e}.png")
        plt.close()


def plot_single_point_geometric_params(metrics):
    for e, line in enumerate(metrics):
        X, y = single_point_params(line, True)
        X = np.array(X)

        tsne = PCA(2)
        tsne_result = tsne.fit_transform(X)

        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
        fig, ax = plt.subplots()
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df)
        lim = (tsne_result.min() - 5, tsne_result.max() + 5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        ax.set_title(f"Epoch {e}")
        plt.savefig(f"plots/tsne-single-geometric/epoch_{e}.png")
        plt.close()


def plot_pairs_params(metrics):
    for e, line in enumerate(metrics):
        X, y = pairs_params(line)
        X = np.array(X)

        tsne = PCA(2)
        tsne_result = tsne.fit_transform(X)

        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
        fig, ax = plt.subplots()
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df)
        lim = (tsne_result.min() - 5, tsne_result.max() + 5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        ax.set_title(f"Epoch {e}")
        plt.savefig(f"plots/tsne-pairs/epoch_{e}.png")
        plt.close()


if __name__ == "__main__":
    # read data
    with open("../../exp_data/metrics.json", "r") as file:
        # with open("../exp/metrics.json", "r") as file:
        # metrics = json.loads(file.readline())
        metrics = list(map(json.loads, file.readlines()))

    plot_single_point_params(metrics)
    plot_single_point_geometric_params(metrics)
    plot_pairs_params(metrics)
