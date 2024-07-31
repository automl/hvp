import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Iterable

from utils import read_file

NUM_CROPS = 4
PLOT_DIR_PATH = "plots/tsne/"
LIM_BS = 16


def flatten(items):
    """Yield items from any nested iterable"""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def flatten_params(x):
    a, b, c, d, e = x
    b = True if b else b
    return [*a, b, c, d, e]


def only_geometric_f_params(x):
    return x[0]


def pair_params(dic, *args):
    params = dic["params"][-1]
    out = []
    y = []
    x1 = [[flatten_params(item) for item in ll[:LIM_BS]] for ll in params]
    s1 = np.array(dic['selected'][-1][:LIM_BS])
    for n in range(LIM_BS):
        a, b = s1[:, n]
        for i in range(NUM_CROPS):
            for j in range(i + 1, NUM_CROPS):
                out.append(x1[i][n] + x1[j][n])
                y.append(1 if (a == i and j == b) else 0)
    return out, y


def single_params(dic, only_geometric=False):
    fn = only_geometric_f_params if only_geometric else flatten_params
    params = dic["params"][-1]
    selected = np.array(dic["selected"][-1])
    bs = selected.shape[-1]
    x = [fn(item) for crop in params for item in crop]
    y = [1 if (n % bs) in selected[:, n // bs] else 0 for n in range(len(x))]
    return x, y


def plots_tsne(metrics, exp_name, geometric=False, pair=False, every_x_epochs=1):
    stg = f"{'pair' if pair else 'single'}" + f"{'-geometric' if geometric else ''}"
    out_path = PLOT_DIR_PATH + f"/{stg}"
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    extract_fn = pair_params if pair else single_params
    for e, line in enumerate(metrics):
        if e % every_x_epochs:
            continue
        X, y = extract_fn(line, geometric)
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
        fig.text(0.5, 0.01, exp_name, fontsize='x-small', ha='center')
        fig.tight_layout()
        plt.savefig(f"{out_path}/epoch_{e}.png")
        print(f"Saved plot at {out_path}/epoch_{e}.png")
        plt.close()


if __name__ == "__main__":
    # get data
    data_path = "../../exp_data/metrics0.json"
    metrics = read_file(data_path)

    exp_name = "simsiam-minsim-collect_metrics-resnet50-ImageNet-ep100-bs256-select_cross-ncrops4-lr0.05-wd0.0001-mom0.9-seed0"
    plots_tsne(metrics, exp_name, geometric=False, pair=False, every_x_epochs=10)
    plots_tsne(metrics, exp_name, geometric=True, pair=False, every_x_epochs=10)
    plots_tsne(metrics, exp_name, geometric=False, pair=True, every_x_epochs=10)
