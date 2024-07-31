import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BASE_COLORS
from pathlib import Path

from utils import read_file, calc_iou, calc_overlap, calc_area, save_plot


def _avg_value_selection_one_epoch(epoch_metrics, typ):
    if typ == "overlap":
        fn = calc_overlap
    else:
        fn = calc_iou
    params_epoch = epoch_metrics["params"]
    selected_epoch = epoch_metrics["selected"]
    select = 0.
    not_select = 0
    s_count = 0.
    not_count = 0
    for i, iteration in enumerate(params_epoch):
        selection = np.array(selected_epoch[i])
        for c in range(len(iteration[0])):
            params = [item[c][0] for item in iteration]
            ss = selection[:, c]
            for n, x1 in enumerate(params):
                for m, x2 in enumerate(params[n + 1:], n + 1):
                    value = fn(x1, x2)
                    if np.equal(np.array([n, m]), ss).all():
                        select += value
                        s_count += 1
                    else:
                        not_select += value
                        not_count += 1

    return select / s_count, not_select / not_count


def iou_s_per_epoch(params, selected):
    n_iter, n_crops, n_params = params.shape

    select = 0.
    not_select = 0.
    s_count = 0
    not_count = 0
    for i in range(n_iter):
        for m in range(n_crops):
            p1 = params[i, m][:6]
            for n in range(m+1, n_crops):
                p2 = params[i, n][:6]
                value = calc_iou(p1, p2)
                if np.equal(np.array([m, n]), selected[i]).all():
                    select += value
                    s_count += 1
                else:
                    not_select += value
                    not_count += 1
    return select / s_count, not_select / not_count


def get_iou_data(params, selected):
    epochs = len(params)
    results = np.array([iou_s_per_epoch(params[n], selected[n]) for n in range(epochs)]).T
    return results[0], results[1]


if __name__ == "__main__":
    path = "C:/Users/ovi/Documents/SelfSupervisedLearning/metrics/pickle/simsiam-minsim-resnet50-ep100-seed0.pkl"
    metrics = pd.read_pickle(path)
    data0 = get_iou_data(metrics["params"], metrics["selected"])

    path = "C:/Users/ovi/Documents/SelfSupervisedLearning/metrics/pickle/simsiam-minsim-resnet50-ep100-seed1.pkl"
    metrics = pd.read_pickle(path)
    data1 = get_iou_data(metrics["params"], metrics["selected"])

    path = "C:/Users/ovi/Documents/SelfSupervisedLearning/metrics/pickle/simsiam-minsim-resnet50-ep100-seed2.pkl"
    metrics = pd.read_pickle(path)
    data2 = get_iou_data(metrics["params"], metrics["selected"])

    select = np.stack((data0[0], data1[0], data2[0]))
    not_select = np.stack((data0[1], data1[1], data2[1]))

    data = [select.mean(axis=0), not_select.mean(axis=0)]
    std = [select.std(axis=0), not_select.std(axis=0)]

    # file_name = "iou.png"
    title = "Average IoU of selected vs not-selected-pairs"
    labels = ["selected", "not-selected"]
    exp_name = ""
    xlabel = "Epoch"
    ylabel = "IoU"

    fig, ax = plt.subplots()
    for y, sig, label, color in zip(data, std, labels, BASE_COLORS):
        x = np.arange(len(y))
        ax.plot(x, y, label=label, color=color)
        if sig is not None:
            ax.fill_between(x, y + sig, y - sig, facecolor=color, alpha=0.25)
    ax.grid()
    ax.legend(loc='center right')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    # ax.set_title(file_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.text(0.5, 0.01, exp_name, fontsize='x-small', ha='center')
    plt.tight_layout()

    # out_dir = Path(...)
    # out_dir.mkdir(parents=True, exist_ok=True)
    # save_path = out_dir.joinpath(file_name)
    plt.show()
    fig.savefig("pic.png")
    plt.close(fig)
