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


def get_overlap_data(metrics):
    results = np.array([_avg_value_selection_one_epoch(epoch, "overlap") for epoch in metrics]).T
    return results[0], results[1]


if __name__ == "__main__":
    paths = ("C:/Users/ovi/Documents/SelfSupervisedLearning/metrics/pickle/simsiam-minsim-resnet50-ep100-seed0.pkl",
             "C:/Users/ovi/Documents/SelfSupervisedLearning/metrics/pickle/simsiam-minsim-resnet50-ep100-seed1.pkl",
             "C:/Users/ovi/Documents/SelfSupervisedLearning/metrics/pickle/simsiam-minsim-resnet50-ep100-seed2.pkl")

    iou_on, iou_off = [], []
    for path in paths:
        metrics = pd.read_pickle(path)
        on, off = get_iou_data(metrics["params"], metrics["selected"])
        iou_on.append(on)
        iou_off.append(off)
    iou_on = np.array(iou_on)
    iou_off = np.array(iou_off)
    print(iou_on.shape)
    data = [iou_on.mean(axis=0), iou_off.mean(axis=0)]
    std = [iou_on.std(axis=0), iou_off.std(axis=0)]

    file_name = "iou.png"
    title = "Average IoU of selected/not-selected pairs"
    labels = ["selected", "not-selected"]
    exp_name = ""
    xlabel = "epoch"
    ylabel = "IoU"

    fig, ax = plt.subplots()
    for y, sig, label, color in zip(data, std, labels, BASE_COLORS):
        x = np.arange(len(y))
        ax.plot(x, y, label=label, color=color)
        if sig is not None:
            ax.fill_between(x, y + sig, y - sig, facecolor=color, alpha=0.25)
    ax.grid()
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax.set_title(file_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.text(0.5, 0.01, exp_name, fontsize='x-small', ha='center')
    plt.tight_layout()

    # out_dir = Path(...)
    # out_dir.mkdir(parents=True, exist_ok=True)
    # save_path = out_dir.joinpath(file_name)
    plt.show()
    # fig.savefig(save_path)
    # print(f"Plot saved at {save_path}")
    # plt.close(fig)
