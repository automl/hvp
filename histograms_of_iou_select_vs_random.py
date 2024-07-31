import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import read_file, calc_area, calc_overlap, calc_iou, calc_overlap_norm, save_histogram


def iou_un_select_one_epoch(epoch_metrics):
    params_epoch = epoch_metrics["params"]
    selected_epoch = epoch_metrics["selected"]
    select = []
    unselect = []
    for i, iteration in enumerate(params_epoch):
        selected = np.array(selected_epoch[i])
        for c in range(len(iteration[0])):
            params = [item[c][0] for item in iteration]
            ss = selected[:, c]
            for n, x1 in enumerate(params):
                for m, x2 in enumerate(params[n + 1:], n + 1):
                    overlap = calc_iou(x1, x2)
                    if np.equal(np.array([n, m]), ss).all():
                        select.append(overlap)
                    else:
                        unselect.append(overlap)
    return select, unselect


def iou_un_select_one_epoch_vs_random(epoch_metrics):
    params_epoch = epoch_metrics["params"]
    selected_epoch = epoch_metrics["selected"]
    select = []
    unselect = []
    for i, iteration in enumerate(params_epoch):
        selected = np.array(selected_epoch[i])
        for c in range(len(iteration[0])):
            params = [item[c][0] for item in iteration]
            ss = selected[:, c]
            for n, x1 in enumerate(params):
                for m, x2 in enumerate(params[n + 1:], n + 1):
                    overlap = calc_iou(x1, x2)
                    if np.equal(np.array([n, m]), ss).all():
                        select.append(overlap)
                    if n == 0 and m == 1:
                        unselect.append(overlap)
    return select, unselect


def normed_overlap_un_select_one_epoch(epoch_metrics):
    params_epoch = epoch_metrics["params"]
    selected_epoch = epoch_metrics["selected"]
    select = []
    unselect = []
    for i, iteration in enumerate(params_epoch):
        selected = np.array(selected_epoch[i])
        for c in range(len(iteration[0])):
            params = [item[c][0] for item in iteration]
            ss = selected[:, c]
            for n, x1 in enumerate(params):
                for m, x2 in enumerate(params[n + 1:], n + 1):
                    overlap = calc_overlap_norm(x1, x2)
                    if np.equal(np.array([n, m]), ss).all():
                        select.append(overlap)
                    else:
                        unselect.append(overlap)
    return select, unselect


def get_hist_data(epoch_metric, typ):
    match typ:
        case "overlap":
            fn = normed_overlap_un_select_one_epoch
        case "iou":
            fn = iou_un_select_one_epoch
        case "iou-vs-rand":
            fn = iou_un_select_one_epoch_vs_random
        case _:
            print(f"{typ} not supported")
            sys.exit()
    sel, uns = fn(epoch_metric)
    return [np.array(sel), np.array(uns)]


def iou_per_epoch(params, selected):
    n_iter, n_crops, n_params = params.shape

    select = []
    not_select = []
    for i in range(n_iter):
        m, n = selected[i]
        p1 = params[i, m][:6]
        p2 = params[i, n][:6]
        value = calc_iou(p1, p2)
        select.append(value)
        if m == 0 and n == 1:
            not_select.append(value)
        else:
            p1 = params[i, 0][:6]
            p2 = params[i, 1][:6]
            value = calc_iou(p1, p2)
            not_select.append(value)
    return select, not_select


def iou_data(params, selected):
    epochs = len(params)
    out1, out2 = [], []
    for n in range(epochs):
        x, y = iou_per_epoch(params[n], selected[n])
        out1.extend(x)
        out2.extend(y)
    return [np.array(out1), np.array(out2)]


if __name__ == "__main__":
    path = "C:/Users/ovi/Documents/SelfSupervisedLearning/metrics/pickle/simsiam-minsim-resnet50-ep100-seed0.pkl"
    metrics = pd.read_pickle(path)

    data = iou_data(metrics["params"], metrics["selected"])

    num_bins = 20
    labels = "selected", "random"
    xlabel = "IoU"
    ylabel = "Density"
    title = "IoU - Selected vs Random"

    fig, ax = plt.subplots()
    ax.hist(data, num_bins, density=True, label=labels, range=(0.1, 1.))
    ax.legend(loc='upper right')
    # ax.set_title(file_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    # fig.text(0.5, 0.01, exp_name, fontsize='x-small', ha='center')
    plt.tight_layout()

    plt.show()
    # out_dir = Path(path)
    # out_dir.mkdir(parents=True, exist_ok=True)
    # save_path = out_dir.joinpath(file_name)
    fig.savefig("all.png")
    # print(f"Plot saved at {save_path}.png")
    # plt.close(fig)

