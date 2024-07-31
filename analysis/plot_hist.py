import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

NUM_BINS = 20
NUM_CROPS = 4


def get_dims(x):
    return x[2], x[2] + x[4], x[3], x[3] + x[5]


def calc_overlap(x1, x2):
    l1, r1, t1, b1 = get_dims(x1)
    l2, r2, t2, b2 = get_dims(x2)
    w = max(0, min(r1, r2) - max(l1, l2))
    h = max(0, min(b1, b2) - max(t1, t2))
    return w * h


def calc_overlap_norm(x1, x2):
    area = calc_overlap(x1, x2)
    norm = x1[0] * x1[1]
    return area / norm


def calc_area(x):
    w, h = x[-2:]
    return w * h


def calc_iou(x1, x2):
    A = calc_area(x1)
    B = calc_area(x2)
    i = calc_overlap(x1, x2)
    return i / (A + B - i)


def iou_un_select_one_epoch(epoch_metrics):
    params_epoch = epoch_metrics["params"]
    selected_epoch = epoch_metrics["selected"]
    select = []
    unselect = []
    for i, iteration in enumerate(params_epoch):
        selected = np.array(selected_epoch[i])
        for c in range(len(iteration[0])):
            params = [item[c][0] for item in iteration[:NUM_CROPS]]
            ss = selected[:, c]
            for n, x1 in enumerate(params):
                for m, x2 in enumerate(params[n + 1:], n + 1):
                    overlap = calc_iou(x1, x2)
                    if np.equal(np.array([n, m]), ss).all():
                        select.append(overlap)
                    else:
                        unselect.append(overlap)
    return select, unselect


def avg_least_typ_selected_one_epoch(epoch_metrics, typ):
    if typ == "overlap":
        fn = calc_overlap
    else:
        fn = calc_iou
    params_epoch = epoch_metrics["params"]
    selected_epoch = epoch_metrics["selected"]
    hits = 0
    count = 0
    for i, iteration in enumerate(params_epoch):
        selected = np.array(selected_epoch[i])
        for c in range(len(iteration[0])):
            params = [item[c][0] for item in iteration[:NUM_CROPS]]
            ss = selected[:, c]
            min_overlap = np.inf
            indices = [0, 0]
            for n, x1 in enumerate(params):
                for m, x2 in enumerate(params[n + 1:], n + 1):
                    overlap = fn(x1, x2)
                    if overlap < min_overlap:
                        indices = [n, m]
                        min_overlap = overlap
            if np.equal(np.array(indices), ss).all():
                hits += 1
            count += 1
    return hits / count, hits, count


def normed_overlap_un_select_one_epoch(epoch_metrics):
    params_epoch = epoch_metrics["params"]
    selected_epoch = epoch_metrics["selected"]
    select = []
    unselect = []
    for i, iteration in enumerate(params_epoch):
        selected = np.array(selected_epoch[i])
        for c in range(len(iteration[0])):
            params = [item[c][0] for item in iteration[:NUM_CROPS]]
            ss = selected[:, c]
            for n, x1 in enumerate(params):
                for m, x2 in enumerate(params[n + 1:], n + 1):
                    overlap = calc_overlap_norm(x1, x2)
                    if np.equal(np.array([n, m]), ss).all():
                        select.append(overlap)
                    else:
                        unselect.append(overlap)
    return select, unselect


def save_plot_avg_select_least_typ(metrics, typ):
    assert typ in ["overlap", "iou"]
    y = []
    for epoch in metrics:
        avg, _, _ = avg_least_typ_selected_one_epoch(epoch, typ)
        y.append(avg)

    X = np.arange(len(y))
    fig, ax = plt.subplots()
    ax.plot(X, y)
    # ax.set(xlim=(-5, len(y)+5),
    #        ylim=(0, 1))
    ax.grid()
    ax.set_title(f"Avg selected pair by {typ}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(typ)
    fig.savefig(f"plots/avg-select-least-typ/plot_{typ}.png")


def create_hist(epoch_metric, typ):
    fn = normed_overlap_un_select_one_epoch if typ == "norm" else iou_un_select_one_epoch
    sel, uns = fn(epoch_metric)
    fig, ax = plt.subplots()
    ax.hist([sel, uns], NUM_BINS, density=True, label=['selected', 'not-selected'])
    ax.legend(loc='upper right')
    txt = "normalized overlap (w.r.t. original image)" if typ == "norm" else "IoU"
    ax.set_xlabel(txt)
    return fig


def save_all_hists(metrics, typ="norm"):
    assert typ in ["norm", "iou"]
    for n, epoch in enumerate(metrics):
        fig = create_hist(epoch, typ)
        fig.suptitle(f"Epoch {n}")
        fig.savefig(f"plots/histograms-{typ}/epoch_{n}.png")
        plt.close(fig)


if __name__ == "__main__":
    data_path = "../../exp_data/metrics0_dino.json"
    with open(data_path, "r") as file:
        metrics = list(map(json.loads, file.readlines()))

    # save_all_hists(metrics, "iou")
    save_plot_avg_select_least_typ(metrics, "iou")
