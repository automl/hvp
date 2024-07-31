import math
import numpy as np

from plot_utils import read_file, save_plot


def calc_rel_dist(x1, x2):
    h, w = x1[0], x1[1]
    c1x, c1y = x1[2], x1[3]
    c2x, c2y = x2[2], x2[3]
    return math.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2) / math.sqrt(h ** 2 + w ** 2)


def _calc_avg_dist_one_epoch(metrics_epoch):
    params = metrics_epoch["params"]
    selected = metrics_epoch["selected"]

    dist_sel = []
    dist_oth = []
    for i, itr in enumerate(params):
        sel = np.array(selected[i])
        for c in range(len(itr[0])):
            prms = [item[c][0] for item in itr]
            ss = sel[:, c]
            for n, x1 in enumerate(prms):
                for m, x2 in enumerate(prms[n + 1:], n + 1):
                    dst = calc_rel_dist(x1, x2)
                    if np.equal(np.array([n, m]), ss).all():
                        dist_sel.append(dst)
                    else:
                        dist_oth.append(dst)

    dist_sel = np.array(dist_sel)
    dist_oth = np.array(dist_oth)
    return dist_sel.mean(), dist_sel.std(), dist_oth.mean(), dist_oth.std()


def get_relative_distance(metrics):
    results = np.array([_calc_avg_dist_one_epoch_vs_random(epoch) for epoch in metrics]).T
    return [results[0], results[1]], [results[2], results[3]]


def _calc_avg_dist_one_epoch_vs_random(metrics_epoch):
    params = metrics_epoch["params"]
    selected = metrics_epoch["selected"]

    dist_sel = []
    dist_oth = []
    for i, itr in enumerate(params):
        sel = np.array(selected[i])
        for c in range(len(itr[0])):
            prms = [item[c][0] for item in itr]
            ss = sel[:, c]
            for n, x1 in enumerate(prms):
                for m, x2 in enumerate(prms[n + 1:], n + 1):
                    dst = calc_rel_dist(x1, x2)
                    if np.equal(np.array([n, m]), ss).all():
                        dist_sel.append(dst)
                    if n == 0 and m == 1:
                        dist_oth.append(dst)

    dist_sel = np.array(dist_sel)
    dist_oth = np.array(dist_oth)
    return dist_sel.mean(), dist_oth.mean(), dist_sel.std(),  dist_oth.std()


def get_relative_distance_vs_random(metrics):
    results = np.array([_calc_avg_dist_one_epoch_vs_random(epoch) for epoch in metrics]).T
    return [results[0], results[1]], [results[2], results[3]]


if __name__ == "__main__":
    seeds = 0, 1, 2
    for seed in seeds:
        data_path = f"../../exp_data/metrics{seed}.json"
        metrics = read_file(data_path)

        exp_name = f"simsiam-minsim-collect_metrics-resnet50-ImageNet-ep100-bs256-select_cross-ncrops4-lr0.05-wd0.0001-mom0.9-seed{seed}"
        path = f"plots/{exp_name}/relative-distance-of-centerpoints"
        file_name = f"distance_seed{seed}.png"  # name.png

        data, std = get_relative_distance(metrics)
        # data, std = get_relative_distance_vs_random(metrics)
        labels = "selected", "not-selected"

        title = "relative distance between center-points (avg. per epoch)"
        xlabel = "epoch"
        ylabel = "relative-distance"

        save_plot(
            data=data, std=std, labels=labels,
            title=title, xlabel=xlabel, ylabel=ylabel,
            path=path, file_name=file_name, exp_name=exp_name)
