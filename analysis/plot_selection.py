import numpy as np
import matplotlib.pyplot as plt

from plot_utils import read_file, calc_iou, calc_overlap, calc_area, save_plot


def _avg_least_typ_selected_one_epoch(epoch_metrics, typ):
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
            params = [item[c][0] for item in iteration]
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
    return hits / count


def get_selection_by_iou(metrics):
    results = np.array([_avg_least_typ_selected_one_epoch(epoch, "iou") for epoch in metrics]).T
    return results


def get_selection_by_overlap(metrics):
    results = np.array([_avg_least_typ_selected_one_epoch(epoch, "overlap") for epoch in metrics]).T
    return results


if __name__ == "__main__":
    seeds = 0, 1, 2
    for seed in seeds:
        data_path = f"../../exp_data/metrics{seed}.json"
        metrics = read_file(data_path)

        exp_name = f"simsiam-minsim-collect_metrics-resnet50-ImageNet-ep100-bs256-select_cross-ncrops4-lr0.05-wd0.0001-mom0.9-seed{seed}"
        path = f"plots/{exp_name}/iou-selection"
        file_name = f"selection_seed{seed}.png"  # name.png

        data = get_selection_by_iou(metrics)
        labels = "avg per epoch"

        title = "How many selected MinSim-pairs are also Min-IoU?"
        xlabel = "epoch"
        ylabel = "% selected"

        save_plot(data=data, labels=labels,
                  title=title, xlabel=xlabel, ylabel=ylabel,
                  path=path, file_name=file_name, exp_name=exp_name)
