import sys
import numpy as np

from plot_utils import read_file, calc_area, calc_overlap, calc_iou, calc_overlap_norm, save_histogram


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


if __name__ == "__main__":
    seeds = 0, 1, 2
    for seed in seeds:
        data_path = f"../../exp_data/metrics{seed}.json"
        metrics = read_file(data_path)

        typ = "iou"  # "iou" "overlap" "iou-vs-rand"
        exp_name = f"simsiam-minsim-collect_metrics-resnet50-ImageNet-ep100-bs256-select_cross-ncrops4-lr0.05-wd0.0001-mom0.9-seed{seed}"
        path = f"plots/{exp_name}/histograms-{typ}"

        title = f"{typ} of Selected vs Not-Selected"  # "IoU of Selected vs Not-Selected (Random)"
        xlabel = f"{typ}"
        ylabel = "density"

        every_x_epochs = 10
        num_bins = 20
        labels = "selected", "not-selected"
        for n, epoch in enumerate(metrics):
            if n % every_x_epochs:
                continue
            data = get_hist_data(epoch, typ)

            file_name = f"epoch{n}.png"  # name.png
            save_histogram(
                data=data, labels=labels,
                title=title, xlabel=xlabel, ylabel=ylabel,
                path=path, file_name=file_name, exp_name=exp_name
            )

