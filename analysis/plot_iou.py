import numpy as np

from plot_utils import read_file, calc_iou, calc_overlap, calc_area, save_plot


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


def get_iou_data(metrics):
    results = np.array([_avg_value_selection_one_epoch(epoch, "iou") for epoch in metrics]).T
    return results[0], results[1]


def get_overlap_data(metrics):
    results = np.array([_avg_value_selection_one_epoch(epoch, "overlap") for epoch in metrics]).T
    return results[0], results[1]


if __name__ == "__main__":
    seeds = 0, 1, 2
    for seed in seeds:
        data_path = f"../../exp_data/metrics{seed}.json"
        metrics = read_file(data_path)

        exp_name = f"simsiam-minsim-collect_metrics-resnet50-ImageNet-ep100-bs256-select_cross-ncrops4-lr0.05-wd0.0001-mom0.9-seed{seed}"
        path = f"plots/{exp_name}/iou-avg"
        file_name = f"iou_seed{seed}.png"  # name.png

        data = get_iou_data(metrics)
        labels = "selected", "not-selected"

        title = "Average IoU of selected/not-selected pairs"
        xlabel = "epoch"
        ylabel = "IoU"

        save_plot(data=data, labels=labels,
                  title=title, xlabel=xlabel, ylabel=ylabel,
                  path=path, file_name=file_name, exp_name=exp_name)
