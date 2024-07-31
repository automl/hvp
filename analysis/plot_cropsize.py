import numpy as np

from plot_utils import read_file, save_plot


def get_crop_size(x1, x2):
    gp1, gp2 = x1[0], x2[0]
    cs1, cs2 = calc_cs(gp1), calc_cs(gp2)
    return min(cs1, cs2), max(cs1, cs2)


def calc_cs(gp):
    H, W, _, _, h, w = gp
    return h / H * w / W


def _get_cropsize_one_epoch_avg_min_max(epoch_metrics):
    params_epoch = epoch_metrics["params"]
    selected = epoch_metrics["selected"]
    val1, val2 = [], []

    for i, iteration in enumerate(params_epoch):
        selection = np.array(selected[i])

        for idx in range(len(iteration)):
            params = [item[idx] for item in iteration]
            ss = selection[:, idx]
            for n, x1 in enumerate(params):
                for m, x2 in enumerate(params[n + 1:], n + 1):
                    cs1, cs2 = get_crop_size(x1, x2)
                    if np.equal(np.array([n, m]), ss).all():
                        val1.append(cs1)
                        val2.append(cs2)

    v1, v2 = np.array(val1), np.array(val2)
    return v1.mean(), v2.mean(), v1.std(),  v2.std()


def get_cropsize(metrics):
    results = np.array([_get_cropsize_one_epoch_avg_min_max(epoch) for epoch in metrics]).T
    return [results[0], results[1]], [results[2], results[3]]


if __name__ == "__main__":
    seeds = 0, 1, 2
    for seed in seeds:
        data_path = f"../../exp_data/metrics{seed}.json"
        metrics = read_file(data_path)

        exp_name = f"simsiam-minsim-collect_metrics-resnet50-ImageNet-ep100-bs256-select_cross-ncrops4-lr0.05-wd0.0001-mom0.9-seed{seed}"
        path = f"plots/{exp_name}/crop-size"
        file_name = f"cropsize_seed{seed}.png"  # name.png

        data, std = get_cropsize(metrics)
        labels = "crop1", "crop2"

        title = "Crop-Size of selected crops"
        xlabel = "epoch"
        ylabel = "crop-size"

        save_plot(
            data=data, std=std, labels=labels,
            title=title, xlabel=xlabel, ylabel=ylabel,
            path=path, file_name=file_name, exp_name=exp_name)
