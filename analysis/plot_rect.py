import json
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as colors

NUM_CROPS = 4


def rect_last_iter(metrics):
    """Plots the rectangles of the last iteration of every epoch (1st 4-tuple)."""
    for epoch, line in enumerate(metrics):
        params = line["params"][-1][:NUM_CROPS]
        selected = np.array(line["selected"][-1])[:, 0]
        fig, ax = plt.subplots(figsize=(5, 5))
        for n, (data, color) in enumerate(zip(params, colors.BASE_COLORS)):
            style = '-' if n in selected else ':'
            w, h, a, b, xw, yh = data[0][0]
            x = np.array([a, a+xw, a+xw, a, a]) / w
            y = np.array([b, b, b+yh, b+yh, b]) / h

            ax.plot(x, y, linestyle=style, color=color, linewidth=2.0)

        ax.set(xlim=(0, 1), xticks=[],
               ylim=(0, 1), yticks=[])
        ax.set_title(f"Epoch {epoch}")

        plt.savefig(f"plots/rectangles/epoch_{epoch}.png")


def rect_all_iter(metrics):
    """Plots the rectangles for the first 4-tuple of crops in every saved iteration."""
    for epoch, line in enumerate(metrics):
        for m, params in enumerate(line["params"]):
            selected = np.array(line["selected"][m])[:, 0]  # first 4-tuple
            fig, ax = plt.subplots(figsize=(5, 5))
            for n, (data, color) in enumerate(zip(params[:NUM_CROPS], colors.BASE_COLORS)):
                style = '-' if n in selected else ':'
                w, h, a, b, xw, yh = data[0][0]  # first 4-tuple, geometric (RRC) parameters
                x = np.array([a, a+xw, a+xw, a, a]) / w
                y = np.array([b, b, b+yh, b+yh, b]) / h

                ax.plot(x, y, linestyle=style, color=color, linewidth=2.0)

            ax.set(xlim=(0, 1), xticks=[],
                   ylim=(0, 1), yticks=[])
            ax.set_title(f"Epoch {epoch} - Iteration {m}")

            plt.savefig(f"plots/rectangles/epoch_{epoch}_{m}.png")
            plt.close()


if __name__ == "__main__":
    # read data
    data_path = "../../exp_data/metrics0_dino.json"
    with open(data_path, "r") as file:
        # metrics = json.loads(file.readline())
        metrics_file = list(map(json.loads, file.readlines()))

    print(f"Num epochs: {len(metrics_file)}")
    print(f"Keys: {metrics_file[-1].keys()}")

    rect_last_iter(metrics_file)
