import json
from typing import Union, List, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BASE_COLORS
import numpy as np


def read_file(path):
    with open(path, "r") as file:
        data = list(map(json.loads, file.readlines()))
    return data


def save_plot(
        data: Union[np.ndarray, List[np.ndarray]],
        std: Union[np.ndarray, List[np.ndarray]] = None,
        labels: Union[str, List[str], Tuple[str, ...]] = None,
        title: str = "Title",
        xlabel: str = "xlabel",
        ylabel: str = "ylabel",
        path: str = "./",
        file_name: str = "out.png",
        exp_name: str = "",
):
    if isinstance(data, np.ndarray):
        data = [data]

    if not isinstance(std, list):
        std = [None for _ in data] if std is None else [std]
    assert len(data) == len(std)

    if not isinstance(labels, Union[List, Tuple]):
        labels = np.arange(len(data)) if labels is None else [labels]
    assert len(data) == len(labels)

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

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir.joinpath(file_name)
    fig.savefig(save_path)
    print(f"Plot saved at {save_path}")
    plt.close(fig)


def save_histogram(
        data: Union[np.ndarray, List[np.ndarray]],
        labels: Union[str, List[str], Tuple[str, ...]] = None,
        num_bins: int = 10,
        range: tuple = (0.1, 1.),
        title: str = "Title",
        xlabel: str = "xlabel",
        ylabel: str = "ylabel",
        path: str = "./",
        file_name: str = "out.png",
        exp_name: str = "",
):
    fig, ax = plt.subplots()
    ax.hist(data, num_bins, density=True, label=labels, range=range)
    ax.legend(loc='upper right')
    ax.set_title(file_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.text(0.5, 0.01, exp_name, fontsize='x-small', ha='center')
    plt.tight_layout()

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir.joinpath(file_name)
    fig.savefig(save_path)
    print(f"Plot saved at {save_path}.png")
    plt.close(fig)


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
