"""
Converts the json file to a pandas dataframe
"""

import numpy as np
import pandas as pd

from utils import extract_list, read_file


def extract_params(raw):
    rrc = extract_list(raw[0])
    # rrc = [rrc[2] / rrc[0], rrc[3] / rrc[1], rrc[4] / rrc[0], rrc[5] / rrc[1]]
    jitter = extract_list(raw[1]) if raw[1] else [0, 0, 0, 0, 0, 0, 0, 0]
    gray = 1 if raw[2] else 0
    blur = raw[3] if raw[3] else 0
    hf = 1 if raw[4] else 0
    return *rrc, *jitter, gray, blur, hf


def extract_params_static(raw):
    """
    the static color experiments have a different structure
    rrc, hf, jitter, gray, blur
    """
    rrc = extract_list(raw[0])
    # rrc = [rrc[2] / rrc[0], rrc[3] / rrc[1], rrc[4] / rrc[0], rrc[5] / rrc[1]]
    jitter = extract_list(raw[2]) if raw[2] else [0, 0, 0, 0, 0, 0, 0, 0]
    gray = 1 if raw[3] else 0
    blur = raw[4] if raw[3] else 0
    hf = 1 if raw[1] else 0
    return *rrc, *jitter, gray, blur, hf


if __name__ == '__main__':
    path = ...
    data = read_file(path)

    # convert the params of an epoch from a nested list to np.ndarray
    for n in range(len(data)):
        selected = data[n]['params']
        out = [[] for _ in range(4)]
        for i in range(len(selected)):
            for j in range(len(selected[i])):
                for p in selected[i][j]:
                    out[j].append(extract_params_static(p))
        out = np.array(out)
        out = np.swapaxes(out, 0, 1)
        data[n]['params'] = out

    for n in range(len(data)):
        selected = data[n]['selected']
        out = [[] for _ in range(2)]
        for i in range(len(selected)):
            for j in range(2):
                out[j].extend(selected[i][j])
        out = np.array(out)
        out = np.swapaxes(out, 0, 1)
        data[n]['selected'] = out

    for n in range(len(data)):
        lr = data[n]['lr']
        out = np.array(lr)
        data[n]['lr'] = out

        loss = data[n]['loss']
        out = np.array(loss)
        data[n]['loss'] = out

        loss = data[n]['sample-loss']
        out = np.array(loss)
        data[n]['sample-loss'] = out

    df = pd.DataFrame(data)
    out_path = path[:-4] + "pkl"
    df.to_pickle(out_path)
