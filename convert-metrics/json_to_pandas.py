import os

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


def convert_params_to_list(raw, epoch):
    out = []
    bs = len(raw[0][0])
    for mini_batch in raw:
        for i in range(bs):
            temp = []
            for j in range(len(mini_batch)):
                extracted = extract_params(mini_batch[j][i])
                temp.extend(extracted)
            temp.append(epoch)
            out.append(temp)
    return out


def convert_selected_to_int(raw):
    out = []
    for mini_batch in raw:
        raw = np.array(mini_batch)
        for sample in raw.T:
            match list(sample):
                case [0, 1]:
                    n = 0
                case [0, 2]:
                    n = 1
                case [0, 3]:
                    n = 2
                case [1, 2]:
                    n = 3
                case [1, 3]:
                    n = 4
                case [2, 3]:
                    n = 5
            out.append(n)
    return out


def json_to_pandas(json_file):
    X = []
    Y = []
    for line in json_file:
        epoch = line['epoch']
        params = convert_params_to_list(line['params'], epoch)
        X.extend(params)
        Y.extend(convert_selected_to_int(line['selected']))
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    path = ...
    data = read_file(path)

    X_array, Y_array = json_to_pandas(data)

    df_X = pd.DataFrame(X_array)
    df_Y = pd.DataFrame(Y_array)

    head, tail = os.path.split(path)
    x_out_path = os.path.join(tail, "X-" + tail.replace(".json", ".csv"))
    y_out_path = os.path.join(tail, "Y-" + tail.replace(".json", ".csv"))
    df_X.to_csv(x_out_path, index=False)
    df_Y.to_csv(y_out_path, index=False)
