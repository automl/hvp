import csv
import json
import math
import numpy as np

from plot_relative_dist import calc_rel_dist
from utils import read_file, calc_iou


def calc_clr_dist(x1, x2):
    a1, b1, c1, d1, *e = x1 if x1 else (0, 0, 0, 0, 0)
    a2, b2, c2, d2, *e = x2 if x2 else (0, 0, 0, 0, 0)
    return math.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2 + (c1 - c2) ** 2 + (d1 - d2) ** 2)


# Read in the JSON file
path = "../../exp_data/metrics0.json"
with open(path, 'r') as file:
    data = list(map(json.loads, file.readlines()))

# print(len(data[0]['sample-loss']))
# print(len(data[0]['params'][0]))
# print(data[0]['params'][0])
#
# print(len(data[0]['selected']))
# print(len(data[0]['selected'][0]))
# print(data[0]['selected'][0])
# print(len(data[0]['params']))


sample_loss = data[-1]['sample-loss']
selected = data[-1]['selected']
params = data[-1]['params']

iou = []
dst = []
clr = []
brg = []
hue = []
con = []
sat = []
aug = []
loss = []

for n in range(len(sample_loss)):
    loss.extend(sample_loss[n])
    selects = np.array(selected[n]).T
    params_i = params[n]
    print(selects.shape)
    for m in range(selects.shape[0]):
        a, b = selects[m]
        p1, p2 = params_i[a][m], params_i[b][m]

        tmp = 0
        x = calc_iou(p1[0], p2[0])
        iou.append(x)
        tmp += x
        x = calc_rel_dist(p1[0], p2[0])
        dst.append(x)
        tmp += x
        x = calc_clr_dist(p1[1], p2[1])
        clr.append(x)
        if p1[1] and p2[1]:
            brg.append(abs(p1[1][0] - p2[1][0]))
            con.append(abs(p1[1][1] - p2[1][1]))
            sat.append(abs(p1[1][2] - p2[1][2]))
            hue.append(abs(p1[1][3] - p2[1][3]))
        else:
            brg.append(0)
            con.append(0)
            sat.append(0)
            hue.append(0)
        tmp += x
        if p1[2] ^ p2[2]:
            tmp += 1
        if bool(p1[3]) ^ bool(p2[3]):
            tmp += 1
        if p1[4] ^ p2[4]:
            tmp += 1
        aug.append(tmp)

with open('features.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the data
    writer.writerow(iou)
    writer.writerow(dst)
    writer.writerow(clr)
    writer.writerow(brg)
    writer.writerow(con)
    writer.writerow(sat)
    writer.writerow(hue)
    writer.writerow(aug)

with open('responses.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the data
    writer.writerow(loss)
