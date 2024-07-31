import argparse

from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd
import numpy as np


def function(x_train, y_train, x_test, y_test):
    clf = HistGradientBoostingClassifier(max_iter=1000, verbose=True).fit(x_train, y_train)
    print(clf.score(x_test, y_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Using GradientBoostingClassifier to predict the selection of crops')
    parser.add_argument('-x', type=str, nargs='+', help='csv-files with metrics')
    parser.add_argument('-y', type=str, nargs='+', help='csv-files with selection')
    args = parser.parse_args()

    assert len(args.x) == len(args.y)

    X = []
    for x in args.x:
        tmp = pd.read_csv(x)
        X.append(tmp)
    X = np.concatenate(X, axis=0)

    Y = []
    for y in args.y:
        tmp = pd.read_csv(y).squeeze()
        Y.append(tmp)
    Y = np.concatenate(Y, axis=0)

    function(X, Y, X, Y)
