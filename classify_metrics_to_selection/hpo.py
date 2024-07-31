import argparse

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd
import numpy as np
from scipy.stats import loguniform, randint, uniform


def function(x_train, y_train, x_test, y_test):
    clf = HistGradientBoostingClassifier()
    param_distributions = {
        "learning_rate": loguniform(1e-4, 1e0),
        "max_leaf_nodes": [randint(11, 51), None],
        "min_samples_leaf": randint(10, 40),
        "l2_regularization": uniform(0.0, 1.0),
    }
    gsh = HalvingRandomSearchCV(clf, param_distributions, n_jobs=-1, verbose=10)
    gsh.fit(x_train, y_train)
    print(gsh.best_params_)
    print(gsh.score(x_test, y_test))


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
