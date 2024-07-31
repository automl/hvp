import os
import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import shuffle


if __name__ == "__main__":
    w_dir = "../../metrics-data"
    X_0 = pd.read_csv(os.path.join(w_dir, "X-simsiam-minsim-seed0.csv"))
    X_1 = pd.read_csv(os.path.join(w_dir, "X-simsiam-minsim-seed1.csv"))
    X_2 = pd.read_csv(os.path.join(w_dir, "X-simsiam-minsim-seed2.csv"))
    X = np.concatenate((X_0, X_1), axis=0)
    print(X.shape)
    Y_0 = pd.read_csv(os.path.join(w_dir, "Y-simsiam-minsim-seed0.csv")).squeeze()
    Y_1 = pd.read_csv(os.path.join(w_dir, "Y-simsiam-minsim-seed1.csv")).squeeze()
    Y_2 = pd.read_csv(os.path.join(w_dir, "Y-simsiam-minsim-seed2.csv")).squeeze()
    Y = np.concatenate((Y_0, Y_1), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    X_shuffle, Y_shuffle = shuffle(X, Y, random_state=0)

    clf = HistGradientBoostingClassifier(verbose=True)
    clf.fit(X_shuffle, Y_shuffle)
    print(clf.classes_)
    print(clf.n_features_in_)
    score = clf.score(X_2[int(len(X_2) * 0.5):], Y_2[int(len(X_2) * 0.5):])
    print(score)
    # scores = cross_val_score(clf, X, Y, cv=5, verbose=1)
    # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
