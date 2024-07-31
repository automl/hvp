from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np


if __name__ == "__main__":
    X = pd.read_csv("X.csv")
    Y = pd.read_csv("Y.csv")
    X_train, X_test = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):]
    Y_train, Y_test = Y[:int(len(Y) * 0.8)].squeeze(), Y[int(len(Y) * 0.8):].squeeze()
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    clf = GradientBoostingClassifier(random_state=0, verbose=True).fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    print(score)
    # 0.3453756322987573
