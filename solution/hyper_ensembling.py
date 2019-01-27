"""Optimize meta model by Ridge"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import hyperopt
from hyperopt import hp

import sys

__DATA_VER__ = "0.6"
VALIDATION_RANGE = 1

PROCESSED_PATH = f"./processed_data/data_{__DATA_VER__}.pickle"


SPACE = dict(
    random_state=284704,
    alpha=hp.loguniform("alpha", np.log(10 ** -1), np.log(10 ** 7))
)


def get_train_val_split():
    x_train = pd.read_pickle("./processed_data/meta_train.pickle")
    y_train = pd.read_pickle("./processed_data/meta_y_train.pickle")

    x_valid = pd.read_pickle("./processed_data/meta_valid.pickle")
    y_valid = pd.read_pickle("./processed_data/meta_y_valid.pickle")

    return x_train, y_train, x_valid, y_valid


class EnsemblingHyperRidge:
    def __init__(self, train, valid):
        self._train = train
        self._valid = valid
        self._counter = 0
        self._rmse = None
        self._clf = None

    def __call__(self, params):

        train_x, train_y = self._train
        valid_x, valid_y = self._valid
        clf = Ridge(**params)
        clf.fit(
            train_x,
            train_y,
        )

        rmse = mean_squared_error(clf.predict(valid_x), valid_y) ** 0.5
        print(f"Run {self._counter}: RMSE - {rmse:0.5f}")
        self._counter += 1
        if self._rmse is None or rmse < self._rmse:
            self._rmse = rmse
            self._clf = clf
            print(f"Best run: params - {params}")
            print()
        return rmse


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = get_train_val_split()

    opt = EnsemblingHyperRidge((x_train, y_train), (x_valid, y_valid))
    hyperopt.fmin(
        opt,
        space=SPACE,
        algo=hyperopt.tpe.suggest,
        max_evals=1000)
