"""Optimize hyperparameters for lightgbm"""

import numpy as np
import pandas as pd
import lightgbm
import hyperopt
from hyperopt import hp

import sys

__DATA_VER__ = "0.6"
VALIDATION_RANGE = 1

PROCESSED_PATH = f"./processed_data/data_{__DATA_VER__}.pickle"

SPACE = dict(
    silent=True,
    random_state=284704,
    learning_rate=0.3,
    n_estimators=200,
    max_depth=hp.choice("depth", list(range(-1, 32))),
    num_leaves=hp.loguniform("l2_leaf_reg", np.log(2), np.log(10000)),
    cat=hp.choice("cat", list(range(2)))
)


def get_processed_data():
    return pd.read_pickle(PROCESSED_PATH)


def get_train_val_split():
    data = get_processed_data()
    test_date_block = data.date_block_num.max()

    x_train = data[data.date_block_num < test_date_block - VALIDATION_RANGE].drop("y", axis=1)
    y_train = data.loc[data.date_block_num < test_date_block - VALIDATION_RANGE, "y"]

    mask = (test_date_block - VALIDATION_RANGE <= data.date_block_num) & (data.date_block_num < test_date_block)
    x_valid = data[mask].drop("y", axis=1)
    y_valid = data.loc[mask, "y"]

    return x_train, y_train, x_valid, y_valid


class LightGBMHyper:
    def __init__(self, train, valid):
        self._train = train
        self._valid = valid
        self._counter = 0
        self._rmse = None
        self._clf = None

    def __call__(self, params):
        params["num_leaves"] = int(params["num_leaves"])
        cat = params.pop("cat")
        clf = lightgbm.LGBMRegressor(**params)
        cat_columns = [i for i, col in enumerate(x_train) if not issubclass(x_train[col].dtype.type, np.floating)]
        clf.fit(
            *self._train,
            early_stopping_rounds=20,
            eval_set=[self._valid],
            verbose=False,
            categorical_feature=cat_columns if cat else 'auto'
        )

        rmse = clf.best_score_["valid_0"]["l2"] ** 0.5
        print(f"Run {self._counter}: RMSE - {rmse:0.5f}")
        self._counter += 1
        if self._rmse is None or rmse < self._rmse:
            self._rmse = rmse
            self._clf = clf
            print(f"Best run: params - {clf.get_params()}")
            print(f"{'Cat used' if cat else 'Cat not used'}")
            print()
        return rmse

    @property
    def best_clf(self):
        return self._clf


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = get_train_val_split()

    opt = LightGBMHyper((x_train, y_train), (x_valid, y_valid))
    hyperopt.fmin(
        opt,
        space=SPACE,
        algo=hyperopt.tpe.suggest,
        max_evals=100)
