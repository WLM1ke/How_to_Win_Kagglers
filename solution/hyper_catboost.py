"""Optimize hyperparameters for catboost"""

import numpy as np
import pandas as pd
import catboost
import hyperopt
from hyperopt import hp

import sys

__DATA_VER__ = "0.6"
VALIDATION_RANGE = 1

PROCESSED_PATH = f"./processed_data/data_{__DATA_VER__}.pickle"

SPACE = dict(
    random_state=284704,
    od_type="Iter",
    learning_rate=0.3,
    task_type="GPU",
    verbose=False,
    depth=hp.choice("depth", list(range(1, 17))),
    l2_leaf_reg=hp.loguniform("l2_leaf_reg", np.log(0.1), np.log(10000)),
)

if sys.platform == "darwin":
    del SPACE["task_type"]


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


class CatboostHyper:
    def __init__(self, train, valid):
        self._train = train
        self._valid = valid
        self._counter = 0
        self._rmse = None
        self._clf = None

    def __call__(self, params, plot=False):
        clf = catboost.CatBoostRegressor(**params)
        clf.fit(
            X=self._train,
            eval_set=self._valid
        )
        rmse = clf.get_best_score()["validation_0"]["RMSE"]
        print(f"Run {self._counter}: "
              f"RMSE - {rmse:0.5f}, "
              f"Best Iteration - {clf.get_best_iteration() + 1}"
              )
        self._counter += 1
        if self._rmse is None or rmse < self._rmse:
            self._rmse = rmse
            self._clf = clf
            print(f"Best run: params - {clf.get_params()}")
            print()
        return rmse

    def feature_importance(self):
        clf = self.best_clf
        print(f"Params - {clf.get_params()}")
        for i, v in clf.get_feature_importance(prettified=True):
            print(i.ljust(20), v)
        print()
        for i, j, value in clf.get_feature_importance(fstr_type="Interaction", prettified=True)[:10]:
            print(x_train.columns[i].ljust(20), x_train.columns[j].ljust(20), value)

    @property
    def best_clf(self):
        return self._clf


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = get_train_val_split()
    cat_columns = [i for i, col in enumerate(x_train) if not issubclass(x_train[col].dtype.type, np.floating)]

    train = catboost.Pool(
        data=x_train,
        label=y_train,
        cat_features=cat_columns
    )

    valid = catboost.Pool(
        data=x_valid,
        label=y_valid,
        cat_features=cat_columns
    )

    opt = CatboostHyper(train, valid)
    hyperopt.fmin(
        opt,
        space=SPACE,
        algo=hyperopt.tpe.suggest,
        max_evals=100)
    opt.feature_importance()
