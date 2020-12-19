from skimage.metrics import mean_squared_error

from .base_preprocessor import BasePreprocessor
from .preprocessors import *
from .base_regressor import BaseRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
import random
import numpy as np
import pandas as pd
SEED = 14300631
N_FOLDS = 10

random.seed(SEED)
np.random.seed(SEED)


class LinearReg(BaseRegressor):
    def __init__(self, path_to_raw):
        self.preprocessor = LinearPrep(path_to_raw)

    def create_model_and_fit(self, X_train, y_train, X_val, y_val, hparams=None):
        if hparams is None:
            self.preprocessor.fit(X_train, y_train)
            X_train = self.preprocessor.transform(X_train)
            X_val = self.preprocessor.transform(X_val)
            for col in X_train.columns:
                mean_value = X_train[col].mean()
                X_train[col] = X_train[col].fillna(mean_value)
                X_val[col] = X_val[col].fillna(mean_value)
                X_val[col] = X_val[col].fillna(mean_value)

            model = ElasticNet()
            model.fit(
                X_train,
                y_train,
            )
        else:
            self.preprocessor.fit(X_train, y_train)
            X_train = self.preprocessor.transform(X_train)
            X_val = self.preprocessor.transform(X_val)
            for col in X_train.columns:
                mean_value = X_train[col].mean()
                X_train[col] = X_train[col].fillna(mean_value)
                X_val[col] = X_val[col].fillna(mean_value)
                X_val[col] = X_val[col].fillna(mean_value)

            model = ElasticNet(**hparams)
            model.fit(
                X_train,
                y_train,
            )

        return model, mean_squared_error(model.predict(X_val), y_val)


class CatReg(BaseRegressor):
    def __init__(self, path_to_raw):
        super().__init__(path_to_raw)

    def create_model_and_fit(self, X_train, y_train, X_val, y_val):
        model = CatBoostRegressor(random_seed=SEED, task_type='GPU')
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            cat_features=self.preprocessor.cat_columns,
        )
        return model


class LGBMReg(BaseRegressor):
    def __init__(self, path_to_raw):
        self.preprocessor = LGBMPrep(path_to_raw)
        self.fold_preprocessors = []

    def create_model_and_fit(self, X_train, y_train, X_val, y_val, hparams=None):
        if hparams is not None:
            self.preprocessor.fit(X_train, y_train)
            self.fold_preprocessors.append(self.preprocessor)
            X_train = self.preprocessor.transform(X_train)
            X_val = self.preprocessor.transform(X_val)
            model = LGBMRegressor(**hparams)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=0
            )
        else:
            self.preprocessor.fit(X_train, y_train)
            self.fold_preprocessors.append(self.preprocessor)
            X_train = self.preprocessor.transform(X_train)
            X_val = self.preprocessor.transform(X_val)

            model = LGBMRegressor()
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100
            )

        return model, mean_squared_error(model.predict(X_val), y_val)