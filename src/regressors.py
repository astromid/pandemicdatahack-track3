from .base_preprocessor import BasePreprocessor
from .base_regressor import BaseRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
import random
import numpy as np
import pandas as pd
SEED = 14300631
N_FOLDS = 10

random.seed(SEED)
np.random.seed(SEED)

class LinearReg(BaseRegressor):
    def __init__(self, path_to_raw):
        super().__init__(path_to_raw)

    def create_model_and_fit(self, X_train, y_train, X_val, y_val):
        model = Lasso()
        model.fit(
            X_train,
            y_train,
        )
        return model

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