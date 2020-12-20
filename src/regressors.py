from skimage.metrics import mean_squared_error

from .base_preprocessor import BasePreprocessor
from .preprocessors import *
from .base_regressor import BaseRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import ElasticNet
import random
import numpy as np
import pandas as pd
SEED = 1377
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
            params = {'lambda_l1': 1.4691057735214796e-06, 'lambda_l2': 4.461231059744859e-05, 'num_leaves': 41,
                        'feature_fraction': 0.90954941668357, 'bagging_fraction': 0.9621487566952892,
                        'bagging_freq': 7, 'min_child_samples': 52}
            self.preprocessor.fit(X_train, y_train)
            self.fold_preprocessors.append(self.preprocessor)
            X_train = self.preprocessor.transform(X_train)
            X_val = self.preprocessor.transform(X_val)
            # 1st model - zeros classifier
            y_clf_train = (y_train > 0).astype('int')
            y_clf_val = (y_val > 0).astype('int')

            clf_model = LGBMClassifier(**params)
            print('--------- Train zeros classifier ---------')
            clf_model.fit(
                X_train,
                y_clf_train,
                eval_set=[(X_val, y_clf_val)],
                early_stopping_rounds=100,
                verbose=1,
            )
            val_zero_probes = clf_model.predict_proba(X_val)[:, 1]

            model = LGBMRegressor(n_estimators=1000, **params)
            print('--------- Train regressor ---------')
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=1,
            )
            y_val_reg = model.predict(X_val)
            y_val_pred = y_val_reg * val_zero_probes
            val_rmsle = np.sqrt(mean_squared_error(y_val, y_val_pred))

        return clf_model, model, val_rmsle

    def create_test_submission(self, file):
        sub = self.test[["id"]]
        X_test = self.test.drop(['id'], axis=1)
        test_preds = []
        for idx, model in enumerate(self.models):
            X_test_ = self.fold_preprocessors[idx].transform(X_test)
            test_zero_probes = self.clf_models[idx].predict_proba(X_test_)[:, 1]
            y_test_reg = model.predict(X_test_)
            test_preds.append(np.exp(y_test_reg * test_zero_probes) - 1)
        prediction = np.mean(test_preds, axis=0)
        sub["salary"] = prediction
        sub.to_csv(file, index=False)
