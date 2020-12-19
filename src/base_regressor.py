import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from .base_preprocessor import BasePreprocessor
SEED = 14300631
N_FOLDS = 10

random.seed(SEED)
np.random.seed(SEED)


class BaseRegressor:
    def __init__(self, path_to_raw):
        self.preprocessor = BasePreprocessor(path_to_raw)

    def preprocess(self, path_to_raw='../data/raw'):
        self.preprocessor.fit()
        self.train = self.preprocessor.transform_train()
        self.test = self.preprocessor.transform_test()

    def create_model_and_fit(self, X_train, y_train, X_val, y_val):
        model = LGBMRegressor(max_depth=5, n_estimators=500, n_jobs=16)
        model.fit(
            X_train,
            y_train,
        )
        return model

    def fit(self):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        self.cv_metrics = []

        self.models = []
        i = 0
        for train_indexes, val_indexes in skf.split(self.train, self.train['publish_year']):
            print(f"Fold {i}")
            i += 1
            X_train = self.train.iloc[train_indexes]
            y_train = X_train['salary']
            X_train = X_train.drop(['id', 'salary'], axis=1)

            X_val = self.train.iloc[val_indexes]
            y_val = X_val['salary']
            X_val = X_val.drop(['id', 'salary'], axis=1)

            model = self.create_model_and_fit(X_train, y_train, X_val, y_val)

            self.models.append(model)
            val_metric = mean_squared_error(model.predict(X_val), y_val)
            print(f"Val rmsle: {val_metric}")
            self.cv_metrics.append(val_metric)

    def val_info(self):
        print(self.cv_metrics)
        print("Mean: ", np.mean(self.cv_metrics))
        print("Std: ", np.std(self.cv_metrics))

    def create_test_submission(self, file):
        sub = self.test[["id"]]
        X_test = self.test.drop(['id'], axis=1)
        test_preds = []
        for model in self.models:
            test_pred = model.predict(X_test)
            test_preds.append(np.exp(test_pred))
        prediction = np.mean(test_preds, axis=0)
        sub["salary"] = prediction
        sub.to_csv(file, index=False)
