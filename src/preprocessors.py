from .base_preprocessor import BasePreprocessor
from category_encoders.cat_boost import CatBoostEncoder
import category_encoders as ce
import random
import numpy as np
import pandas as pd
SEED = 1377
N_FOLDS = 10

random.seed(SEED)
np.random.seed(SEED)

class CatPrep(BasePreprocessor):
    def __init__(self, path):
        super().__init__(path)

    def transform_train(self):
        full_train = self.preprocess_train_test(self.raw_train)
        # full_train = pd.merge(full_train, self.education, how='left', on='id')
        # full_train = pd.merge(full_train, self.employements, how='left', on='id')
        # full_train = pd.merge(full_train, self.worldskills, how='left', on='id')
        # new_drop_columns = ['status', 'code', 'is_international', 'int_name', 'ru_name']
        # full_train = full_train.drop(new_drop_columns, axis=1, errors='ignore')
        # full_train[self.cat_columns] = full_train[self.cat_columns].fillna("nan")
        # full_train = full_train.dropna()
        return full_train

    def transform_test(self):
        full_test = self.preprocess_train_test(self.raw_test)
        # full_test = pd.merge(full_test, self.education, how='left', on='id')
        # full_test = pd.merge(full_test, self.employements, how='left', on='id')
        # full_test = pd.merge(full_test, self.worldskills, how='left', on='id')
        # full_test = full_test.drop(self.drop_columns, axis=1, errors='ignore')
        # new_drop_columns = ['status', 'code', 'is_international', 'int_name', 'ru_name']
        # full_test = full_test.drop(new_drop_columns, axis=1, errors='ignore')
        # full_test[self.cat_columns] = full_test[self.cat_columns].fillna("nan")
        return full_test


class LinearPrep(BasePreprocessor):
    def __init__(self, path):
        super().__init__(path)


    def transform_train(self):
        full_train = self.preprocess_train_test(self.raw_train)
        # full_train = pd.merge(full_train, self.education, how='left', on='id')
        # full_train = pd.merge(full_train, self.employements, how='left', on='id')
        # full_train = pd.merge(full_train, self.worldskills, how='left', on='id')
        # new_drop_columns = ['status', 'code', 'is_international', 'int_name', 'ru_name']
        # full_train = full_train.drop(self.drop_columns, axis=1, errors='ignore')
        # full_train = full_train.drop(new_drop_columns, axis=1, errors='ignore')
        # full_train[self.cat_columns] = full_train[self.cat_columns].fillna("nan")
        # full_train = full_train.dropna()
        return full_train

    def transform_test(self):
        full_test = self.preprocess_train_test(self.raw_test)
        # full_test = pd.merge(full_test, self.education, how='left', on='id')
        # full_test = pd.merge(full_test, self.employements, how='left', on='id')
        # full_test = pd.merge(full_test, self.worldskills, how='left', on='id')
        # full_test = full_test.drop(self.drop_columns, axis=1, errors='ignore')
        # new_drop_columns = ['status', 'code', 'is_international', 'int_name', 'ru_name']
        # full_test = full_test.drop(new_drop_columns, axis=1, errors='ignore')
        # full_test[self.cat_columns] = full_test[self.cat_columns].fillna("nan")
        self.new_cat_columns = full_test.select_dtypes(include=['category', 'boolean']).columns

        self.cat_encoder = CatBoostEncoder(
            cols=self.new_cat_columns,
        )
        return full_test

    def fit(self, X, y):
        self.cat_encoder.fit(X, y)

    def transform(self, X):
        return self.cat_encoder.transform(X)

class LGBMPrep(BasePreprocessor):
    def __init__(self, path):
        super().__init__(path)


    def transform_train(self):
        # full_train = self.preprocess_train_test(self.raw_train)
        # full_train = pd.merge(full_train, self.education, how='left', on='id')
        # full_train = pd.merge(full_train, self.employements, how='left', on='id')
        # full_train = pd.merge(full_train, self.worldskills, how='left', on='id')
        # new_drop_columns = ['status', 'code', 'is_international', 'int_name', 'ru_name']
        # full_train = full_train.drop(self.drop_columns, axis=1, errors='ignore')
        # full_train = full_train.drop(new_drop_columns, axis=1, errors='ignore')
        # full_train[self.cat_columns] = full_train[self.cat_columns].fillna("nan")
        # full_train = full_train.dropna()
        full_train = self.raw_train
        # cols = [item for item in (self.train_cat + self.train_binary)
        #                                           if item in full_train.columns]
        self.cat_features = full_train.select_dtypes(include=['category', 'boolean']).columns.values
        # self.cat_encoder = ce.BinaryEncoder(cols=self.cat_features)
        # full_train = pd.merge(full_train, self.prep_empls, how='left', on='id')
        return full_train

    def transform_test(self):
        # full_test = self.preprocess_train_test(self.raw_test)
        # full_test = pd.merge(full_test, self.education, how='left', on='id')
        # full_test = pd.merge(full_test, self.employements, how='left', on='id')
        # full_test = pd.merge(full_test, self.worldskills, how='left', on='id')
        # full_test = full_test.drop(self.drop_columns, axis=1, errors='ignore')
        # new_drop_columns = ['status', 'code', 'is_international', 'int_name', 'ru_name']
        # full_test = full_test.drop(new_drop_columns, axis=1, errors='ignore')
        # full_test[self.cat_columns] = full_test[self.cat_columns].fillna("nan")
        # self.new_cat_columns = full_test.select_dtypes(include=['category', 'boolean']).columns
        # full_test = pd.merge(full_test, self.prep_empls, how='left', on='id')
        return self.raw_test

    def fit(self, X_train, y_train):
        # categorical
        # self.cat_encoder.fit(X_train, y_train)
        #
        self.cols_mean = {}
        for col in [item for item in (self.train_idk+self.train_dates) if item in X_train.columns
                                                                          and item not in self.cat_features]:
            mean_value = X_train[col].mean()
            self.cols_mean[col] = mean_value

    def transform(self, X_test):
        # X_test = self.cat_encoder.transform(X_test)
        X_test = X_test.drop(self.train_text+self.train_idk, axis=1, errors='ignore')
        for k, v in self.cols_mean.items():
            X_test[k] = X_test[k].fillna(v)
        return X_test




