from .base_preprocessor import BasePreprocessor

import random
import numpy as np
import pandas as pd
SEED = 14300631
N_FOLDS = 10

random.seed(SEED)
np.random.seed(SEED)

class CatPrep(BasePreprocessor):
    def __init__(self, path):
        super().__init__(path)

    def transform_train(self):
        full_train = self.preprocess_train_test(self.raw_train)
        full_train = pd.merge(full_train, self.education, how='left', on='id')
        full_train = pd.merge(full_train, self.employements, how='left', on='id')
        full_train = pd.merge(full_train, self.worldskills, how='left', on='id')
        full_train = full_train.drop(self.drop_columns, axis=1, errors='ignore')
        full_train[self.cat_columns] = full_train[self.cat_columns].fillna("nan")
        full_train = full_train.dropna()
        return full_train

    def transform_test(self):
        full_test = self.preprocess_train_test(self.raw_test)
        full_test = pd.merge(full_test, self.education, how='left', on='id')
        full_test = pd.merge(full_test, self.employements, how='left', on='id')
        full_test = pd.merge(full_test, self.worldskills, how='left', on='id')
        full_test = full_test.drop(self.drop_columns, axis=1, errors='ignore')
        full_test = full_test.drop(self.cat_columns, axis=1, errors='ignore')
        full_test[self.cat_columns] = full_test[self.cat_columns].fillna("nan")
        return full_test