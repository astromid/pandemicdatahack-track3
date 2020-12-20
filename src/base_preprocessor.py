import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import category_encoders as ce
from pathlib import Path
SEED = 1377
N_FOLDS = 10

random.seed(SEED)
np.random.seed(SEED)
import pickle

class BasePreprocessor:
    def __init__(self, path_to_raw):
        self.raw_data_dir = Path(path_to_raw)
        self.raw_train = pd.read_pickle(self.raw_data_dir / "train_covid_prices.pkl")
        self.raw_train["publish_year"] = self.raw_train["publish_date"].dt.year
        self.raw_train = self.raw_train.drop("publish_date", axis=1)
        self.raw_test = pd.read_pickle(self.raw_data_dir / "test_covid_prices.pkl")
        self.raw_test["publish_year"] = self.raw_test["publish_date"].dt.year
        self.raw_test = self.raw_test.drop("publish_date", axis=1)
        # embeds = pickle.load(open(self.raw_data_dir.parent / "employements_mult_new_ft_1.pkl", "rb"))
        # self.train = pd.merge(train, self.education, how='left', on='id')

        # self.raw_train = pd.read_csv(self.raw_data_dir / 'train.csv', sep=';',
        #                         parse_dates=['creation_date', 'modification_date', 'publish_date'])
        # self.raw_test = pd.read_csv(self.raw_data_dir / 'test.csv', sep=';',
        #                        parse_dates=['creation_date', 'modification_date', 'publish_date'])
        # self.raw_education = pd.read_csv(self.raw_data_dir / 'education.csv', sep=';')
        # self.raw_employements = pd.read_csv(self.raw_data_dir / 'employements.csv', sep=';')
        # self.raw_worldskills = pd.read_csv(self.raw_data_dir / 'worldskills.csv', sep=';')
        #
        # self.education = self.preprocess_education(self.raw_education)
        # self.employements = self.preprocess_employements(self.raw_employements)
        # self.worldskills = self.preprocess_worldskills(self.raw_worldskills)
        #
        # self.prep_empls = pickle.load(open("../data/employements_mult_new_ft_1.pkl", "rb"))
        # self.prep_empls = self.prep_empls.drop(["position", "employer", "achievements", "responsibilities",
        #                       "start_date", "finish_date", "position_clean", "employer_clean"], axis=1)

        self.train_cat = [
            "region",
            "industry",
            "education_type",
            "drive_licences",  #
            "citizenship",
            "schedule",
            "employement_type",
            "gender"
        ]

        self.train_binary = [
            "relocation_ready",
            "travel_ready",
            "retraining_ready",
            "is_worldskills_participant",
            "has_qualifications"
        ]

        self.train_text = [
            "position",
        ]

        self.train_float = [
            "age",
            "experience",
            "salary_desired",  #
            "completeness_rate"
        ]

        self.train_idk = [
            "locality",
            "locality_name"
        ]

        self.train_dates = [
            "creation_date",
            "modification_date",
            "publish_date"
        ]

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass

    def transform_train(self):
        full_train = self.preprocess_train_test(self.raw_train)
        # full_train = pd.merge(train, self.education, how='left', on='id')
        # full_train = pd.merge(full_train, self.employements, how='left', on='id')
        # full_train = pd.merge(full_train, self.worldskills, how='left', on='id')
        # full_train = full_train.drop(self.drop_columns, axis=1, errors='ignore')
        # full_train = full_train.drop(self.cat_columns, axis=1, errors='ignore')
        # full_train[self.cat_columns] = full_train[self.cat_columns].fillna("nan")
        # full_train = full_train.dropna()
        full_train = pd.merge(full_train, self.prep_empls, how='left', on='id')
        cols = ["is_worldskills_participant", "has_qualifications"]
        # full_train = full_train.drop(cols, axis=1, errors='ignore')
        return full_train

    def transform_test(self):
        full_test = self.preprocess_train_test(self.raw_test)
        # full_test = pd.merge(test, self.education, how='left', on='id')
        # full_test = pd.merge(full_test, self.employements, how='left', on='id')
        # full_test = pd.merge(full_test, self.worldskills, how='left', on='id')
        # full_test = full_test.drop(self.drop_columns, axis=1, errors='ignore')
        # full_test = full_test.drop(self.cat_columns, axis=1, errors='ignore')
        # full_test[self.cat_columns] = full_test[self.cat_columns].fillna("nan")
        # full_test = full_test.dropna()
        cols = ["is_worldskills_participant", "has_qualifications"]
        # full_test = full_test.drop(cols, axis=1, errors='ignore')
        full_test = pd.merge(full_test, self.prep_empls, how='left', on='id')
        return full_test

    def filter_experience(self, x):
        if np.isnan(x) or x > 50:
            return np.nan
        return x

    def filter_age(self, x):
        if np.isnan(x) or x < 14 or x > 83:
            return np.nan
        return x

    def preprocess_train_test(self, df):
        df['publish_year'] = df['publish_date'].dt.year

        all_drive_licences = ['A', 'B', 'C', 'D', 'E']
        for licence_type in all_drive_licences:
            df[f'drive_licences_{licence_type}'] = df['drive_licences'].fillna('').apply(lambda x: int(licence_type in x))

        all_schedules = [
            ('vahta', 'Вахтовый метод'),
            ('gibkiy', 'Гибкий график'),
            ('nenorm', 'Ненормированный рабочий день'),
            ('nepoln', 'Неполный рабочий день'),
            ('poln', 'Полный рабочий день'),
            ('smena', 'Сменный график'),
        ]

        for schedule_label, schedule_type in all_schedules:
            df[f'schedule_{schedule_label}'] = df['schedule'].apply(lambda x: int(schedule_type in x))

        df['experience'] = df['experience'].apply(self.filter_experience)
        df['age'] = df['age'].apply(self.filter_age)
        # date features
        date_cols = ["creation_date", "modification_date", "publish_date"]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d")
        for col in date_cols:
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_year"] = df[col].dt.year

        for i in range(len(date_cols)):
            for j in range(i, len(date_cols)):
                col1 = date_cols[i]
                col2 = date_cols[j]
                df[f"{col1}_minus_{col2}"] = (df[col1] - df[col2]).dt.days.abs()

        df = df.drop("publish_date_year", axis=1)
        df = df.drop([
            'locality', 'position', 'locality_name', 'drive_licences',
            'schedule',
            # 'is_worldskills_participant', 'has_qualifications',
            'creation_date', 'modification_date', 'publish_date',
        ], axis=1)

        if 'salary' in df.columns:
            # df = df[df['salary'] > 0]
            df['salary'] = np.log(df['salary'] + 1)
        df['salary_desired'] = np.log(df['salary_desired'] + 1)

        # df['region'] = df['region'].astype('category')
        # df['education_type'] = df['education_type'].astype('category')
        # df['industry'] = df['industry'].astype('category')
        # df['citizenship'] = df['citizenship'].astype('category')
        # df['employement_type'] = df['employement_type'].astype('category')
        # df['gender'] = df['gender'].astype('category')
        # df['relocation_ready'] = df['relocation_ready'].astype('boolean')
        # df['travel_ready'] = df['travel_ready'].astype('boolean')
        # df['retraining_ready'] = df['retraining_ready'].astype('boolean')
        return df

    def preprocess_education(self, df):
        df['graduation_year'] = df['graduation_year'].astype('category')
        df['institution'] = df['institution'].str.lower().str.replace('\"', '').astype('category')
        df = df.drop('description', axis=1)
        return df

    def preprocess_employements(self, df):
        df['employer'] = df['employer'].str.lower().str.replace('\"', '').astype('category')
        df['position'] = df['position'].str.lower().str.replace('\"', '').astype('category')
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['finish_date'] = pd.to_datetime(df['finish_date'], errors='coerce')
        df['work_duration'] = df['finish_date'] - df['start_date']
        df['work_duration'] = df['work_duration'].dt.days
        df = df.drop(['achievements', 'responsibilities', 'start_date', 'finish_date'], axis=1)
        return df

    def preprocess_worldskills(self, df):
        df['status'] = df['status'].astype('category')
        df['int_name'] = df['int_name'].astype('category')
        df['ru_name'] = df['ru_name'].astype('category')
        df['code'] = df['code'].astype('category')
        df['is_international'] = df['is_international'].astype('boolean')
        return df
