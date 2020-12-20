import numpy as np
import pandas as pd


def preprocess_train_test(df):
    # number of NaNs
    df['number_nans'] = df.isnull().sum(axis=1)

    # position
    positions_splitted = df['position'].str.lower().str.replace('.', ' ').str.split(',').tolist()
    df['positions_first'] = [row[0].strip() for row in positions_splitted]
    df['positions_first'] = df['positions_first'].fillna('NA_category').astype('category')

    df['positions_second'] = [row[1].strip() if len(row) > 1 else 'NA_category' for row in positions_splitted]
    df['positions_second'] = df['positions_second'].fillna('NA_category').astype('category')

    df['positions_other'] = [' '.join([x.strip() for x in row[2:]]) if len(row) > 2 else 'NA_category' for row in positions_splitted]
    df['positions_other'] = df['positions_other'].fillna('NA_category').astype('category')
    df = df.drop('position', axis=1)
    print('Position preprocessed')

    # region
    df['region'] = df['region'].str.lower().fillna('NA_category').astype('category')
    print('Region preprocessed')

    # industry
    df['industry'] = df['industry'].str.lower().fillna('NA_category').astype('category')
    print('Industry preprocessed')

    # locality
    df = df.drop('locality', axis=1)

    # locality name
    df['locality_name'] = df['locality_name'].str.lower().fillna('NA_category').astype('category')
    print('Locality name preprocessed')

    # education type
    df['education_type'] = df['education_type'].str.lower().fillna('NA_category').astype('category')
    print('Education preprocessed')

    # drive licenses
    all_licenses = ['A', 'B', 'C', 'D', 'E']
    df['has_drive'] = ~df['drive_licences'].isna()
    for license in all_licenses:
        df[f'has_license_{license}'] = df['drive_licences'].str.contains(license).fillna(False)
    df = df.drop('drive_licences', axis=1)
    print('Drive licenses preprocessed')

    # citizenship
    df['citizenship'] = df['citizenship'].str.lower().fillna('NA_category').astype('category')
    print('Citizenship preprocessed')

    # schedule
    all_schedules = [
        ('watch', 'вахтовый метод'),
        ('flexible', 'гибкий график'),
        ('irregular', 'ненормированный рабочий день'),
        ('parttime', 'неполный рабочий день'),
        ('fulltime', 'полный рабочий день'),
        ('shift', 'сменный график'),
    ]
    df['schedule'] = df['schedule'].str.lower()
    for schedule_name, schedule_type in all_schedules:
        df[f'{schedule_name}_yes'] = df['schedule'].str.contains(schedule_type)
    df = df.drop('schedule', axis=1)
    print('Schedule preprocessed')

    # employement type
    df['employement_type'] = df['employement_type'].str.lower().fillna('NA_category').astype('category')
    print('Employement preprocessed')

    # age
    df['age'] = df['age'].apply(lambda x: np.nan if (x < 14) or (x > 82) else x)
    # fill na with median + IQR
    na_age = df['age'].median() + df['age'].quantile(0.75) - df['age'].quantile(0.25)
    df['age'] = df['age'].fillna(na_age).astype('int')
    print('Age preprocessed')

    # gender
    df['gender'] = df['gender'].str.lower().fillna('NA_category').astype('category')
    print('Gender preprocessed')

    # experience
    experience_q95 = df['experience'].quantile(0.95)
    df.loc[df['experience'] > experience_q95, 'experience'] = np.nan
    df['experience'] = df['experience'].fillna(0).astype('int')
    print('Experience preprocessed')

    # salary desired
    df.loc[df['salary_desired'] < 300, 'salary_desired'] = np.nan
    df['salary_desired'] = df['salary_desired'].fillna(0)
    df['salary_desired'] = np.log(df['salary_desired'] + 1)
    print('Salary desired preprocessed')

    # relocation ready
    df['relocation_ready'] = df['relocation_ready'].fillna(2).astype('int').astype('category')
    print('Relocation ready preprocessed')

    # travel ready
    df['travel_ready'] = df['travel_ready'].fillna(2).astype('int').astype('category')
    print('Travel ready preprocessed')

    # retraining ready
    df['retraining_ready'] = df['retraining_ready'].fillna(2).astype('int').astype('category')
    print('Retraining ready preprocessed')

    # is_worldskills_participant
    df['is_worldskills_participant'] = df['is_worldskills_participant'].fillna(False).astype('bool')
    print('Is worldskills participant preprocessed')

    # has_qualifications
    df['has_qualifications'] = df['has_qualifications'].fillna(False).astype('bool')
    print('Has qualifications preprocessed')

    # completeness_rate
    df['completeness_rate'] = df['completeness_rate'].fillna(0).astype('int')

    # creation date
    df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
    print('Creation date preprocessed')

    # modification_date
    df['modification_date'] = pd.to_datetime(df['modification_date'], errors='coerce')
    print('Modification date preprocessed')

    # publish date
    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
    print('Publish date preprocessed')

    # days between
    df['days_between_c_m'] = (df['modification_date'] - df['creation_date']).dt.days
    df['days_between_c_m'] = df['days_between_c_m'].fillna(0).astype('int')
    df['days_between_m_p'] = (df['publish_date'] - df['modification_date']).dt.days
    df['days_between_m_p'] = df['days_between_m_p'].fillna(0).astype('int')
    df['days_between_c_p'] = (df['publish_date'] - df['creation_date']).dt.days
    df['days_between_c_p'] = df['days_between_c_p'].fillna(0).astype('int')
    df = df.drop(['creation_date', 'modification_date'], axis=1)
    print('Days between preprocessed')

    df.loc[df['salary'] < 300, 'salary'] = np.nan
    df['salary'] = df['salary'].fillna(0)
    df['salary'] = np.log(df['salary'] + 1)

    return df
