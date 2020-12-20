import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

SEED = 14300631
N_FOLDS = 3
TRAIN_PATH = Path('data/preprocessed/train_final.pkl')

TRAIN = pd.read_pickle(TRAIN_PATH)
TRAIN = TRAIN.drop(['id', 'publish_date'], axis=1)

random.seed(SEED)
np.random.seed(SEED)


def objective(trial):

    params = {
        'clf_lr': trial.suggest_float('clf_lr', 0.005, 0.1),
        'reg_lr': trial.suggest_float('reg_lr', 0.01, 0.2),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 10),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Poisson']),
        'depth': trial.suggest_int('depth', 1, 12),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'one_hot_max_size': trial.suggest_categorical('one_hot_max_size', [2, 10]),
        'nan_mode': trial.suggest_categorical('nan_mode', ['Min', 'Max']),
        'clf_auto_class': trial.suggest_categorical('clf_auto_class', ['None', 'SqrtBalanced']),
        'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 256]),

    }
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    elif params['bootstrap_type'] == 'Poisson':
        params['subsample'] = trial.suggest_float('subsample', 0.1, 1)
    
    if params['grow_policy'] in ['Depthwise', 'Lossguide']:
        params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 1, 8)

    print('Start trial with parameters:')
    print(params)
    trial_cv_metrics = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for idx, (train_indexes, val_indexes) in enumerate(skf.split(TRAIN, TRAIN['publish_year'])):
        X_train = TRAIN.loc[train_indexes].drop(['publish_year', 'salary'], axis=1)
        y_train = TRAIN.loc[train_indexes, 'salary']
        
        X_val = TRAIN.loc[val_indexes].drop(['publish_year', 'salary'], axis=1)
        y_val = TRAIN.loc[val_indexes, 'salary']

        cat_features = X_train.select_dtypes('category').columns.values
        # 1st model - zeros classifier
        y_clf_train = (y_train > 0).astype('int')
        y_clf_val = (y_val > 0).astype('int')

        clf_model = CatBoostClassifier(
            iterations=8000,
            random_seed=SEED,
            task_type='GPU',
            use_best_model=True,
            od_pval=10e-4,
            learning_rate=params['clf_lr'],
            l2_leaf_reg=params['l2_leaf_reg'],
            bootstrap_type=params['bootstrap_type'],
            depth=params['depth'],
            grow_policy=params['grow_policy'],
            one_hot_max_size=params['one_hot_max_size'],
            nan_mode=params['nan_mode'],
            auto_class_weights=params['clf_auto_class'],
            border_count=params['border_count'],
            bagging_temperature=params.get('bagging_temperature', None),
            subsample=params.get('subsample', None),
            min_data_in_leaf=params.get('min_data_in_leaf', None),
        )
        print(f'Fold {idx + 1} / {N_FOLDS}: zeros classifier stage')
        clf_model.fit(
            X_train,
            y_clf_train,
            eval_set=(X_val, y_clf_val),
            cat_features=cat_features,
            verbose_eval=1000,
        )
        val_zero_probes = clf_model.predict_proba(X_val)[:, 1]
        # 2nd model - regressor
        X_reg_train = X_train[y_train > 0]
        y_reg_train = y_train[y_train > 0]

        X_reg_val = X_val[y_val > 0]
        y_reg_val = y_val[y_val > 0]
        
        reg_model = CatBoostRegressor(
            iterations=8000,
            random_seed=SEED,
            task_type='GPU',
            use_best_model=True,
            od_pval=10e-5,
            learning_rate=params['reg_lr'],
            l2_leaf_reg=params['l2_leaf_reg'],
            bootstrap_type=params['bootstrap_type'],
            depth=params['depth'],
            grow_policy=params['grow_policy'],
            one_hot_max_size=params['one_hot_max_size'],
            nan_mode=params['nan_mode'],
            border_count=params['border_count'],
            bagging_temperature=params.get('bagging_temperature', None),
            subsample=params.get('subsample', None),
            min_data_in_leaf=params.get('min_data_in_leaf', None),
        )
        print(f'Fold {idx + 1} / {N_FOLDS}: regressor stage')
        reg_model.fit(
            X_reg_train,
            y_reg_train,
            eval_set=(X_reg_val, y_reg_val),
            cat_features=cat_features,
            verbose_eval=1000,
        )
        y_val_reg = reg_model.predict(X_val)
        y_val_pred = y_val_reg * val_zero_probes
        val_rmsle = np.sqrt(mean_squared_error(y_val, y_val_pred))
        trial_cv_metrics.append(val_rmsle)
        print(f'Fold {idx + 1}:: RMSLE = {val_rmsle}')

    print('RMSLE by folds:', trial_cv_metrics)
    return np.mean(trial_cv_metrics)


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print(f'Number of finished trials: {len(study.trials)}')

    print('Best trial:')
    trial = study.best_trial

    print(f'  Value: {trial.value}')

    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
