import os

import copy

import pandas as pd
import numpy as np
import logging

import matplotlib.pyplot as plt
import seaborn as sns


from scipy.stats import shapiro

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold
from sklearn.inspection import permutation_importance

from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error

import optuna

from typing import List, Dict, Set


from concrete_strength.a_constants import *
from concrete_strength.f_utils.common import read_yaml
from concrete_strength.f_utils.common import save_pickle, load_pickle

gparams = read_yaml(GLOBALPARAMS_FILE_PATH).globalparams
RANDOM_STATE = gparams.RANDOM_STATE
N_SPLITS = gparams.N_SPLITS
N_TRIALS = gparams.N_TRIALS
# Set the logging level to ERROR
optuna.logging.set_verbosity(optuna.logging.ERROR)


def rmse(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return rmse


def eval_metrics(y_true, y_pred):
    r2_sqr = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r_mse = np.sqrt(mse)
    return r2_sqr, mae, mse, r_mse


def kfold_cv(X, y, model, metric=rmse, nsplits=N_SPLITS):
    kf = KFold(n_splits=nsplits, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    trained_models = []

    for idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # different instance will be created for to dump in pickle/joblib
        # Else same instance will appended to list, which gives same result when using pickle/joblib
        model = copy.deepcopy(model)

        model.fit(X_train, y_train)
        trained_models.append(model)

        y_pred = model.predict(X_val)
        metric_val = metric(y_val, y_pred)
        metrics.append(metric_val)

    return np.mean(metrics), trained_models


def optimize_hyperparameters(X, y, estimator, hyperparameters, metric_name=rmse, nsplits=N_SPLITS, ntrials=N_TRIALS):
    def objective(trial):

        params = {}
        for param_name, param_config in hyperparameters.items():
            param_type = param_config['type']

            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high'], step=param_config.get('step', 1))

            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'], log=True)

            elif param_type == 'categorical':
                choices = param_config['choices']
                params[param_name] = trial.suggest_categorical(
                    param_name, choices)

        model = estimator(**params, random_state=RANDOM_STATE)

        # Use K-Fold cross-validation with 5 splits and minimize RMSE
        loss_val = kfold_cv(
            X, y,  model, metric=metric_name, nsplits=nsplits)[0]
        return loss_val

    # We want to minimize RMSE
    study = optuna.create_study(
        direction='minimize', study_name='model_tuning')
    # Adjust the number of trials as needed
    study.optimize(objective, n_trials=ntrials)

    # Get the best hyperparameters
    best_params = study.best_params

    return best_params


def train_kfold_cv(X, y, model, nsplits=N_SPLITS, eval_=None, xtest=None, ytest=None):
    kf = KFold(n_splits=nsplits, shuffle=True, random_state=RANDOM_STATE)

    trained_models = []
    model_score = {}

    model_score['valid'] = {'r2_square': [], 'mae': [], 'mse': [], 'r_mse': []}
    model_score['test'] = {'r2_square': [], 'mae': [], 'mse': [], 'r_mse': []}

    for idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # different instance will be created for to dump in pickle/joblib
        # Else same instance will appended to list, which gives same result when using pickle/joblib
        model = copy.deepcopy(model)

        model.fit(X_train, y_train)
        trained_models.append(model)

        y_pred = model.predict(X_val)

        if eval_ is not None:
            val_metrics = eval_metrics(y_val, y_pred)
            model_score['valid']['r2_square'].append(val_metrics[0])
            model_score['valid']['mae'].append(val_metrics[1])
            model_score['valid']['mse'].append(val_metrics[2])
            model_score['valid']['r_mse'].append(val_metrics[3])
            if xtest is not None:
                test_metrics = eval_metrics(ytest, model.predict(xtest))
                model_score['test']['r2_square'].append(test_metrics[0])
                model_score['test']['mae'].append(test_metrics[1])
                model_score['test']['mse'].append(test_metrics[2])
                model_score['test']['r_mse'].append(test_metrics[3])

    return trained_models, model_score
