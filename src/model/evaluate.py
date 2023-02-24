import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss, f1_score
from bayes_opt import BayesianOptimization


#TRAINING_COLS = ['WinRatioA', 'PtsForRatioA', 'WinRatioB', 'PtsForRatioB', 'RatingA', 'RatingB', 'RatingDiff']
#TRAINING_COLS = ['RatingDiff', 'WinRatioA', 'WinRatioB', 'SeedDiff', 'SeedA', 'SeedB']
TRAINING_COLS = ['SeedDiff', 'EloWinProbA', 'Gender']
TARGET_COL = ['Win']

DEFAULT_PARAMS = {'gamma' : 0.499538, 'learning_rate' : 0.104466, 'max_depth' : 2, 'n_estimators' : 113}

def load_training_data():
    """Loads in training data with pre-calculated features from ETL."""
    return pd.read_csv('./data/etl/training_set.csv')


def build_model(training_data, params={}):
    """Fits given data to an XGBoost model."""
    model = XGBClassifier(**params)

    x = training_data[TRAINING_COLS]
    y = training_data[TARGET_COL]
    model.fit(x, y)

    return model


def cross_validate_model(training_data, start_year=2017, end_year=2022, verbose=True, params={}):
    """Cross validate model by building a new model per year."""

    # Store metrics for evaluation
    metrics = {'acc' : [],
               'brier' : [],
               'logloss' : [],
               'f1' : []}

    for year in range(start_year, end_year+1):
        # Split into training and validation set (for all data before year, and data of year)
        df_train = training_data[training_data.Season < year]
        df_valid = training_data[training_data.Season == year]
        df_valid_res = df_valid[TARGET_COL]

        # If validation set, then get predictions and metrics
        if df_valid.shape[0] > 0:
            # Build model with training set
            model = build_model(df_train, params=params)

            # Get predictions for year
            preds_proba = model.predict_proba(df_valid[TRAINING_COLS])[:,1]
            preds_cat = [1 if x >= 0.5 else 0 for x in preds_proba]

            # Get metrics
            metrics['acc'].append(accuracy_score(df_valid_res, preds_cat))
            metrics['brier'].append(brier_score_loss(df_valid_res, preds_proba))
            metrics['logloss'].append(log_loss(df_valid_res, preds_proba))
            metrics['f1'].append(f1_score(df_valid_res, preds_cat))

            if verbose:
                print(year)
                print('Accuracy   ', metrics['acc'][-1])
                print('Brier Loss ', metrics['brier'][-1])
                print('Log Loss   ', metrics['logloss'][-1])
                print('F1 Score   ', metrics['f1'][-1])
                print()
    
    # Calculate final metrics
    avg_acc = sum(metrics['acc']) / len(metrics['acc'])
    avg_brier = sum(metrics['brier']) / len(metrics['brier'])
    avg_logloss = sum(metrics['logloss']) / len(metrics['logloss'])
    avg_f1 = sum(metrics['f1']) / len(metrics['f1'])
    
    # Output final metrics
    if verbose:
        print('Final Metrics')
        print('Accuracy   ', avg_acc)
        print('Brier Loss ', avg_brier)
        print('Log Loss   ', avg_logloss)
        print('F1 Score   ', avg_f1)
        print()
    
    return avg_acc, avg_brier, avg_logloss, avg_f1


def tune_model_bayesian_optimisation(training_data, iterations=5, initial_points=8):
    """Perform Bayesian Optimisation to find best hyperparameters for model."""
    def bo_tune_function(max_depth, gamma, n_estimators, learning_rate):
        params = {'max_depth' : int(max_depth),
                  'gamma' : gamma,
                  'n_estimators' : int(n_estimators),
                  'learning_rate' : learning_rate}
        acc, brier, loglogg, f1 = cross_validate_model(training_data, verbose=False, params=params)
        return -1*brier

    # Search field for bayesian optimisation of hyperparameters
    search_params = {'max_depth' : (2,7),
                     'gamma' : (0,1),
                     'n_estimators' : (50,200),
                     'learning_rate' : (0,1)}

    # Find best hyperparameters
    xgb_bo = BayesianOptimization(bo_tune_function, search_params, random_state=0)
    xgb_bo.maximize(n_iter=iterations, init_points=initial_points)
    best_params = xgb_bo.max['params']

    # Fix type issues
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])

    return best_params


def validate_and_build_model(model_name='default_model', tune=True):
    """Gets metrics with cross validation, then saves complete model."""

    # Load training data
    training_data = load_training_data()

    # Get separate mens and womens data
    training_data = {'men' : training_data.loc[training_data['Gender'] == 0],
                     'women' : training_data.loc[training_data['Gender'] == 1]}

    for gen, data in training_data.items():
        print('Now Building Model For', gen.title())
        # Find best hyperparameters by tuning
        if tune:
            print()
            params = tune_model_bayesian_optimisation(data, iterations=50, initial_points=10)
            print()
            print('Best Hyperparameters')
            print(params)
            print()
        else:
            params = DEFAULT_PARAMS

        # CV and output metrics
        cross_validate_model(data, params=params)

        # Build final model and save
        model = build_model(data, params=params)
        model.save_model(f'./data/models/{model_name}_{gen}.mdl')


if __name__ == "__main__":
    validate_and_build_model()