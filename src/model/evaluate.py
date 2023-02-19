import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss, f1_score


TRAINING_COLS = ['WinRatioA', 'PtsForRatioA', 'WinRatioB', 'PtsForRatioB']
TARGET_COL = ['Win']


def load_training_data():
    """Loads in training data with pre-calculated features from ETL."""
    return pd.read_csv('./data/etl/training_set.csv')


def build_model(training_data, params={}):
    """Fits given data to an XGBoost model."""
    model = XGBClassifier()

    x = training_data[TRAINING_COLS]
    y = training_data[TARGET_COL]
    if len(params) > 0:
        model.fit(x, y, params)
    else:
        model.fit(x, y)

    return model


def cross_validate_model(training_data, start_year=2017, end_year=2022, verbose=True):
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
            model = build_model(df_train)

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
    
    avg_acc = sum(metrics['acc']) / len(metrics['acc'])
    avg_brier = sum(metrics['brier']) / len(metrics['brier'])
    avg_logloss = sum(metrics['logloss']) / len(metrics['logloss'])
    avg_f1 = sum(metrics['f1']) / len(metrics['f1'])

    print('Final Metrics')
    print('Accuracy   ', avg_acc)
    print('Brier Loss ', avg_brier)
    print('Log Loss   ', avg_logloss)
    print('F1 Score   ', avg_f1)
    print()


def validate_and_build_model(model_name='default_model.mdl'):
    """Gets metrics with cross validation, then saves complete model."""

    # Load training data
    training_data = load_training_data()

    # CV and output metrics
    cross_validate_model(training_data)

    # Build final model and save
    model = build_model(training_data)
    model.save_model(f'./data/models/{model_name}')


if __name__ == "__main__":
    validate_and_build_model()