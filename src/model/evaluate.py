import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from joblib import dump


TRAINING_COLS = ['SeedDiff', 'EloWinProbA', 'WinRatioA', 'WinRatioB', 'ClutchRatioA', 'ClutchRatioB']
TARGET_COL = ['Win']
DEFAULT_PARAMS = {'gamma' : 0.499538, 'learning_rate' : 0.104466, 'max_depth' : 2, 'n_estimators' : 113}


def load_training_data():
    """
    Load in training data with pre-calculated features from ETL.
    
    Returns:
        Pandas DataFrame with training set.
    """
    return pd.read_csv('./data/etl/training_set.csv')


def build_model(training_data, training_columns=TRAINING_COLS, params={},
                calibrate=False, calibrator_size=0.2, random_state=0):
    """
    Fit given data to an XGBoost model.

    Args:
        training_data: Data used to train model. Must contain training columns.
        training_columns: Columns in training_data used to train model.
        params: Dictionary of XGBClassifier hyperparameters.
        calibrate: If True, fits a probability calibrator after training model.
        calibrator_size: Proportion of training set to use for calibration.
        random_state: Random value for model and calibration.
    
    Returns:
        Fitted XGBClassifier model, or CalibratedClassifierCV fitted with
        XGBClassifier if calibrate is True.
    """

    # If calibrating, split training set
    if calibrate:
        training_data, calib_data = train_test_split(training_data,
                                                     test_size=calibrator_size,
                                                     random_state=random_state)

    # Train standard model
    model = XGBClassifier(**params)
    x = training_data[training_columns]
    y = training_data[TARGET_COL]
    model.fit(x, y)

    # Fit calibrator
    if calibrate:
        model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        cx = calib_data[training_columns]
        cy = calib_data[TARGET_COL]
        model.fit(cx, cy)

    return model


def cross_validate_model(training_data, training_columns=TRAINING_COLS,
                         start_year=2017, end_year=2023, verbose=True, params={},
                         calibrate=False, random_state=0, calibrator_size=0.2):
    """
    Cross validate model by building a new model per year.

    Args:
        training_data: Data used to train model. Must contain training columns.
        training_columns: Columns in training_data used to train model.
        start_year: First year for validation.
        end_year: Final year for validation.
        verbose: If True, output validation metrics.
        params: Dictionary of XGBClassifier hyperparameters.
        calibrate: If True, fits a probability calibrator after training model.
        random_state: Random value for model and calibration.
        calibrator_size: Proportion of training set to use for calibration.
    
    Returns:
        Tuple containing validation accuracy, Brier score, log loss,
        F1 score, and calibrated Brier score. If calibrate is False then
        calibrated Brier score will be 0.
    """

    # Store metrics for evaluation
    metrics = {'acc' : [],
               'brier' : [],
               'logloss' : [],
               'f1' : [],
               'calib-brier' : []}

    for year in range(start_year, end_year+1):
        # Split into training and validation set (data before year, data of year)
        df_train = training_data[training_data.Season < year]
        df_valid = training_data[training_data.Season == year]
        df_valid_res = df_valid[TARGET_COL]

        # If validation set, then get predictions and metrics
        if df_valid.shape[0] > 0:
            # Build uncalibrated model with training set
            model = build_model(df_train, training_columns=training_columns,
                                params=params, calibrate=False)

            # Get predictions for year
            preds_proba = model.predict_proba(df_valid[training_columns])[:,1]
            preds_cat = [1 if x >= 0.5 else 0 for x in preds_proba]

            # Get metrics
            metrics['acc'].append(accuracy_score(df_valid_res, preds_cat))
            metrics['brier'].append(brier_score_loss(df_valid_res, preds_proba))
            metrics['logloss'].append(log_loss(df_valid_res, preds_proba))
            metrics['f1'].append(f1_score(df_valid_res, preds_cat))

            # If testing calibration, repeat brier with a calibrator
            if calibrate:
                model = build_model(df_train,
                                    training_columns=training_columns,
                                    params=params,
                                    random_state=random_state,
                                    calibrator_size=calibrator_size,
                                    calibrate=True)
                # Get predictions for year
                preds_proba = model.predict_proba(df_valid[training_columns])[:,1]
                preds_cat = [1 if x >= 0.5 else 0 for x in preds_proba]
                metrics['calib-brier'].append(brier_score_loss(df_valid_res, preds_proba))

            if verbose:
                print(year)
                print('Accuracy   ', metrics['acc'][-1])
                print('Brier Loss ', metrics['brier'][-1])
                print('Log Loss   ', metrics['logloss'][-1])
                print('F1 Score   ', metrics['f1'][-1])
                if calibrate:
                    print('Calib Brier', metrics['calib-brier'][-1])
                print()
    
    # Calculate final metrics
    avg_acc = sum(metrics['acc']) / len(metrics['acc'])
    avg_brier = sum(metrics['brier']) / len(metrics['brier'])
    avg_logloss = sum(metrics['logloss']) / len(metrics['logloss'])
    avg_f1 = sum(metrics['f1']) / len(metrics['f1'])
    avg_calib = 0
    if calibrate:
        avg_calib = sum(metrics['calib-brier']) / len(metrics['calib-brier'])
    
    # Output final metrics
    if verbose:
        print('Final Metrics')
        print('Accuracy   ', avg_acc)
        print('Brier Loss ', avg_brier)
        print('Log Loss   ', avg_logloss)
        print('F1 Score   ', avg_f1)
        if calibrate:
            print('Calib Brier', avg_calib)
        print()
    
    return avg_acc, avg_brier, avg_logloss, avg_f1, avg_calib


def tune_model_bayesian_optimisation(training_data, training_columns=TRAINING_COLS,
                                     iterations=5, initial_points=8,
                                     random_state=0, calibrate=False,
                                     calibrator_size=0.2):
    """
    Perform Bayesian Optimisation to find best hyperparameters for model.

    Args:
        training_data: Data used to train model. Must contain training columns.
        training_columns: Columns in training_data used to train model.
        iterations: Bayesian optimisation rounds. Integer.
        initial_points: Bayesian optimisation starting points. Integer.
        random_state: Random value for model and calibration.
        calibrate: If True, fits a probability calibrator after training model.
        calibrator_size: Proportion of training set to use for calibration.

    Returns:
        Dictionary of best hyperparameters for model.
    """
    def bo_tune_function(max_depth, gamma, n_estimators, learning_rate):
        """Objective function for Bayesian optimisation."""
        params = {'max_depth' : int(max_depth),
                  'gamma' : gamma,
                  'n_estimators' : int(n_estimators),
                  'learning_rate' : learning_rate}
        acc, brier, loglogg, f1, calib = cross_validate_model(training_data,
                                                              training_columns=training_columns,
                                                              verbose=False,
                                                              params=params,
                                                              calibrate=calibrate,
                                                              random_state=random_state,
                                                              calibrator_size=calibrator_size)
        if not calibrate:
            return -1*brier
        else:
            return -1*calib

    # Search field for bayesian optimisation of hyperparameters
    search_params = {'max_depth' : (2,7),
                     'gamma' : (0,1),
                     'n_estimators' : (50,200),
                     'learning_rate' : (0,1)}

    # Find best hyperparameters
    xgb_bo = BayesianOptimization(bo_tune_function, search_params, random_state=random_state)
    xgb_bo.maximize(n_iter=iterations, init_points=initial_points)
    best_params = xgb_bo.max['params']

    # Fix type issues
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])

    return best_params


def validate_and_build_model(model_name='default_model', training_columns_men=TRAINING_COLS,
                             training_columns_women=TRAINING_COLS, tune=True, random_state=0,
                             optimisation_iterations=50, optimisation_initial_points=10,
                             calibrate=False, calibrator_size=0.2):
    """
    Get metrics with cross validation, then save complete model.
    
    Args:
        model_name: File name for model. String.
        training_columns_men: Columns in training_data used to train mens
            model.
        training_columns_women: Columns in training_data used to train womens
            model.
        tune: If True, then find best hyperparameters for model.
        random_state: Random value for model and calibration.
        optimisation_iterations: Bayesian optimisation rounds. Integer.
        optimisation_initial_points: Bayesian optimisation starting points.
            Integer.
        calibrate: If True, fits a probability calibrator after training model.
        calibrator_size: Proportion of training set to use for calibration.
    
    Returns:
        Validation Brier score if calibrate is False. Calibrated validation
        Brier score if calibrate is True.
    """

    # Load training data
    training_data = load_training_data()

    # Get separate mens and womens data
    training_data = {'men' : training_data.loc[training_data['Gender'] == 0],
                     'women' : training_data.loc[training_data['Gender'] == 1]}

    for gen, data in training_data.items():
        print('Now Building Model For', gen.title())

        # Find training columns
        training_columns = training_columns_men if gen == 'men' else training_columns_women
        
        # Find best hyperparameters by tuning
        if tune:
            print()
            params = tune_model_bayesian_optimisation(data,
                                                      training_columns=training_columns,
                                                      iterations=optimisation_iterations,
                                                      initial_points=optimisation_initial_points,
                                                      random_state=random_state,
                                                      calibrate=calibrate,
                                                      calibrator_size=calibrator_size)
            print()
            print('Best Hyperparameters')
            print(params)
            print()
        else:
            params = DEFAULT_PARAMS

        # CV and output metrics
        _, brier, _, _, calib = cross_validate_model(data,
                                                     training_columns=training_columns,
                                                     params=params,
                                                     calibrate=calibrate,
                                                     random_state=random_state,
                                                     calibrator_size=calibrator_size)

        # Build final model and save
        model = build_model(data, training_columns=training_columns, params=params,
                            calibrate=calibrate, calibrator_size=calibrator_size)
        dump(model, f'./data/models/{model_name}_{gen}.mdl')

    # Return brier validation metric
    return calib if calibrate else brier


if __name__ == "__main__":
    validate_and_build_model()