import pandas as pd
from joblib import load


MODEL_COLS = ['SeedDiff', 'EloWinProbA', 'WinRatioA', 'WinRatioB', 'ClutchRatioA', 'ClutchRatioB']
PRED_COLS = ['ID', 'Pred']


def load_submission_frame():
    """
    Load in template prediction frame with calculated features.
    
    Returns:
        Pandas DataFrame with test features.
    """
    return pd.read_csv('./data/etl/test_set.csv')


def load_models(name='default_model'):
    """
    Loads separate mens and womens classifier models.
    
    Args:
        name: Name of model to load. Default is default_model.
    
    Returns:
        Loaded mens model and womens model.
    """
    men_model = load(f'./data/models/{name}_men.mdl')
    women_model = load(f'./data/models/{name}_women.mdl')
    return men_model, women_model


def predict_frame(df, men_model, women_model, model_columns_men, model_columns_women):
    """
    Predict the outcome of multiple games.
    
    Args:
        df: Test dataset.
        men_model: Model for mens predictions. Must have predict_proba method.
        women_model: Model for womens predictions. Must have predict_proba method.
        model_columns_men: List of columns required for mens predictions.
        model_columns_women: List of columns required for womens predictions.
    
    Returns:
        Test dataset with added Pred column, with model predictions.
    """
    df.loc[df['Gender'] == 0, 'Pred'] = men_model.predict_proba(df.loc[df['Gender'] == 0][model_columns_men])[:,1]
    df.loc[df['Gender'] == 1, 'Pred'] = women_model.predict_proba(df.loc[df['Gender'] == 1][model_columns_women])[:,1]
    return df


def save_predictions(df, file_name="preds", file_path='./data/predictions/'):
    """
    Save all game predictions to file.

    Args:
        df: Pandas DataFrame with predictions.
        file_name: Name of file to save predictions to. String.
        file_path: Directory to save predictions to. String.
    """
    df.to_csv(file_path + file_name + '.csv', index=False)


def run(model_columns=MODEL_COLS, model_columns_men=None, model_columns_women=None,
        file_name="preds", file_path='./data/predictions/'):
    """
    Load in the prediction template, make predictions, and save to file.

    Args:
        model_columns: Default columns if mens or womens columns are None.
        model_columns_men: List of columns required for mens predictions.
        model_columns_women: List of columns required for womens predictions.
        file_name: Name of file to save predictions to. String.
        file_path: Directory to save predictions to. String.
    """
    
    # Set model columns
    model_columns_men = model_columns if model_columns_men == None else model_columns_men
    model_columns_women = model_columns if model_columns_women == None else model_columns_women
    
    # Load data and models
    df = load_submission_frame()
    mmodel, wmodel = load_models()

    # Create and save predictions
    df = predict_frame(df, mmodel, wmodel, model_columns_men, model_columns_women)
    df = df[PRED_COLS]
    df['Pred'] = df['Pred'].clip(0, 1)
    save_predictions(df, file_name=file_name, file_path=file_path)


if __name__ == "__main__":
    run()