import pandas as pd
from joblib import load


MODEL_COLS = ['SeedDiff', 'EloWinProbA', 'WinRatioA', 'WinRatioB', 'ClutchRatioA', 'ClutchRatioB']
PRED_COLS = ['ID', 'Pred']


def load_submission_frame():
    """Load in template prediction frame with calculated features."""
    return pd.read_csv('./data/etl/test_set.csv')


def load_models(name='default_model'):
    """Loads separate mens and womens classifier models."""
    men_model = load(f'./data/models/{name}_men.mdl')
    women_model = load(f'./data/models/{name}_women.mdl')
    return men_model, women_model


def predict_row(row):
    """Predict the outcome of a single game."""
    return 0.2


def predict_frame(df, men_model, women_model, model_columns_men, model_columns_women):
    """Predict the outcome of multiple games."""
    df.loc[df['Gender'] == 0, 'Pred'] = men_model.predict_proba(df.loc[df['Gender'] == 0][model_columns_men])[:,1]
    df.loc[df['Gender'] == 1, 'Pred'] = women_model.predict_proba(df.loc[df['Gender'] == 1][model_columns_women])[:,1]
    return df


def save_predictions(df, file_name="preds", file_path='./data/predictions/'):
    """Save all game predictions to file."""
    df.to_csv(file_path + file_name + '.csv', index=False)


def run(model_columns=MODEL_COLS, model_columns_men=None, model_columns_women=None, file_name="preds",
        file_path='./data/predictions/'):
    """Load in the prediction template, make predictions, and save to file."""
    
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