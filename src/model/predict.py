import pandas as pd
from xgboost import XGBClassifier


MODEL_COLS = ['SeedDiff', 'EloWinProbA', 'Gender']
PRED_COLS = ['ID', 'Pred']


def load_submission_frame():
    """Load in template prediction frame with calculated features."""
    return pd.read_csv('./data/etl/test_set.csv')


def load_models(name='default_model'):
    """Loads separate mens and womens classifier models."""
    men_model = XGBClassifier()
    men_model.load_model(f'./data/models/{name}_men.mdl')
    women_model = XGBClassifier()
    women_model.load_model(f'./data/models/{name}_women.mdl')
    return men_model, women_model


def predict_row(row):
    """Predict the outcome of a single game."""
    return 0.2


def predict_frame(df, men_model, women_model, model_columns):
    """Predict the outcome of multiple games."""
    #df['Pred'] = df.apply(predict_row, axis=1)
    df.loc[df['Gender'] == 0, 'Pred'] = men_model.predict_proba(df.loc[df['Gender'] == 0][model_columns])[:,1]
    df.loc[df['Gender'] == 1, 'Pred'] = women_model.predict_proba(df.loc[df['Gender'] == 1][model_columns])[:,1]
    return df


def save_predictions(df, file_name="preds", file_path='./data/predictions/'):
    """Save all game predictions to file."""
    df = df[PRED_COLS]
    df.to_csv(file_path + file_name + '.csv', index=False)


def run(model_columns=MODEL_COLS):
    """Load in the prediction template, make predictions, and save to file."""
    df = load_submission_frame()
    mmodel, wmodel = load_models()
    df = predict_frame(df, mmodel, wmodel, model_columns)
    save_predictions(df)


if __name__ == "__main__":
    run()