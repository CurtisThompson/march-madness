# Scott Lundberg examples
# https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27

# SHAP GitHub
# https://github.com/slundberg/shap

# Microsoft Responsible AI Standard
# https://blogs.microsoft.com/wp-content/uploads/prod/sites/5/2022/06/Microsoft-Responsible-AI-Standard-v2-General-Requirements-3.pdf

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


MODEL_COLS = ['SeedDiff', 'EloWinProbA', 'WinRatioA', 'WinRatioB', 'ClutchRatioA', 'ClutchRatioB']


def load_features():
    """Load in calculated features."""
    return pd.read_csv('./data/etl/test_set.csv')


def load_models(name='default_model'):
    """Loads separate mens and womens classifier models."""
    men_model = XGBClassifier()
    men_model.load_model(f'./data/models/{name}_men.mdl')
    women_model = XGBClassifier()
    women_model.load_model(f'./data/models/{name}_women.mdl')
    return men_model, women_model


def load_predictions(name='preds'):
    """Load model predictions."""
    return pd.read_csv(f'./data/predictions/{name}.csv')


def load_training_data():
    """Load data used to train models."""
    df = pd.read_csv('./data/etl/training_set.csv')
    df_women = df.loc[df.Gender == 1]
    df_men = df.loc[df.Gender == 0]
    return df_men, df_women


def get_shap_values(model, df, columns, training_data):
    """Gets the shap values for a dataset with respect to a given model."""

    # Get shap values
    shap_cols = ['shap_'+c for c in columns]
    shap_exp = shap.TreeExplainer(model, model_output='probability', data=training_data[columns])
    shap_vals = shap_exp.shap_values(df[columns])
    shap_vals = pd.DataFrame(shap_vals, columns=shap_cols)

    # Add row ID and base value
    shap_vals['ID'] = df['ID']
    shap_vals['shap_BaseVal'] = shap_exp.expected_value
    return shap_vals


def force_plot(shapset, columns):
    """Save a force plot of an individual prediction."""

    # Plot values
    row_id = shapset['ID']
    base_val = shapset['shap_BaseVal']
    shap_vals = shapset[['shap_'+c for c in columns]].values
    column_vals = shapset[columns].values
    out_name = 'Team A Win Probability'

    # Round column values for display
    column_vals = np.array(["{:.2f}".format(c) if type(c) == np.float64 else c for c in column_vals])

    # Plot and save
    shap.plots.force(base_val, shap_values=shap_vals, features=column_vals, feature_names=columns, out_names='',
                     matplotlib=True, show=False, contribution_threshold=0)
    plt.tight_layout()
    plt.savefig(f'viz/visuals/force_plot_{row_id}.png')


def build_shap_sets(mens_columns=MODEL_COLS, womens_columns=MODEL_COLS, model_names='default_model'):
    # Get data
    df = load_features()
    df = df.drop('Pred', axis=1)
    df_pred = load_predictions()
    df = pd.merge(df, df_pred, how='left', on='ID')

    # Split into mens and womens
    df_men = df.loc[df.Gender == 0, mens_columns+['ID', 'Pred']]
    df_women = df.loc[df.Gender == 1, womens_columns+['ID', 'Pred']]

    # Sample
    df_men = df_men.sample(100, random_state=0).reset_index(drop=True)
    df_women = df_women.sample(100, random_state=0).reset_index(drop=True)

    # Get models
    mmodel, wmodel = load_models(name=model_names)

    # Get training_data
    train_men, train_women = load_training_data()
    train_men = train_men[mens_columns]
    train_women = train_women[womens_columns]

    # Find shap values for mens
    shap_men = get_shap_values(mmodel, df_men, mens_columns, train_men)
    df_men = pd.merge(df_men, shap_men, how='left', on='ID')

    # Find shap values for womens
    shap_women = get_shap_values(wmodel, df_women, womens_columns, train_women)
    df_women = pd.merge(df_women, shap_women, how='left', on='ID')

    return df_men, df_women


def save_shap_set(df_men, df_women):
    """Save shap datasets."""
    df_men.to_csv('./data/explain/shap_men.csv', index=False)
    df_women.to_csv('./data/explain/shap_women.csv', index=False)


def load_shap_set():
    """Load shap datasets."""
    df_men = pd.read_csv('./data/explain/shap_men.csv')
    df_women = pd.read_csv('./data/explain/shap_women.csv')
    return df_men, df_women


if __name__ == "__main__":
    is_load_shap_set = False

    if is_load_shap_set:
        df_men, df_women = load_shap_set()
    else:
        df_men, df_women = build_shap_sets()
        save_shap_set(df_men, df_women)
    
    for i in range(3):
        force_plot(df_men.iloc[i], MODEL_COLS)
        force_plot(df_women.iloc[i], MODEL_COLS)