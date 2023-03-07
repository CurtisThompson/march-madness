import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def get_teams_list(is_women=True):
    """Returns a list of dicts with team ID and name as label and value."""
    path = './data/kaggle/WTeams.csv' if is_women else './data/kaggle/MTeams.csv'
    teams = pd.read_csv(path).rename(columns={'TeamID':'value', 'TeamName':'label'})
    teams = teams[['label', 'value']].to_dict(orient='records')
    return teams


def get_teams_dict(team_list):
    """Returns a label:value dict of team IDs and names."""
    return dict([(x['value'], x['label']) for x in team_list])


def load_shap_sets():
    """Returns separate shap datasets for mens and womens teams."""
    return pd.read_csv('./data/explain/shap_men.csv'), pd.read_csv('./data/explain/shap_women.csv')


def load_game_predictions():
    """Returns a DataFrame of match predictions for every team."""
    return pd.read_csv('./data/predictions/preds.csv')


def create_force_plot(shapset):
    """Builds a base64 string of a shap force plot from a given row in the shap dataset."""

    # Plot values
    columns = ['SeedDiff', 'EloWinProbA', 'WinRatioA', 'WinRatioB', 'ClutchRatioA', 'ClutchRatioB']
    base_val = shapset['shap_BaseVal']
    shap_vals = shapset[['shap_'+c for c in columns]].values
    column_vals = shapset[columns].values

    # Round column values for display
    column_vals = np.array(["{:.2f}".format(c) if type(c) == np.float64 else c for c in column_vals])

    # Allocate memory buffer for image
    buffer = io.BytesIO()

    # Plot and save
    shap.plots.force(base_val, shap_values=shap_vals, features=column_vals, feature_names=columns, out_names='',
                     matplotlib=True, show=False, contribution_threshold=0)
    plt.savefig(buffer, format='png')
    plt.close()

    # Get image data from buffer, clear and return
    buffer_data = base64.b64encode(buffer.getbuffer()).decode("utf8")
    buffer.close()
    return buffer_data


def fancy_win_prob(num):
    """Returns a pretty string for a match win probability."""
    str1 = str(round(num, 3) * 100)
    str_parts = str1.split('.')
    str_final = str_parts[0]
    if len(str_parts) > 1 and str_parts[1][0] != '0':
        str_final += '.' + str_parts[1][0]
    return str_final + '%'


def cannot_update_visuals(gender, teama, teamb):
    """Returns True if inputs mean app visuals cannot be updated."""
    if (gender == None) or (teama == None) or (teamb == None) or (teama == teamb):
        return True
    return False


def get_match_id(year, teama, teamb):
    """Returns the match ID, as well as home and away teams."""
    home_team = min(teama, teamb)
    away_team = max(teama, teamb)
    match_id = str(year) + '_' + str(home_team) + '_' + str(away_team)
    return match_id, home_team, away_team


def get_match_win_probs(df_prob, match_id):
    """Returns the home and away win probabilities of the match."""

    # Find match predictions in shap dataset
    result = df_prob.loc[df_prob.ID == match_id].reset_index(drop=True).iloc[0]

    # Calculate win probs
    home_win_prob = fancy_win_prob(result['Pred'])
    away_win_prob = fancy_win_prob(1-result['Pred'])

    return home_win_prob, away_win_prob