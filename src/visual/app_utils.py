import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from PIL import Image


def get_teams_list(is_women=True):
    """
    Return a list of dicts with team ID and name as label and value.
    
    Args:
        is_women: If True, get womens team. If False, get mens team.
    
    Returns:
        List of dictionaries with team IDs as value and team name as label.
    """
    path = './data/kaggle/WTeams.csv' if is_women else './data/kaggle/MTeams.csv'
    teams = pd.read_csv(path).rename(columns={'TeamID':'value', 'TeamName':'label'})
    teams = teams[['label', 'value']].to_dict(orient='records')
    return teams


def get_teams_dict(team_list):
    """
    Return a label:value dict of team IDs and names.

    Args:
        team_list: List of dictionaries for each team.

    Returns:
        Dictionary of team IDs to names.
    """
    return dict([(x['value'], x['label']) for x in team_list])


def load_shap_sets():
    """
    Return separate shap datasets for mens and womens teams.

    Returns:
        Two Pandas DataFrames, for men shap values and women shap values.
    """
    return pd.read_csv('./data/explain/shap_men.csv'), pd.read_csv('./data/explain/shap_women.csv')


def load_game_predictions():
    """
    Returns a DataFrame of match predictions for every team.

    Returns:
        Pandas DataFrame with match predictions.
    """
    return pd.read_csv('./data/predictions/preds.csv')


def create_force_plot(shapset):
    """
    Build a base64 string of a shap force plot from a given row in
    the shap dataset.
    
    Args:
        shapset: DataFrame of Shap values.
    
    Returns:
        String that includes representation of force plot image, which can
        be included in a HTML document.
    """

    # Plot values
    columns = ['SeedDiff', 'EloWinProbA', 'WinRatioA', 'WinRatioB', 'ClutchRatioA', 'ClutchRatioB']
    base_val = shapset['shap_BaseVal']
    shap_vals = shapset[['shap_'+c for c in columns]].values
    column_vals = shapset[columns].values
    final_shap = sum(shap_vals)+base_val

    # Round column values for display
    column_vals = np.array(["{:.2f}".format(c) if type(c) == np.float64 else c for c in column_vals])

    # Allocate memory buffer for image
    buffer = io.BytesIO()

    # Plot
    shap.plots.force(base_val, shap_values=shap_vals, feature_names=columns, out_names='',
                     matplotlib=True, show=False, contribution_threshold=0)
    
    # Modify axes to have 0.5 as centre
    ax = plt.gca()
    lim_bot, lim_top = ax.get_xlim()
    max_lim_move = max(0.5-lim_bot, lim_top-0.5)
    ax.set_xlim(0.5-max_lim_move, 0.5+max_lim_move)
    plt.axvline(x=0.5, color='black', linestyle='--', ymin=0.5)
    plt.axvline(x=final_shap, color='red' if final_shap > 0.5 else 'blue', ymin=0.5)

    # Save plot
    plt.savefig(buffer, format='png', pad_inches = 0)
    plt.close()

    # Get image data from buffer, clear and return
    buffer_data = base64.b64encode(buffer.getbuffer()).decode("utf8")
    buffer.close()
    buffer_data = chop_force_plot_image(buffer_data)
    return buffer_data


def chop_force_plot_image(buffer_data):
    """
    Removes white space from bottom of force plot image.
    
    Args:
        buffer_data: base64 string of force plot image.
    
    Returns:
        base64 string of force plot image with whitespace removed.
    """
    im = Image.open(io.BytesIO(base64.b64decode(buffer_data)))
    im = im.crop((0, 0, im.size[0], int(im.size[1]/2)))
    temp_buf = io.BytesIO()
    im.save(temp_buf, format='png')
    im_str = base64.b64encode(temp_buf.getbuffer()).decode("utf8")
    temp_buf.close()
    return im_str


def fancy_win_prob(num):
    """
    Returns a pretty string for a match win probability.

    Args:
        num: Float value between 0 and 1.

    Returns:
        String represention of input rounded to 1 decimal place, e.g. 74.1%.
    """
    str1 = str(round(num, 3) * 100)
    str_parts = str1.split('.')
    str_final = str_parts[0]
    if len(str_parts) > 1 and str_parts[1][0] != '0':
        str_final += '.' + str_parts[1][0]
    return str_final + '%'


def cannot_update_visuals(gender, teama, teamb):
    """
    Returns True if inputs mean app visuals cannot be updated.
    
    Args:
        gender: Updated gender.
        teama: Updated team A.
        teamb: Updated team B.
    
    Returns:
        Boolean. True if any inputs are None, or if teams are same.
    """
    if (gender == None) or (teama == None) or (teamb == None) or (teama == teamb):
        return True
    return False


def get_match_id(year, teama, teamb):
    """
    Return the match ID, as well as home and away teams.
    
    Args:
        year: Prediction year.
        teama: First user-inputted team.
        teamb: Second user-inputted team.

    Returns:
        String of format year_team_team, as well as teams ordered ascending.
    """
    home_team = min(teama, teamb)
    away_team = max(teama, teamb)
    match_id = str(year) + '_' + str(home_team) + '_' + str(away_team)
    return match_id, home_team, away_team


def get_match_win_probs(df_prob, match_id):
    """
    Returns the home and away win probabilities of the match.

    Args:
        df_prob: Pandas DataFrame of match probabilities.
        match_id: String of format year_team_team.
    
    Returns:
        Team A win probability and Team B win probability.
    """

    # Find match predictions in shap dataset
    result = df_prob.loc[df_prob.ID == match_id].reset_index(drop=True).iloc[0]

    # Calculate win probs
    home_win_prob = fancy_win_prob(result['Pred'])
    away_win_prob = fancy_win_prob(1-result['Pred'])

    return home_win_prob, away_win_prob


def get_teams_win_loss(year):
    """
    Returns a DataFrame with win and loss count for each team in given season.
    
    Args:
        year: Season.
    
    Returns:
        Pandas DataFrame.
    """
    df = pd.read_csv('./data/etl/win_ratios.csv')
    df = df.loc[df.Season == year, ['TeamID', 'Wins', 'Losses']]
    return df


def get_teams_form(year=2023):
    """
    Returns a DataFrame of the results of the last 5 games for each team.
    
    Args:
        year: Season.
    
    Returns:
        Pandas DataFrame.
    """
    df_form = pd.read_csv('./data/etl/team_form.csv').replace(np.nan, 'D', inplace=True)
    df_form = df_form.loc(df_form.Season == year, ['TeamID', 'G1', 'G2', 'G3', 'G4', 'G5'])
    return df_form


def get_team_win_record(team, df):
    """
    Get a string version of a team win record.
    
    Args:
        team: Team ID. Integer.
        df: Pandas DataFrame with wins and losses in season.
    
    Returns:
        String containing wins and losses.
    """
    row = df.loc[df.TeamID == team].reset_index(drop=True).iloc[0]
    wins = row['Wins']
    losses = row['Losses']
    return f'{wins} - {losses}'


def load_test_set():
    """
    Load test set.
    
    Returns:
        Pandas DataFrame with test dataset.
    """
    return pd.read_csv('./data/etl/test_set.csv')


def get_feature_table(test_set, features, match_id):
    """
    Get table of features for match.

    Args:
        test_set: Pandas DataFrame with test features.
        features: List of features to display.
        match_id: Match to display.
    
    Returns:
        Pandas DataFrame formatted with given features.
    """
    
    # Get match features
    row = test_set.loc[test_set.ID == match_id].reset_index(drop=True).iloc[0]

    # Find feature values
    features_a = row[[x+'A' for x in features]].values
    features_b = row[[x+'B' for x in features]].values
    df = pd.DataFrame([features_a, features, features_b]).transpose().rename(columns={0:'TeamA', 1:'Statistics', 2:'TeamB'})

    # Change types
    for index, row in df.iterrows():
        for col in df:
            if (type(df.loc[index, col]) == np.float64):
                if (df.loc[index, col] <= 1):
                    df.loc[index, col] = fancy_win_prob(df.loc[index, col])
                else:
                    df.loc[index, col] = str(int(df.loc[index, col]))
    return df