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


def get_teams_win_loss(year):
    """Returns a DataFrame with win and loss count for each team in given season."""
    df = pd.read_csv('./data/etl/win_ratios.csv')
    df = df.loc[df.Season == year, ['TeamID', 'Wins', 'Losses']]
    return df


def get_teams_form():
    """Returns a DataFrame of the results of the last 5 games for each team."""

    # Get all results
    dfm = pd.read_csv('./data/kaggle/MRegularSeasonCompactResults.csv')
    dfw = pd.read_csv('./data/kaggle/WRegularSeasonCompactResults.csv')
    df = pd.concat([dfm, dfw])[['Season', 'DayNum', 'WTeamID', 'LTeamID']]
    df['Result'] = 'W'

    # Get inverse results (include losing team first)
    df2 = df.copy()
    df2[['WTeamID', 'LTeamID']] = df[['LTeamID', 'WTeamID']]
    df2['Result'] = 'L'
    df = pd.concat([df, df2], ignore_index=True).sort_values(['Season', 'DayNum']).reset_index(drop=True)

    # Make form DataFrame
    df_form = pd.DataFrame(df.WTeamID.unique(), columns=['Team'])
    df = df.groupby('WTeamID')
    df_form['G1'] = df_form.Team.apply(lambda x: df.get_group(x).iloc[-1].Result)
    df_form['G2'] = df_form.Team.apply(lambda x: df.get_group(x).iloc[-2].Result)
    df_form['G3'] = df_form.Team.apply(lambda x: df.get_group(x).iloc[-3].Result)
    df_form['G4'] = df_form.Team.apply(lambda x: df.get_group(x).iloc[-4].Result)
    df_form['G5'] = df_form.Team.apply(lambda x: df.get_group(x).iloc[-5].Result)
    return df_form


def get_team_win_record(team, df):
    """Get a string version of a team win record."""
    row = df.loc[df.TeamID == team].reset_index(drop=True).iloc[0]
    wins = row['Wins']
    losses = row['Losses']
    return f'{wins} - {losses}'


def load_test_set():
    """Load test set."""
    return pd.read_csv('./data/etl/test_set.csv')


def get_feature_table(test_set, features, match_id):
    """Get table of features for match."""
    
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