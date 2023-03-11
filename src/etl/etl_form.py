import numpy as np
import pandas as pd


def get_game_res_from_group(df, group, game):
    """Gets the x last game from a (Season, TeamID) group."""
    try:
        res = df.get_group(group).iloc[-game].Result
    except:
        res = np.nan
    return res


def get_teams_form():
    """Saves a DataFrame of the results of the last 5 games for each team per season."""

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
    df_form = df[['Season', 'WTeamID']].drop_duplicates()
    df = df.groupby(['Season', 'WTeamID'])
    df_form['G1'] = df_form.apply(lambda x: get_game_res_from_group(df, (x.Season, x.WTeamID), 1), axis=1)
    df_form['G2'] = df_form.apply(lambda x: get_game_res_from_group(df, (x.Season, x.WTeamID), 2), axis=1)
    df_form['G3'] = df_form.apply(lambda x: get_game_res_from_group(df, (x.Season, x.WTeamID), 3), axis=1)
    df_form['G4'] = df_form.apply(lambda x: get_game_res_from_group(df, (x.Season, x.WTeamID), 4), axis=1)
    df_form['G5'] = df_form.apply(lambda x: get_game_res_from_group(df, (x.Season, x.WTeamID), 5), axis=1)


    df_form = df_form.rename(columns={'WTeamID' : 'TeamID'})
    df_form.to_csv('./data/etl/team_form.csv', index=False)


if __name__ == "__main__":
    get_teams_form()