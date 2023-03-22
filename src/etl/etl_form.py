import numpy as np
import pandas as pd


def get_uniform_form(results, group, last_n=10):
    """
    Calculate a form value with each match taking equal weighting.
    
    Args:
        results: Pandas DataFrame with match results.
        group: (Season, TeamID) tuple for team to extract form.
        last_n: Number of games to include in form calculations. Integer.
    
    Returns:
        Integer value between 0 and 1 representing the win ratio in the last
        n games.
    """

    # Get results for season and team
    results = results.get_group(group).reset_index(drop=True)

    # Only calculate form based on last n games
    results = results.tail(last_n)

    # Get uniform form
    return sum([1 if x == 'W' else 0 for x in results['Result'].values]) / results.shape[0]


def get_harmonic_form(results, group, last_n=10, similar_n=3):
    """
    Calculate a form value based on the harmonic series, where more recent
    games are more valuable.

    Args:
        results: Pandas DataFrame with match results.
        group: (Season, TeamID) tuple for team to extract form.
        last_n: Number of games to include in form calculations. Integer.
        similar_n: The most recent number of games to take the same weighting.
    
    Returns:
        Integer value between 0 and 1 representing the recent form of a team.
    """

    # Get results for season and team
    results = results.get_group(group).reset_index(drop=True)

    # Change day num to force most recent n to be the same
    results.loc[results.index[-similar_n:], 'DayNum'] = results.loc[results.index[-1], 'DayNum']

    # Work out values in harmonic series
    results['Sign'] = results['Result'].apply(lambda x: 1 if x == 'W' else -1)
    results['Fraction'] = 1 / results['DayNum'].rank(method='dense', ascending=False)
    results['Value'] = results['Sign'] * results['Fraction']

    # Only calculate form based on last n games
    results = results.tail(last_n)

    # Sum and normalise
    max_val = results['Fraction'].sum()
    min_val = -1 * max_val
    total = results['Value'].sum()
    total = (total - min_val) / (max_val - min_val)
    
    # Force between 0 and 1 in case of division errors
    total = min(max(total, 0), 1)

    return total


def get_game_res_from_group(df, group, game):
    """
    Gets the x last game from a (Season, TeamID) group.
    
    Args:
        df: A Pandas GroupBy DataFrame with results for each Season and Team.
        group: (Season, TeamID) tuple for team to extract.
        game: The nth last game to get results for. Integer.
    
    Returns:
        String 'W' or 'L' representing result of game.
    """
    try:
        res = df.get_group(group).iloc[-game].Result
    except:
        res = np.nan
    return res


def get_teams_form(form_game_window=10, form_game_similar=3):
    """
    Save a DataFrame of the results of the last 5 games for each team per
    season. Save to team_form.csv.
    
    Args:
        form_game_window: The number of games to include in form calculation.
        form_game_similar: The most recent number of games to take the same
            weighting in harmonic form calculations.
    """

    # Get all results
    dfm = pd.read_csv('./data/kaggle/MRegularSeasonCompactResults.csv')
    dfw = pd.read_csv('./data/kaggle/WRegularSeasonCompactResults.csv')
    df = pd.concat([dfm, dfw])[['Season', 'DayNum', 'WTeamID', 'LTeamID']]
    df['Result'] = 'W'

    # Get inverse results (include losing team first)
    df2 = df.copy()
    df2[['WTeamID', 'LTeamID']] = df[['LTeamID', 'WTeamID']]
    df2['Result'] = 'L'
    df = pd.concat([df, df2], ignore_index=True)
    df = df.sort_values(['Season', 'DayNum']).reset_index(drop=True)

    # Make form DataFrame
    df_form = df[['Season', 'WTeamID']].drop_duplicates()
    df = df.groupby(['Season', 'WTeamID'])
    df_form['G1'] = df_form.apply(lambda x: get_game_res_from_group(df, (x.Season, x.WTeamID), 1), axis=1)
    df_form['G2'] = df_form.apply(lambda x: get_game_res_from_group(df, (x.Season, x.WTeamID), 2), axis=1)
    df_form['G3'] = df_form.apply(lambda x: get_game_res_from_group(df, (x.Season, x.WTeamID), 3), axis=1)
    df_form['G4'] = df_form.apply(lambda x: get_game_res_from_group(df, (x.Season, x.WTeamID), 4), axis=1)
    df_form['G5'] = df_form.apply(lambda x: get_game_res_from_group(df, (x.Season, x.WTeamID), 5), axis=1)

    # Add harmonic and uniform form numerical values
    df_form['FormHarmonic'] = df_form.apply(lambda x: get_harmonic_form(df,
                                                                        (x.Season, x.WTeamID),
                                                                        last_n=form_game_window,
                                                                        similar_n=form_game_similar), axis=1)
    df_form['FormUniform'] = df_form.apply(lambda x: get_uniform_form(df,
                                                                      (x.Season, x.WTeamID),
                                                                      last_n=form_game_window), axis=1)
    
    df_form = df_form.rename(columns={'WTeamID' : 'TeamID'})
    df_form.to_csv('./data/etl/team_form.csv', index=False)


if __name__ == "__main__":
    get_teams_form()