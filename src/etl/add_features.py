import pandas as pd


def merge_team_features(df, file_path, columns, nan_value=0):
    """
    Combines a feature file with the training set twice, once for each team
    to add their features.
    
    Args:
        df: Existing features DataFrame.
        file_path: File path of ETL features to add.
        columns: List of ETL features to add.
        nan_value: Value to replace NaN values with Default is 0.
    
    Returns:
        Combined Pandas DataFrame with new features.
    """
    
    # Load in file and get correct columns
    file = pd.read_csv(file_path)
    file = file[['Season', 'TeamID'] + columns]

    # Merge for team A
    cols_a = [str(x)+'A' for x in columns]
    conv_dict = dict(zip(columns, cols_a))
    df = pd.merge(df,
                  file,
                  how='left',
                  left_on=['Season', 'TeamA'],
                  right_on=['Season', 'TeamID'])
    df = df.rename(columns=conv_dict).drop('TeamID', axis=1)

    # Merge for team B
    cols_b = [str(x)+'B' for x in columns]
    conv_dict = dict(zip(columns, cols_b))
    df = pd.merge(df,
                  file,
                  how='left',
                  left_on=['Season', 'TeamB'],
                  right_on=['Season', 'TeamID'])
    df = df.rename(columns=conv_dict).drop('TeamID', axis=1)

    # Fill NaN
    cols = cols_a + cols_b
    df[cols] = df[cols].fillna(nan_value)

    return df


def add_win_ratio(df):
    """
    Merge win ratio and pts ratio into DataFrame.
    
    Args:
        df: Existing features DataFrame.
    
    Returns:
        Combined Pandas DataFrame with new features.
    """
    df = merge_team_features(df,
                             './data/etl/win_ratios.csv',
                             ['WinRatio', 'PtsForRatio'],
                             nan_value=0)
    return df


def add_massey(df):
    """
    Merge massey ranking stats into DataFrame.
    
    Args:
        df: Existing features DataFrame.
    
    Returns:
        Combined Pandas DataFrame with new features.
    """
    df = merge_team_features(df,
                             './data/etl/massey.csv',
                             ['MasseyMedian', 'MasseyMean', 'MasseyStd'],
                             nan_value=0)
    df['MasseyMedianDiff'] = df['MasseyMedianA'] - df['MasseyMedianB']
    df['MasseyMeanDiff'] = df['MasseyMeanA'] - df['MasseyMeanB']
    return df


def add_form(df):
    """
    Merge form values into DataFrame.
    
    Args:
        df: Existing features DataFrame.
    
    Returns:
        Combined Pandas DataFrame with new features.
    """
    df = merge_team_features(df,
                             './data/etl/team_form.csv',
                             ['FormHarmonic', 'FormUniform'],
                             nan_value=0.5)
    return df


def add_538_ratings(df):
    """
    Merge 538 ratings into DataFrame.
    
    Args:
        df: Existing features DataFrame.
    
    Returns:
        Combined Pandas DataFrame with new features.
    """
    df = merge_team_features(df,
                             './data/etl/538_ratings.csv',
                             ['Rating'],
                             nan_value=60)
    df['RatingDiff'] = df.RatingA - df.RatingB
    return df


def add_tournament_round(df):
    """
    Merge tournament rounds into DataFrame.
    
    Args:
        df: Existing features DataFrame.
    
    Returns:
        Combined Pandas DataFrame with new features.
    """
    ratings = pd.read_csv('./data/etl/tournament_rounds.csv')
    df = pd.merge(df,
                  ratings,
                  how='left',
                  left_on=['Season', 'TeamA', 'TeamB'],
                  right_on=['Season', 'TeamA', 'TeamB'])
    df['Round'] = df['Round'].fillna(0)
    return df


def add_seeds(df):
    """
    Merge tournament seeds into DataFrame.
    
    Args:
        df: Existing features DataFrame.
    
    Returns:
        Combined Pandas DataFrame with new features.
    """
    df = merge_team_features(df,
                             './data/etl/seeds.csv',
                             ['Seed'],
                             nan_value=16)
    df['SeedDiff'] = df.SeedA - df.SeedB
    return df


def add_elo(df, K=32):
    """
    Merge Elo ratings into DataFrame.
    
    Args:
        df: Existing features DataFrame.
        K: K Factor used in Elo calculations. Integer.
    
    Returns:
        Combined Pandas DataFrame with new features.
    """
    df = merge_team_features(df,
                             f'./data/etl/elo{"_"+str(K) if K != 32 else ""}.csv',
                             ['Elo'],
                             nan_value=1500)
    df['EloDiff'] = df.EloA - df.EloB
    df['EloWinProbA'] = 1 / (1 + ( 10 ** ((df.EloB-df.EloA) / 400) ))
    df['EloWinProbB'] = 1 / (1 + ( 10 ** ((df.EloA-df.EloB) / 400) ))
    return df


def add_gender(df):
    """
    Merge gender feature with DataFrame.
    
    Args:
        df: Existing features DataFrame.
    
    Returns:
        Combined Pandas DataFrame with new features.
    """
    df_gender = pd.read_csv('./data/etl/genders.csv')
    df_gender = df_gender.rename(columns={'TeamID' : 'TeamA'})
    df = pd.merge(df, df_gender, how='left', on='TeamA')
    df.fillna(0)
    return df


def add_clutch(df):
    """
    Merge clutch game features with DataFrame.

    Args:
        df: Existing features DataFrame.
    
    Returns:
        Combined Pandas DataFrame with new features.
    """
    df = merge_team_features(df,
                             './data/etl/clutch_games.csv',
                             ['ClutchRatio'],
                             nan_value=0.5)
    return df


def build_training_set(elo_K=32, mens_start_year=1985, womens_start_year=1985):
    """
    Calculate all features for training dataset and save to file.

    Args:
        elo_K: K Factor used in Elo calculations. Integer.
        mens_start_year: First year to include mens training data.
        womens_start_year: First year to include womens training data.
    
    Returns:
        Pandas DataFrame for test set with all loaded features.
    """

    # Load mens tourney results
    df_men = pd.read_csv('./data/kaggle/MNCAATourneyCompactResults.csv')
    df_men['WinGap'] = df_men['WScore'] - df_men['LScore']
    df_men = df_men[['Season', 'WTeamID', 'LTeamID', 'WinGap']]

    # Load womens tourney results
    df_women = pd.read_csv('./data/kaggle/WNCAATourneyCompactResults.csv')
    df_women['WinGap'] = df_women['WScore'] - df_women['LScore']
    df_women = df_women[['Season', 'WTeamID', 'LTeamID', 'WinGap']]

    # Concat tourney results
    df = pd.concat([df_men, df_women], ignore_index=True)

    # Duplicate rows by swapping winning and losing teams
    df_flipped = df.copy()
    df_flipped['WinGap'] = 0 - df_flipped['WinGap']
    df_flipped = df_flipped.rename(columns={'WTeamID':'LTeamID', 'LTeamID':'WTeamID'})
    df = pd.concat([df, df_flipped], ignore_index=True)

    # Change column names
    df['TeamA'] = df['WTeamID']
    df['TeamB'] = df['LTeamID']
    df = df[['Season', 'TeamA', 'TeamB', 'WinGap']]
    
    # Add binary win feature
    df['Win'] = df['WinGap'].apply(lambda x: 1 if x > 0 else 0)

    # Add training features
    df = add_gender(df)
    df = add_form(df)
    df = add_win_ratio(df)
    df = add_elo(df, K=elo_K)
    df = add_seeds(df)
    df = add_tournament_round(df)
    df = add_538_ratings(df)
    df = add_clutch(df)
    df = add_massey(df)
    
    # Save training set
    df = df.sort_values('Season', ignore_index=True)
    df = df[((df.Season >= mens_start_year) & (df.Gender == 0)) |
            ((df.Season >= womens_start_year) & (df.Gender == 1))]
    df.to_csv('./data/etl/training_set.csv', index=False)


def prep_submission_frame_2023():
    """
    Load in template prediction frame.

    Returns:
        Pandas DataFrame with all test games to predict.
    """
    df_template = pd.read_csv('./data/kaggle/SampleSubmission2023.csv')

    # Split ID into values
    df_template['Season'] = df_template['ID'].apply(lambda x: x.split('_')[0]).astype(int)
    df_template['TeamA'] = df_template['ID'].apply(lambda x: x.split('_')[1]).astype(int)
    df_template['TeamB'] = df_template['ID'].apply(lambda x: x.split('_')[2]).astype(int)

    return df_template


def prep_submission_frame():
    """
    Build the submission frame.

    Returns:
        Pandas DataFrame with all test games to predict.
    """
    df = pd.read_csv('./data/kaggle/2024_tourney_seeds.csv')

    # Get a list of teams
    w_teams = df.loc[df.Tournament == 'W', 'TeamID'].values
    m_teams = df.loc[df.Tournament == 'M', 'TeamID'].values
    w_teams_comb = [[i, j] for i in w_teams for j in w_teams if i < j]
    m_teams_comb = [[i, j] for i in m_teams for j in m_teams if i < j]
    teams = w_teams_comb + m_teams_comb

    # Build dataset to predict on
    df = pd.DataFrame(teams, columns=['TeamA', 'TeamB'])
    df['Season'] = 2024
    df['ID'] = df.Season.astype(str) + '_' + df.TeamA.astype(str) + '_' + df.TeamB.astype(str)

    return df


def build_test_set(elo_K=32):
    """
    Calculate all features for test dataset and save to file.
    
    Args:
        elo_K: K Factor used in Elo calculations. Integer.
    
    Returns:
        Pandas DataFrame for test set with all loaded features.
    """

    # Load template for current season
    df = prep_submission_frame()
    
    # Add training features
    df = add_gender(df)
    df = add_form(df)
    df = add_win_ratio(df)
    df = add_elo(df, K=elo_K)
    df = add_seeds(df)
    df = add_tournament_round(df)
    df = add_538_ratings(df)
    df = add_clutch(df)
    df = add_massey(df)

    # Save training set
    df.to_csv('./data/etl/test_set.csv', index=False)


if __name__ == "__main__":
    build_training_set()
    build_test_set()