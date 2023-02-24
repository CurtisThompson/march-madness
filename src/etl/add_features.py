import numpy as np
import pandas as pd


def add_win_ratio(df):
    """Merge win ratio and pts rato into DataFrame."""
    win_ratios = pd.read_csv('./data/etl/win_ratios.csv')
    win_ratios = win_ratios[['Season', 'TeamID', 'WinRatio', 'PtsForRatio']]
    df = pd.merge(df, win_ratios, how='left', left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'])
    df = df.rename({'WinRatio':'WinRatioA', 'PtsForRatio':'PtsForRatioA'}, axis=1).drop('TeamID', axis=1)
    df = pd.merge(df, win_ratios, how='left', left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'])
    df = df.rename({'WinRatio':'WinRatioB', 'PtsForRatio':'PtsForRatioB'}, axis=1).drop('TeamID', axis=1)
    df[['WinRatioA', 'WinRatioB']] = df[['WinRatioA', 'WinRatioB']].fillna(0)
    df[['PtsForRatioA', 'PtsForRatioB']] = df[['PtsForRatioA', 'PtsForRatioB']].fillna(0)
    return df


def add_538_ratings(df):
    """Merge 538 ratings into DataFrame."""
    ratings = pd.read_csv('./data/etl/538_ratings.csv')
    df = pd.merge(df, ratings, how='left', left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'])
    df = df.rename({'Rating':'RatingA'}, axis=1).drop('TeamID', axis=1)
    df = pd.merge(df, ratings, how='left', left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'])
    df = df.rename({'Rating':'RatingB'}, axis=1).drop('TeamID', axis=1)
    df[['RatingA', 'RatingB']] = df[['RatingA', 'RatingB']].fillna(60)
    df['RatingDiff'] = df.RatingA - df.RatingB
    return df


def add_seeds(df):
    """Merge tournament seeds into DataFrame."""
    seeds = pd.read_csv('./data/etl/seeds.csv')
    df = pd.merge(df, seeds, how='left', left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'])
    df = df.rename({'Seed':'SeedA'}, axis=1).drop('TeamID', axis=1)
    df = pd.merge(df, seeds, how='left', left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'])
    df = df.rename({'Seed':'SeedB'}, axis=1).drop('TeamID', axis=1)
    df[['SeedA', 'SeedB']] = df[['SeedA', 'SeedB']].fillna(16)
    df['SeedDiff'] = df.SeedA - df.SeedB
    return df


def add_elo(df, K=32):
    """Merge Elo ratings into DataFrame."""
    elos = pd.read_csv(f'./data/etl/elo{"_"+str(K) if K != 32 else ""}.csv')
    df = pd.merge(df, elos, how='left', left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'])
    df = df.rename({'Elo':'EloA'}, axis=1).drop('TeamID', axis=1)
    df = pd.merge(df, elos, how='left', left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'])
    df = df.rename({'Elo':'EloB'}, axis=1).drop('TeamID', axis=1)
    df[['EloA', 'EloB']] = df[['EloA', 'EloB']].fillna(1500)
    df['EloDiff'] = df.EloA - df.EloB
    df['EloWinProbA'] = 1 / (1 + ( 10 ** ((df.EloB-df.EloA) / 400) ))
    df['EloWinProbB'] = 1 / (1 + ( 10 ** ((df.EloA-df.EloB) / 400) ))
    return df


def add_gender(df):
    """Merge gender feature with DataFrame."""
    df_gender = pd.read_csv('./data/etl/genders.csv')
    df_gender = df_gender.rename(columns={'TeamID' : 'TeamA'})
    df = pd.merge(df, df_gender, how='left', on='TeamA')
    df.fillna(0)
    return df


def build_training_set(elo_K=32):
    """Calculate all features for training dataset and save to file."""

    # Load mens tourney results
    df_men = pd.read_csv('./data/kaggle/MNCAATourneyCompactResults.csv')
    df_men['WinGap'] = df_men['WScore'] - df_men['LScore']
    df_men = df_men[['Season', 'WTeamID', 'LTeamID', 'WinGap']]

    # Load womens tourney results
    df_women = pd.read_csv('./data/kaggle/MNCAATourneyCompactResults.csv')
    df_women['WinGap'] = df_women['WScore'] - df_women['LScore']
    df_women = df_women[['Season', 'WTeamID', 'LTeamID', 'WinGap']]

    # Concat tourney results
    df = pd.concat([df_men, df_women], ignore_index=True)

    # Flip random rows so that winner isn't always first team
    num_rows = df.shape[0]
    np.random.seed(0)
    index_rand = [True if x == 1 else False for x in np.random.randint(0, 2, num_rows)]
    df['TeamA'] = df['WTeamID']
    df['TeamB'] = df['LTeamID']
    df.loc[index_rand, 'TeamA'] = df.loc[index_rand, 'LTeamID']
    df.loc[index_rand, 'TeamB'] = df.loc[index_rand, 'WTeamID']
    df.loc[index_rand, 'WinGap'] = 0 - df.loc[index_rand, 'WinGap']
    df = df[['Season', 'TeamA', 'TeamB', 'WinGap']]
    
    # Add binary win feature
    df['Win'] = df['WinGap'].apply(lambda x: 1 if x > 0 else 0)

    # Add training features
    df = add_gender(df)
    df = add_win_ratio(df)
    df = add_elo(df, K=elo_K)
    df = add_seeds(df)
    df = add_538_ratings(df)
    
    # Save training set
    df = df.sort_values('Season', ignore_index=True)
    df = df[df.Season >= 1985]
    df.to_csv('./data/etl/training_set.csv', index=False)


def prep_submission_frame():
    """Load in template prediction frame."""
    df_template = pd.read_csv('./data/kaggle/SampleSubmission2023.csv')

    # Split ID into values
    df_template['Season'] = df_template['ID'].apply(lambda x: x.split('_')[0])
    df_template['TeamA'] = df_template['ID'].apply(lambda x: x.split('_')[1])
    df_template['TeamB'] = df_template['ID'].apply(lambda x: x.split('_')[2])

    return df_template


def build_test_set(elo_K=32):
    """Calculate all features for test dataset and save to file."""

    # Load template for current season
    df = prep_submission_frame()

    # Add training features
    #df = add_gender(df)
    #df = add_win_ratio(df)
    #df = add_elo(df, K=elo_K)
    #df = add_seeds(df)
    #df = add_538_ratings(df)
    #df['RatingDiff'] = df.RatingA - df.RatingB

    # Save training set
    df.to_csv('./data/etl/test_set.csv', index=False)


if __name__ == "__main__":
    build_training_set()
    build_test_set()