import numpy as np
import pandas as pd


def add_win_ratio(df):
    """Merge win ratio and pts ratio into DataFrame."""
    win_ratios = pd.read_csv('./data/etl/win_ratios.csv')
    win_ratios = win_ratios[['Season', 'TeamID', 'WinRatio', 'PtsForRatio']]
    df = pd.merge(df, win_ratios, how='left', left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'])
    df = df.rename({'WinRatio':'WinRatioA', 'PtsForRatio':'PtsForRatioA'}, axis=1).drop('TeamID', axis=1)
    df = pd.merge(df, win_ratios, how='left', left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'])
    df = df.rename({'WinRatio':'WinRatioB', 'PtsForRatio':'PtsForRatioB'}, axis=1).drop('TeamID', axis=1)
    df[['WinRatioA', 'WinRatioB']] = df[['WinRatioA', 'WinRatioB']].fillna(0)
    df[['PtsForRatioA', 'PtsForRatioB']] = df[['PtsForRatioA', 'PtsForRatioB']].fillna(0)
    return df


def add_form(df):
    """Merge form values into DataFrame."""
    forms = pd.read_csv('./data/etl/team_form.csv')
    forms = forms[['Season', 'TeamID', 'FormHarmonic', 'FormUniform']]
    df = pd.merge(df, forms, how='left', left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'])
    df = df.rename({'FormHarmonic':'FormHarmonicA', 'FormUniform':'FormUniformA'}, axis=1).drop('TeamID', axis=1)
    df = pd.merge(df, forms, how='left', left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'])
    df = df.rename({'FormHarmonic':'FormHarmonicB', 'FormUniform':'FormUniformB'}, axis=1).drop('TeamID', axis=1)
    df[['FormHarmonicA', 'FormHarmonicB']] = df[['FormHarmonicA', 'FormHarmonicB']].fillna(0.5)
    df[['FormUniformA', 'FormUniformB']] = df[['FormUniformA', 'FormUniformB']].fillna(0.5)
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


def add_tournament_round(df):
    """Merge tournament rounds into DataFrame."""
    ratings = pd.read_csv('./data/etl/tournament_rounds.csv')
    df = pd.merge(df,
                  ratings,
                  how='left',
                  left_on=['Season', 'TeamA', 'TeamB'],
                  right_on=['Season', 'TeamA', 'TeamB'])
    df['Round'] = df['Round'].fillna(0)
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


def add_clutch(df):
    """Merge clutch game features with DataFrame."""
    df_clutch = pd.read_csv('./data/etl/clutch_games.csv')
    df = pd.merge(df, df_clutch, how='left', left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'])
    df = df.rename({'ClutchRatio':'ClutchRatioA'}, axis=1).drop(['TeamID', 'ClutchWins', 'ClutchLosses'], axis=1)
    df = pd.merge(df, df_clutch, how='left', left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'])
    df = df.rename({'ClutchRatio':'ClutchRatioB'}, axis=1).drop(['TeamID', 'ClutchWins', 'ClutchLosses'], axis=1)
    df[['ClutchRatioA', 'ClutchRatioB']] = df[['ClutchRatioA', 'ClutchRatioB']].fillna(0.5)
    return df


def build_training_set(elo_K=32, start_year=1985):
    """Calculate all features for training dataset and save to file."""

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
    
    # Save training set
    df = df.sort_values('Season', ignore_index=True)
    df = df[df.Season >= start_year]
    df.to_csv('./data/etl/training_set.csv', index=False)


def prep_submission_frame():
    """Load in template prediction frame."""
    df_template = pd.read_csv('./data/kaggle/SampleSubmission2023.csv')

    # Split ID into values
    df_template['Season'] = df_template['ID'].apply(lambda x: x.split('_')[0]).astype(int)
    df_template['TeamA'] = df_template['ID'].apply(lambda x: x.split('_')[1]).astype(int)
    df_template['TeamB'] = df_template['ID'].apply(lambda x: x.split('_')[2]).astype(int)

    return df_template


def build_test_set(elo_K=32):
    """Calculate all features for test dataset and save to file."""

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

    # Save training set
    df.to_csv('./data/etl/test_set.csv', index=False)


if __name__ == "__main__":
    build_training_set()
    build_test_set()