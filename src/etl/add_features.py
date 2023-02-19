import numpy as np
import pandas as pd


def add_win_ratio(season, teama, teamb, win_ratios):
    try:
        teamawins = win_ratios.loc[(win_ratios.Season == season) & (win_ratios.TeamID == teama)].iloc[0]
    except:
        teamawins = {'WinRatio':0.5, 'PtsForRatio':0}
        print('Win ratio not found for team ', str(teama))
    
    try:
        teambwins = win_ratios.loc[(win_ratios.Season == season) & (win_ratios.TeamID == teamb)].iloc[0]
    except:
        teambwins = {'WinRatio':0.5, 'PtsForRatio':0}
        print('Win ratio not found for team ', str(teamb))

    features = [teamawins['WinRatio'], teamawins['PtsForRatio'], teambwins['WinRatio'], teambwins['PtsForRatio']]
    return features


def add_538_ratings(df):
    """Merge 538 ratings into DataFrame."""
    ratings = pd.read_csv('./data/etl/538_ratings.csv')
    df = pd.merge(df, ratings, how='left', left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'])
    df = df.rename({'Rating':'RatingA'}, axis=1).drop('TeamID', axis=1)
    df = pd.merge(df, ratings, how='left', left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'])
    df = df.rename({'Rating':'RatingB'}, axis=1).drop('TeamID', axis=1)
    df[['RatingA', 'RatingB']] = df[['RatingA', 'RatingB']].fillna(60)
    return df


def build_training_set():
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
    win_ratios = pd.read_csv('./data/etl/win_ratios.csv')
    df[['WinRatioA', 'PtsForRatioA', 'WinRatioB', 'PtsForRatioB']] = df.apply(lambda x: add_win_ratio(x.Season,
                                                                                                      x.TeamA,
                                                                                                      x.TeamB,
                                                                                                      win_ratios),
                                                                              axis=1,
                                                                              result_type='expand')
    df = add_538_ratings(df)
    df['RatingDiff'] = df.RatingA = df.RatingB
    
    # Save training set
    df = df.sort_values('Season', ignore_index=True)
    df = df[df.Season >= 2016]
    df.to_csv('./data/etl/training_set.csv', index=False)


def prep_submission_frame():
    """Load in template prediction frame."""
    df_template = pd.read_csv('./data/kaggle/SampleSubmission2023.csv')

    # Split ID into values
    df_template['Season'] = df_template['ID'].apply(lambda x: x.split('_')[0])
    df_template['TeamA'] = df_template['ID'].apply(lambda x: x.split('_')[1])
    df_template['TeamB'] = df_template['ID'].apply(lambda x: x.split('_')[2])

    return df_template


def build_test_set():
    # Load template for current season
    df = prep_submission_frame()

    # Add training features
    #win_ratios = pd.read_csv('./data/etl/win_ratios.csv')
    #df[['WinRatioA', 'PtsForRatioA', 'WinRatioB', 'PtsForRatioB']] = df.apply(lambda x: add_win_ratio(x.Season,
    #                                                                                                  x.TeamA,
    #                                                                                                  x.TeamB,
    #                                                                                                  win_ratios),
    #                                                                          axis=1,
    #                                                                          result_type='expand')                                                                         result_type='expand')
    
    # Save training set
    df.to_csv('./data/etl/test_set.csv', index=False)


if __name__ == "__main__":
    build_training_set()
    build_test_set()