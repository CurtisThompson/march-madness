import pandas as pd

def save_win_ratios():
    # Get all results into a single dataframe
    df_results = pd.concat([
        pd.read_csv('./data/kaggle/MRegularSeasonCompactResults.csv'),
        pd.read_csv('./data/kaggle/WRegularSeasonCompactResults.csv')
    ], ignore_index=True)

    # Get wins by each team per season
    df_wins = df_results.groupby(['Season', 'WTeamID']).count().reset_index()
    df_wins = df_wins[['Season', 'WTeamID', 'DayNum']]
    df_wins = df_wins.rename(columns={'WTeamID':'TeamID', 'DayNum':'Wins'})

    # Get losses by each team per season
    df_loss = df_results.groupby(['Season', 'LTeamID']).count().reset_index()
    df_loss = df_loss[['Season', 'LTeamID', 'DayNum']]
    df_loss = df_loss.rename(columns={'LTeamID':'TeamID', 'DayNum':'Losses'})

    # Get points scored
    df_wpts = df_results.groupby(['Season', 'WTeamID']).sum(numeric_only=True).reset_index()
    df_wpts = df_wpts[['Season', 'WTeamID', 'WScore']]
    df_wpts = df_wpts.rename(columns={'WTeamID':'TeamID', 'WScore':'PtsFor'})

    # Get points against
    df_lpts = df_results.groupby(['Season', 'LTeamID']).sum(numeric_only=True).reset_index()
    df_lpts = df_lpts[['Season', 'LTeamID', 'LScore']]
    df_lpts = df_lpts.rename(columns={'LTeamID':'TeamID', 'LScore':'PtsAgainst'})

    # Combine
    df_res = pd.merge(df_wins, df_loss, how='outer', on=['Season', 'TeamID'])
    df_res = pd.merge(df_res, df_wpts, how='outer', on=['Season', 'TeamID'])
    df_res = pd.merge(df_res, df_lpts, how='outer', on=['Season', 'TeamID'])
    df_res = df_res.fillna(0)

    # Calculate win ratio
    df_res['WinRatio'] = df_res['Wins'] / (df_res['Wins'] + df_res['Losses'])

    # Calculate points difference
    df_res['PtsDiff'] = df_res['PtsFor'] - df_res['PtsAgainst']
    df_res['PtsForRatio'] = df_res['PtsFor'] / (df_res['PtsFor'] + df_res['PtsAgainst'])
    df_res = df_res.fillna(0)

    # Correct types
    int_cols = ['Wins', 'Losses', 'PtsFor', 'PtsAgainst', 'PtsDiff']
    df_res[int_cols] = df_res[int_cols].astype(int)

    # Save features
    df_res.to_csv('./data/etl/win_ratios.csv', index=False)


if __name__ == "__main__":
    save_win_ratios()