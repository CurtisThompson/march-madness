import pandas as pd


def calculate_clutch_win_ratio(max_score_gap=3):
    """Calculate the number of clutch games per team and the clutch win ratio."""

    # Get all results into a single dataframe
    df_results = pd.concat([
        pd.read_csv('./data/kaggle/MRegularSeasonCompactResults.csv'),
        pd.read_csv('./data/kaggle/WRegularSeasonCompactResults.csv')
    ], ignore_index=True)

    # Calculate scoring gap
    df_results['ScoreGap'] = df_results['WScore'] - df_results['LScore']

    # Clutch filters
    filter_overtime = df_results['NumOT'] >= 1
    filter_close = df_results['ScoreGap'] <= max_score_gap

    # Apply filters
    df_results = df_results.loc[filter_overtime | filter_close]

    # Get clutch wins
    df_wins = df_results.groupby(['Season', 'WTeamID']).count().reset_index()
    df_wins = df_wins[['Season', 'WTeamID', 'DayNum']]
    df_wins = df_wins.rename(columns={'WTeamID':'TeamID', 'DayNum':'ClutchWins'})

    # Get clutch losses
    df_loss = df_results.groupby(['Season', 'LTeamID']).count().reset_index()
    df_loss = df_loss[['Season', 'LTeamID', 'DayNum']]
    df_loss = df_loss.rename(columns={'LTeamID':'TeamID', 'DayNum':'ClutchLosses'})

    # Combine wins and losses
    df_results = pd.merge(df_wins, df_loss, how='outer', on=['Season', 'TeamID'])
    df_results[['ClutchWins', 'ClutchLosses']] = df_results[['ClutchWins', 'ClutchLosses']].fillna(0)

    # Calculate clutch ratio
    df_results['ClutchRatio'] = df_results['ClutchWins'] / (df_results['ClutchWins'] + df_results['ClutchLosses'])
    df_results['ClutchRatio'] = df_results['ClutchRatio'].fillna(1)
    
    # Save to file
    df_results.to_csv('./data/etl/clutch_games.csv', index=False)


if __name__ == "__main__":
    calculate_clutch_win_ratio()
