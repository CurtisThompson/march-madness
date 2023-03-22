import pandas as pd
import os


def calculate_massey():
    """
    Calculate the median, mean, and standard deviation of Massey rankings for
    each team per season. Save to massey.csv.
    """

    # Read in and combine all Massey files
    massey_files = [f for f in os.listdir('./data/kaggle/')
                    if f.startswith('MMassey') or f.startswith('WMassey')]
    massey_dfs = [pd.read_csv(f'./data/kaggle/{f}') for f in massey_files]
    df = pd.concat(massey_dfs, ignore_index=True)

    # Keep only closest rankings to March Madness for each season
    df = df.loc[(df.RankingDayNum >= 100) & (df.RankingDayNum <= 134)]
    df = df.sort_values(by='RankingDayNum', ascending=True)
    df = df.drop_duplicates(subset=['Season', 'SystemName', 'TeamID'],
                            keep='last')
    df = df.sort_values(by=['Season', 'RankingDayNum', 'SystemName', 'TeamID'],
                        ignore_index=True)

    # Normalise ranking: 0 is worst, 1 is best
    df['NormalisedRank'] = df.OrdinalRank / df.groupby(['Season', 'SystemName']).OrdinalRank.transform('max')
    df['NormalisedRank'] = 1 - df.NormalisedRank
    df['NormalisedRank'] = df.NormalisedRank.clip(0, 1)

    # Calculate averages and spreads
    df = df[['Season', 'TeamID', 'NormalisedRank']].groupby(['Season', 'TeamID'])
    df = df.agg(['median', 'mean', 'std'])['NormalisedRank']
    df = df.reset_index().rename(columns={'median':'MasseyMedian',
                                          'mean':'MasseyMean',
                                          'std':'MasseyStd'})

    # Save to file
    df.to_csv('./data/etl/massey.csv', index=False)


if __name__ == "__main__":
    calculate_massey()