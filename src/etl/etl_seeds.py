import pandas as pd


def reformat_seeds():
    """
    Reformat seed file so that seeds are integers. Save to seeds.csv.
    """

    df = pd.read_csv('./data/kaggle/MNCAATourneySeeds.csv')
    df = df.rename(columns={'Seed':'FullSeed'})
    df['Seed'] = df['FullSeed'].apply(lambda x: x[1:3]).astype(int)
    df = df[['Season', 'TeamID', 'Seed']]
    df.to_csv('./data/etl/seeds.csv', index=False)


if __name__ == "__main__":
    reformat_seeds()