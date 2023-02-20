import pandas as pd


def reformat_seeds():
    """Reformats seed file so that seeds are integers."""
    df = pd.read_csv('./data/kaggle/MNCAATourneySeeds.csv')
    df = df.rename(columns={'Seed':'FullSeed'})
    df['Seed'] = df['FullSeed'].apply(lambda x: x[1:3]).astype(int)
    df = df[['Season', 'TeamID', 'Seed']]
    df.to_csv('./data/etl/seeds.csv', index=False)


if __name__ == "__main__":
    reformat_seeds()