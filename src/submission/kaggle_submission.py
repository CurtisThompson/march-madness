from kaggle.api.kaggle_api_extended import KaggleApi
from os.path import join
import pandas as pd


def submit(file_name='preds', file_path='./data/predictions/', comp='march-machine-learning-mania-2023',
           message="API Submission"):
    """Submit predictions to Kaggle competition."""

    # Authenticate through API
    api = KaggleApi()
    api.authenticate()

    # Get file path
    path = join(file_path, f'{file_name}.csv')

    # Submit to competition
    api.competition_submit(path, message, comp)


def get_leaderboard(comp='march-machine-learning-mania-2023'):
    """Get leaderboard of Kaggle competition."""

    # Authenticate through API
    api = KaggleApi()
    api.authenticate()

    # Get leaderboard
    leaderboard = api.competition_view_leaderboard(comp)
    leaderboard = pd.DataFrame(leaderboard['submissions'])

    return leaderboard


if __name__ == "__main__":
    submit()