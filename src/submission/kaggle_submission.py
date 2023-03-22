from kaggle.api.kaggle_api_extended import KaggleApi
from os.path import join
import pandas as pd


def submit(file_name='preds', file_path='./data/predictions/',
           comp='march-machine-learning-mania-2023', message='API Submission',
           include_metric=False, metric=None):
    """
    Submit predictions to Kaggle competition.
    
    Args:
        file_name: The name of the file where predictions are stored. Do not
            include the file format at the end, .csv is assumed.
        file_path: The file path for the prediction file.
        comp: The Kaggle competition to submit predictions to.
        message: An optional submission message sent to Kaggle, does not affect
            the predictions.
        include_metric: If True then includes the validation metric in the
            submission metric. Default is False.
        metric: The validation metric. Required if include_metric is True.
    """

    # Authenticate through API
    api = KaggleApi()
    api.authenticate()

    # Get file path
    path = join(file_path, f'{file_name}.csv')

    # Append metric to message if needed
    if include_metric and metric != None:
        message = str(message) + ' | Validation: ' + str(metric)

    # Submit to competition
    api.competition_submit(path, message, comp)


def get_leaderboard(comp='march-machine-learning-mania-2023'):
    """
    Get leaderboard of Kaggle competition.
    
    Args:
        comp: The Kaggle competition to retrieve the public leaderboard for.
    
    Returns:
        A Pandas DataFrame of the current leaderboard.
    """

    # Authenticate through API
    api = KaggleApi()
    api.authenticate()

    # Get leaderboard
    leaderboard = api.competition_view_leaderboard(comp)
    leaderboard = pd.DataFrame(leaderboard['submissions'])

    return leaderboard


if __name__ == "__main__":
    submit()