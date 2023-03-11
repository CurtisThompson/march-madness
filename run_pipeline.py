import os

from src.ingestion.ingestion_kaggle_march_madness import download as download_kaggle
from src.ingestion.ingestion_five_three_eight import download as download_538

from src.etl.etl_win_ratios import save_win_ratios
from src.etl.etl_five_three_eight import find_538_ratings
from src.etl.etl_seeds import reformat_seeds
from src.etl.etl_elo import calculate_elo
from src.etl.add_features import build_training_set, build_test_set
from src.etl.etl_gender import find_gender
from src.etl.etl_clutch_games import calculate_clutch_win_ratio
from src.etl.etl_form import get_teams_form

from src.model.evaluate import validate_and_build_model
from src.model.predict import run as predict_current_year

from src.submission.kaggle_submission import submit as competition_submit


if __name__ == "__main__":
    # Build data directory structure
    folders = [
        './data/etl/',
        './data/five_three_eight_rankings/',
        './data/kaggle/',
        './data/models/',
        './data/predictions/',
        './data/explain/'
        './viz/visuals/',
    ]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Ingestion
    #download_kaggle()
    #download_538(start_year=2016, end_year=2022)

    # ETL
    find_gender()
    save_win_ratios()
    calculate_clutch_win_ratio()
    find_538_ratings(start_year=2016, end_year=2022)
    reformat_seeds()
    get_teams_form()
    #calculate_elo(K=32)

    # Build Datasets
    build_training_set(elo_K=32)
    build_test_set(elo_K=32)

    # Build Models
    validate_and_build_model()

    # Predictions
    predict_current_year()

    # Submit Predictions To Competition
    #competition_submit(message='Test API Submission')