import os

from src.ingestion.ingestion_kaggle_march_madness import download as download_kaggle
from src.ingestion.ingestion_five_three_eight import download as download_538

from src.etl.etl_win_ratios import save_win_ratios
from src.etl.add_features import build_training_set, build_test_set

from src.model.evaluate import validate_and_build_model
from src.model.predict import run as predict_current_year


if __name__ == "__main__":
    # Build data directory structure
    folders = [
        './data/etl/',
        './data/five_three_eight_rankings/',
        './data/kaggle/',
        './data/models/',
        './data/predictions/'
    ]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Ingestion
    download_kaggle()
    download_538()

    # ETL
    save_win_ratios()

    # Build Datasets
    build_training_set()
    build_test_set()

    # Build Models
    validate_and_build_model()

    # Predictions
    predict_current_year()