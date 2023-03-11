import os

from config.config import get_config

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


CONFIG = get_config()


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
    #download_kaggle(comp=CONFIG['kaggle']['competition'])
    #download_538(start_year=CONFIG['etl']['538_start_year'], end_year=CONFIG['etl']['538_end_year'])

    # ETL
    find_gender()
    save_win_ratios()
    calculate_clutch_win_ratio(max_score_gap=CONFIG['etl']['clutch_score_gap_max'])
    find_538_ratings(start_year=CONFIG['etl']['538_start_year'], end_year=CONFIG['etl']['538_end_year'])
    reformat_seeds()
    get_teams_form()
    #calculate_elo(K=CONFIG['etl']['elo_k_factor'])

    # Build Datasets
    build_training_set(elo_K=CONFIG['etl']['elo_k_factor'], start_year=CONFIG['etl']['training_start_year'])
    build_test_set(elo_K=CONFIG['etl']['elo_k_factor'])

    # Build Models
    validate_and_build_model(training_columns_men=CONFIG['model']['mens_columns'],
                             training_columns_women=CONFIG['model']['womens_columns'],
                             tune=CONFIG['model']['is_tune'],
                             random_state=CONFIG['random_state'],
                             optimisation_iterations=CONFIG['model']['optimisation_iterations'],
                             optimisation_initial_points=CONFIG['model']['optimisation_initial'])

    # Predictions
    predict_current_year(model_columns_men=CONFIG['model']['mens_columns'],
                         model_columns_women=CONFIG['model']['womens_columns'])

    # Submit Predictions To Competition
    #competition_submit(comp=CONFIG['kaggle']['competition'], message=CONFIG['kaggle']['submit_message'])