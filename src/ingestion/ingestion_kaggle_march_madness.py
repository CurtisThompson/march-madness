from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os


def download():
    """Downloads competition files from March Machine Learning Mania 2023 on Kaggle."""
    d_path = './data/kaggle/'
    comp = 'march-machine-learning-mania-2023'

    # Authenticate through API
    api = KaggleApi()
    api.authenticate()

    # Download march madness 2023 competition files
    api.competition_download_files(comp, path=d_path)

    # Unzip competition files
    with zipfile.ZipFile(d_path+comp+'.zip', 'r') as zip_ref:
        zip_ref.extractall(d_path)
    
    # Remove original zip
    os.remove(d_path+comp+'.zip')


if __name__ == "__main__":
    download()