from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os


def download(comp='march-machine-learning-mania-2023'):
    """
    Downloads competition files from Kaggle, and saves to file.
    
    Args:
        comp: Name of the Kaggle competition to download files for. String.
    """
    
    d_path = './data/kaggle/'

    # Authenticate through API
    api = KaggleApi()
    api.authenticate()

    # Download competition files
    api.competition_download_files(comp, path=d_path)

    # Unzip competition files
    with zipfile.ZipFile(d_path+comp+'.zip', 'r') as zip_ref:
        zip_ref.extractall(d_path)
    
    # Remove original zip
    os.remove(d_path+comp+'.zip')


if __name__ == "__main__":
    download()