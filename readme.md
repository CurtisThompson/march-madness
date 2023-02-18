# March Madness 2023

This repository is for predicting the outcomes of each match in the 2023 March Madness.

## Set-Up

To run the code in this repository, you will need to create the conda environment by running the following command:

```conda env create --name march-madness-2023 --file ./env/environment.yaml```

If you have already created the environment and need to update it, run the following command:

```conda env update --name march-madness-2023 --file ./env/environment.yaml```

Then, you will need to download API keys for Kaggle so that you can ingest some datasets. Follow the [instructions on the Kaggle website](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication).