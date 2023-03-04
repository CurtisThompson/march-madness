# March Madness 2023

This project aims to predict the outcome of every game in the March Madness basketball tournament.

As well as just predicting each game using machine learning, a complete pipeline is built which ingests open-source data, cleans and processes the data, builds a machine learning model, and evaluates the model. This entire repository is self-contained; it is possible to run and test the whole pipeline by cloning the repo.

## March Madness
March Madness is the annual Division I college basketball tournament in the United States of America. After a qualification round known as the First Four which reduces eight low-seeded teams to four, 64 college basketball teams compete in a straight knockout tournament with the winning team declared the national champions. The men's competition has been played almost yearly since 1939 with an equivalent women's competition introduced in 1982.

March Madness is one of the largest annual sporting events in America and tens of millions of Americans take part in bracket pool contests, attempting to predict the outcome of every game in the tournament.

## Set-Up

1. Create the Python environment by running the following command:  
```conda env create --name march-madness-2023 --file ./env/environment.yaml```  
If you have already created the environment and need to update it, run the following command:  
```conda env update --name march-madness-2023 --file ./env/environment.yaml```  

> :heavy_exclamation_mark: **If conda takes a long time to solve the environment: This is an occassional issue with conda. If conda takes more than 15 minutes to solve the environment then it is recommended to update the conda solver by following the instructions from the [Anaconda blog](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community). **

2. Download API keys for Kaggle (required to ingest some datasets) . Follow the [instructions on the Kaggle website](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication).

3. Run the ```./run_pipeline.py``` Python file.  
This script will build your data directory structure, download and process all datasets, and build a complete model.
