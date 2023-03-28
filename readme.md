# March Madness 2023

This project aims to predict the outcome of every game in the March Madness basketball tournament.

As well as just predicting each game using machine learning, a complete pipeline is built which ingests open-source data, cleans and processes the data, builds a machine learning model, and evaluates the model. This entire repository is self-contained; it is possible to run and test the whole pipeline by cloning the repo.

## March Madness
March Madness is the annual Division I college basketball tournament in the United States of America. After a qualification round known as the First Four which reduces eight low-seeded teams to four, 64 college basketball teams compete in a straight knockout tournament with the winning team declared the national champions. The men's competition has been played almost yearly since 1939 with an equivalent women's competition introduced in 1982.

March Madness is one of the largest annual sporting events in America and tens of millions of Americans take part in bracket pool contests, attempting to predict the outcome of every game in the tournament.

## Set Up

1. Create the Python environment by running the following command:
```
conda env create --name march-madness-2023 --file ./env/environment.yaml
```

2. Download API keys for Kaggle (required to ingest some datasets) . Follow the [instructions on the Kaggle website](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication).

3. Run the ```./run_pipeline.py``` Python file.  
This script will build your data directory structure, download and process all datasets, and build a complete model.

See the [Project Set Up page](https://github.com/CurtisThompson/march-madness/wiki/Project-Set-Up) on the Wiki for more details.

## Pipeline

The pipeline ingests data from two sources; Kaggle and Five Thirty Eight. This data is then processed and converted into a training and test set. The training set is used to train multiple machine learning models, and then current tournament predictions are made with the test set. These predictions can be submitted back to a Kaggle competition or viewed on a local dashboard.

The basic flow of the pipeline is given below.

![Flow_2023](https://user-images.githubusercontent.com/16989865/228359984-42a90142-7c5a-43be-96fc-522965cbdef2.png)

See the [Pipeline Processes page](https://github.com/CurtisThompson/march-madness/wiki/Process) on the Wiki for more details.

## Dashboard

After running the main pipeline, it is possible to run a Dash dashboard on a local server (i.e. localhost). This is achievable with the run_component/run_server parameter in the config file.

The dashboard displays model predictions to the user alongside key feature values and SHAP values. See the image below for a basic example.

![dashboard](https://user-images.githubusercontent.com/16989865/228363926-4197ce63-de7b-46ad-a677-0897c800119d.png)

## Wiki

[The Wiki](https://github.com/CurtisThompson/march-madness/wiki) is the main source of information on this project. It has pages that explain each main source code module, each main data science technique used, and background information. Please check out the Wiki first if unsure about any part of the project.
