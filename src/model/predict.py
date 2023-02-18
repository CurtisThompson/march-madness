import pandas as pd


PRED_COLS = ['ID', 'Pred']


def prep_submission_frame():
    """Load in template prediction frame."""
    df_template = pd.read_csv('./data/kaggle/SampleSubmission2023.csv')

    # Split ID into values
    df_template['Year'] = df_template['ID'].apply(lambda x: x.split('_')[0])
    df_template['TeamA'] = df_template['ID'].apply(lambda x: x.split('_')[1])
    df_template['TeamB'] = df_template['ID'].apply(lambda x: x.split('_')[2])

    return df_template


def predict_row(row):
    """Predict the outcome of a single game."""
    return 0.2


def predict_frame(df):
    """Predict the outcome of multiple games."""
    df['Pred'] = df.apply(predict_row, axis=1)
    return df


def save_predictions(df, file_name="preds", file_path='./data/predictions/'):
    """Save all game predictions to file."""
    df = df[PRED_COLS]
    df.to_csv(file_path + file_name + '.csv', index=False)


def run():
    """Load in the prediction template, make predictions, and save to file."""
    df = prep_submission_frame()
    df = predict_frame(df)
    save_predictions(df)


if __name__ == "__main__":
    run()