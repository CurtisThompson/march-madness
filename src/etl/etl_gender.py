import pandas as pd


def find_gender():
    """Create a table of each team ID with their gender. 1 for women, 0 for men."""

    # Load womens teams
    df_w = pd.read_csv('./data/kaggle/WTeams.csv')
    df_w['Gender'] = 1
    df_w = df_w[['TeamID', 'Gender']]

    # Load mens teams
    df_m = pd.read_csv('./data/kaggle/MTeams.csv')
    df_m['Gender'] = 0
    df_m = df_m[['TeamID', 'Gender']]

    # Concatenate and save
    df = pd.concat([df_w, df_m], ignore_index=True).sort_values('TeamID')
    df.to_csv('./data/etl/genders.csv', index=False)


if __name__ == "__main__":
    find_gender()