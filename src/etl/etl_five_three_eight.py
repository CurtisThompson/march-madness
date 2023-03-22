import pandas as pd


def find_538_ratings(start_year=2016, end_year=2022):
    """
    Create list of 538 rankings that can be joined to other features. Save to
    538_ratings.csv.
    
    Args:
        start_year: The first year of 538 ratings to include. Default is 2016.
        end_year: The final year of 538 ratings to include. Default is 2022.
    """

    # Get day after selection sunday (rankings should be set for tournament)
    df_season = pd.read_csv('./data/kaggle/MSeasons.csv')
    df_season['SelectionSunday'] = pd.to_datetime(df_season.DayZero) + pd.DateOffset(days=133)
    df_season = df_season[df_season.Season >= start_year]
    df_season = df_season.set_index('Season')
    df_season = df_season['SelectionSunday'].astype(str).to_dict()

    # Load in 538 rankings
    df_538 = []
    for i in range(start_year, end_year+1):
        df_year = pd.read_csv(f'./data/five_three_eight_rankings/{str(i)}.csv')
        df_year['Season'] = i
        df_year = df_year[df_year.forecast_date <= df_season[i]]
        df_year = df_year.sort_values('forecast_date')
        df_year = df_year.drop_duplicates(['team_id', 'forecast_date', 'gender'], keep='last')
        df_538.append(df_year)
    df_538 = pd.concat(df_538, ignore_index=True)

    # Reformat columns and split into mens and womens
    df_538['Rating'] = df_538['team_rating']
    df_538['team_name'] = df_538.team_name.str.lower()
    df_538_m = df_538[df_538.gender == 'mens']
    df_538_w = df_538[df_538.gender == 'womens']

    # Men: Find team ID for each team in ranking
    df_spellings_m = pd.read_csv('./data/kaggle/MTeamSpellings.csv', encoding='mbcs')
    df_m = pd.merge(df_538_m,
                    df_spellings_m,
                    how='left',
                    left_on='team_name',
                    right_on='TeamNameSpelling')
    df_m = df_m[['Season', 'TeamID', 'Rating']]
    df_m.sort_values('Rating', inplace=True)

    # Women: Find team ID for each team in ranking
    df_spellings_w = pd.read_csv('./data/kaggle/WTeamSpellings.csv', encoding='mbcs')
    df_w = pd.merge(df_538_w,
                    df_spellings_w,
                    how='left',
                    left_on='team_name',
                    right_on='TeamNameSpelling')
    df_w = df_w[['Season', 'TeamID', 'Rating']]
    df_w.sort_values('Rating', inplace=True)

    # Recombine men and women, and save to file
    df = pd.concat([df_m, df_w], ignore_index=True)
    df.to_csv('./data/etl/538_ratings.csv', index=False)


if __name__ == "__main__":
    find_538_ratings()