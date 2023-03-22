import pandas as pd


OUTPUT_PATH = './data/five_three_eight_rankings/'


def five_three_eight_api_url(year):
    """
    Get URL for 538 yearly ratings.
    
    Args:
        year: March Madness year to include in URL. Integer.
    
    Returns:
        String containing URL to retrieve ratings.
    """
    return 'https://projects.fivethirtyeight.com/march-madness-api/' + str(year) + '/fivethirtyeight_ncaa_forecasts.csv'


def download(start_year=2016, end_year=2022):
    """
    Download 538 college rankings, and store to file.
    
    Args:
        start_year: The first year to retrieve 538 ratings for. Default is 2016.
        end_year: The final year to retrieve 538 ratings for. Default is 2022.
    """

    for year in range(start_year, end_year+1):
        try:
            url = five_three_eight_api_url(year)
            df = pd.read_csv(url)
            df.to_csv(OUTPUT_PATH+str(year)+'.csv', index=False)
        except:
            print(f'Failed to download 538 data for the year {year}')


if __name__ == "__main__":
    download()