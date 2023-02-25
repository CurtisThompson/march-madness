import pandas as pd

OUTPUT_PATH = './data/five_three_eight_rankings/'

def five_three_eight_api_url(year):
    """Get URL for 538 yearly ratings."""
    return 'https://projects.fivethirtyeight.com/march-madness-api/' + str(year) + '/fivethirtyeight_ncaa_forecasts.csv'


def download(start_year=2016, end_year=2022):
    """Download 538 college rankings."""

    for year in range(start_year, end_year+1):
        try:
            url = five_three_eight_api_url(year)
            df = pd.read_csv(url)
            df.to_csv(OUTPUT_PATH+str(year)+'.csv', index=False)
        except:
            print(f'Failed to download 538 data for the year {year}')


if __name__ == "__main__":
    download()