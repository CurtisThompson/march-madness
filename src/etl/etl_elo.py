import pandas as pd

def calculate_elo(K=32):
    """Calculate season Elo for all teams."""

    # Import results data
    df = pd.read_csv('./data/kaggle/MRegularSeasonCompactResults.csv')
    df_2 = pd.read_csv('./data/kaggle/MNCAATourneyCompactResults.csv')
    df_3 = pd.read_csv('./data/kaggle/WRegularSeasonCompactResults.csv')
    df_4 = pd.read_csv('./data/kaggle/WNCAATourneyCompactResults.csv')
    df = pd.concat([df, df_2, df_3, df_4])
    df = df.sort_values(['Season', 'DayNum'], ascending=True)

    # Get list of teams
    teams = list(set(list(df.WTeamID.unique()) + list(df.LTeamID.unique())))

    # Initialise Elo
    df_elo = pd.DataFrame(teams, columns=['TeamID'])
    df_elo['Elo'] = 1500
    df_all_elos = []

    for season in df.Season.unique():
        print(season)
        saved_year = False
        df_year = df.loc[df.Season == season]

        for index, game in df_year.iterrows():
            # Save if reached Selection Sunday
            if (game.DayNum >= 134) and (not saved_year):
                df_save = df_elo.copy()
                df_save['Season'] = season
                df_all_elos.append(df_save)
                saved_year = True

            # Get team IDs
            teamA = game.WTeamID
            teamB = game.LTeamID

            # Get ELO stats
            eloA = df_elo.loc[df_elo.TeamID == teamA, 'Elo'].values[0]
            eloB = df_elo.loc[df_elo.TeamID == teamB, 'Elo'].values[0]

            # Calc ELO changes
            rA = 10 ** (eloA / 400)
            rB = 10 ** (eloB / 400)
            eA = rA / (rA + rB)
            eB = rB / (rA + rB)
            #sA = 1
            #sB = 0
            newEloA = eloA + (K * (1 - eA))
            newEloB = eloB + (K * (0 - eB))

            # Update ELO frame
            df_elo.loc[df_elo.TeamID == teamA, 'Elo'] = newEloA
            df_elo.loc[df_elo.TeamID == teamB, 'Elo'] = newEloB
        
        # Save Elo if not yet saved for this season
        if not saved_year:
            df_save = df_elo.copy()
            df_save['Season'] = season
            df_all_elos.append(df_save)
            saved_year = True

    # Save Elos to file
    df_all_elos = pd.concat(df_all_elos, ignore_index=True)
    df_all_elos.to_csv(f'./data/etl/elo{"_"+str(K) if K != 32 else ""}.csv', index=False)


if __name__ == "__main__":
    calculate_elo()