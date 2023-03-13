from itertools import permutations

import pandas as pd


def route_to_final(df_slots, season, seed):
    """Returns a seeds route to the final as a list of game slots."""
    route = []
    check_slot = seed

    while len(route) <= 8:
        slots = df_slots.loc[(df_slots.Season == season) &
                            ((df_slots.StrongSeed == check_slot) |
                            (df_slots.WeakSeed == check_slot)), 'Slot']
        if len(slots) > 0:
            slot = slots.iloc[0]
            route.append(slot)
            check_slot = slot
        else:
            break

    return route


def find_seed_match_up_round(df_slots, season, seed_a, seed_b):
    """Find the earliest possible round where two seeds could meet."""
    # If first four, then return 0
    if seed_a[:3] == seed_b[:3]:
        return 0
    
    # Find route to final for both teams
    route_a = route_to_final(df_slots, season, seed_a)
    route_b = route_to_final(df_slots, season, seed_b)

    # Find all games which both teams could play in
    matching_rounds = set(route_a) & set(route_b)
    matching_rounds = sorted(list(matching_rounds))

    # Find the earliest round where both teams could play
    if len(matching_rounds) > 0:
        return matching_rounds[0][1]
    else:
        return 0


def find_team_seed(df_seeds, season, team):
    """Looks up the seed of a given team in a season."""
    try:
        seed = df_seeds.loc[(df_seeds.Season == season) & (df_seeds.TeamID == team), 'Seed'].iloc[0]
    except:
        seed = ''
    return seed


def create_team_matchups_from_seedings(df_seeds, season):
    """Lists every two-team seed combination."""
    teams = df_seeds.loc[df_seeds.Season == season, 'TeamID'].values
    match_ups = permutations(teams, 2)
    df = pd.DataFrame(match_ups, columns=['TeamA', 'TeamB'])
    df['Season'] = season
    return df


def find_all_team_matchups_single_gender(df_slots, df_seeds, df_games, current_year=2023):
    """
    Creates a look up table of all March Madness team match ups and the
    round they took place in. Only run for a single gender.
    """

    # Find matchups for current year, assuming no existing results
    current_matchups = create_team_matchups_from_seedings(df_seeds, current_year)

    # Create dataframe of all team match ups
    df_games = df_games[['Season', 'WTeamID', 'LTeamID']]
    df_games = df_games.rename(columns={'WTeamID':'TeamA', 'LTeamID':'TeamB'})
    df_games_2 = df_games.copy()
    df_games_2[['TeamA', 'TeamB']] = df_games[['TeamB', 'TeamA']]
    df_games = pd.concat([df_games, df_games_2, current_matchups], ignore_index=True)

    # Add seeds to each team
    df_games['SeedA'] = df_games.apply(lambda x: find_team_seed(df_seeds, x.Season, x.TeamA), axis=1)
    df_games['SeedB'] = df_games.apply(lambda x: find_team_seed(df_seeds, x.Season, x.TeamB), axis=1)

    # Find round
    df_games['Round'] = df_games.apply(lambda x: find_seed_match_up_round(df_slots, x.Season, x.SeedA, x.SeedB), axis=1)
    df_games = df_games[['Season', 'TeamA', 'TeamB', 'Round']]

    # Save
    return df_games


def find_all_team_matchups(current_year=2023):
    """
    Creates a look up table of all March Madness team match ups and the
    round they took place in.
    """

    # Mens match ups
    df_slots_m = pd.read_csv('./data/kaggle/MNCAATourneySlots.csv')
    df_seeds_m = pd.read_csv('./data/kaggle/MNCAATourneySeeds.csv')
    df_games_m = pd.read_csv('./data/kaggle/MNCAATourneyCompactResults.csv')
    m_matchups = find_all_team_matchups_single_gender(df_slots_m,
                                                      df_seeds_m,
                                                      df_games_m,
                                                      current_year=current_year)
    
    # Womens match ups
    df_slots_w = pd.read_csv('./data/kaggle/WNCAATourneySlots.csv')
    df_seeds_w = pd.read_csv('./data/kaggle/WNCAATourneySeeds.csv')
    df_games_w = pd.read_csv('./data/kaggle/WNCAATourneyCompactResults.csv')
    w_matchups = find_all_team_matchups_single_gender(df_slots_w,
                                                      df_seeds_w,
                                                      df_games_w,
                                                      current_year=current_year)
    
    # Combine and save
    df = pd.concat([m_matchups, w_matchups], ignore_index=True)
    df = df.drop_duplicates(ignore_index=True)
    df.to_csv('./data/etl/tournament_rounds.csv', index=False)


if __name__ == "__main__":
    find_all_team_matchups()