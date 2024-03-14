import pandas as pd
import numpy as np


# https://www.kaggle.com/code/samdaltonjr/preliminary-preds-into-bracket



def get_seed_matchup(df, seed_a, seed_b, tournament):
    vals = df.loc[((df.SeedA==seed_a) & (df.SeedB==seed_b) & (df.Tournament == tournament)), 'Pred'].values
    vals_b = 1 - df.loc[((df.SeedA==seed_b) & (df.SeedB==seed_a) & (df.Tournament == tournament)), 'Pred'].values
    return (list(vals) + list(vals_b))[0]


def get_winner(df, seed_a, seed_b, tournament, win_style='Best'):
    # Find win probability
    score = get_seed_matchup(df, seed_a, seed_b, tournament)

    # If 'Best' style, return most likely
    if win_style == 'Best':
        if score > 0.5:
            return seed_a
        return seed_b
    # If 'Prob' style, return based on probability
    if win_style == 'Prob':
        return np.random.choice([seed_a, seed_b], p=[score, 1-score])
    # If no style, use 'Prob'
    return np.random.choice([seed_a, seed_b], p=[score, 1-score])


def find_slot_winner(df, slot, tournament):
    return df.loc[(df.Slot == slot) & (df.Tournament == tournament), 'Team'].values[0]


def calculate_round(df, round, df_matchup, win_style='Best'):
    # Fill in winners of previous round
    if round > 1:
        df.loc[df.Round == round, 'StrongSeed'] = df.loc[df.Round == round].apply(lambda x: find_slot_winner(df, x.StrongSeed, x.Tournament), axis=1)
        df.loc[df.Round == round, 'WeakSeed'] = df.loc[df.Round == round].apply(lambda x: find_slot_winner(df, x.WeakSeed, x.Tournament), axis=1)
    
    # Find winners of current round
    df.loc[df.Round == round, 'Team'] = df.loc[df.Round == round].apply(lambda x: get_winner(df_matchup, x.StrongSeed, x.WeakSeed, x.Tournament, win_style=win_style), axis=1)

    return df



def run(file_path='./data/predictions/', file_name='bracket',
        pred_file='./data/predictions/preds.csv',
        seed_file='./data/kaggle/2024_tourney_seeds.csv',
        slot_file='./data/kaggle/MNCAATourneySlots.csv',
        year=2023, num_brackets=1, win_style='Best'):
    """
    """
    # Load prediction file
    df = pd.read_csv(pred_file)
    # Extract teams
    df['TeamA'] = df.ID.apply(lambda x: x.split('_')[1]).astype(int)
    df['TeamB'] = df.ID.apply(lambda x: x.split('_')[2]).astype(int)

    # Load seed file
    df_seed = pd.read_csv(seed_file)

    # Merge seed and pred file
    df = df.merge(df_seed, left_on='TeamA', right_on='TeamID')
    df = df.merge(df_seed, left_on='TeamB', right_on='TeamID', suffixes=['A', 'B'])
    df = df.drop(['TeamIDA', 'TeamIDB', 'TournamentB', 'ID'], axis=1)
    df = df.rename({'TournamentA':'Tournament'}, axis=1)

    # Load slot file
    df_slot = pd.read_csv(slot_file)
    # Create mens and womens slots
    df_slot = df_slot[df_slot.Season == year]
    df_slot_m = df_slot.copy()
    df_slot_m['Tournament'] = 'M'
    df_slot_w = df_slot.copy()
    df_slot_w['Tournament'] = 'W'
    df_slot = pd.concat([df_slot_m, df_slot_w])

    # Extract round for slot matches
    df_slot['Round'] = df_slot.Slot.apply(lambda x: int(x[1]) if x[0] == 'R' else 0)
    df_slot = df_slot[df_slot.Round > 0]

    # Create team column
    df_slot['Team'] = ''

    # Generate all brackets
    brackets = []
    for ib in range(num_brackets):
        bracket = df_slot.copy()
        # Calculate winners
        for round in range(1, 7):
            bracket = calculate_round(bracket, round, df, win_style=win_style)
        # Save bracket
        bracket['Bracket'] = ib + 1
        brackets.append(bracket)

    # Combine brackets
    df_all = pd.concat(brackets)

    # Format slot file
    df_all.reset_index(drop=True, inplace=True)
    df_all.reset_index(drop=False, inplace=True, names='RowId')
    df_all = df_all[['RowId','Tournament', 'Bracket', 'Slot', 'Team']]

    # Save bracket predictions
    df_all.to_csv(file_path + file_name + '.csv', index=False)


if __name__ == "__main__":
    run()