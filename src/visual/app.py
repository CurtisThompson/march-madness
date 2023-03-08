import os

from dash import Dash, dcc, html, no_update, dash_table
from dash.dependencies import Input, Output

import app_utils


# Datasets
data_shap_men, data_shap_women = app_utils.load_shap_sets()
data_preds = app_utils.load_game_predictions()
mens_teams = app_utils.get_teams_list(False)
mens_id_name_dict = app_utils.get_teams_dict(mens_teams)
womens_teams = app_utils.get_teams_list(True)
womens_id_name_dict = app_utils.get_teams_dict(womens_teams)
team_forms = app_utils.get_teams_form()
team_win_records = app_utils.get_teams_win_loss(2023)
test_set = app_utils.load_test_set()


# Main app
assets_path = os.getcwd() + '/viz/assets'
app = Dash(__name__, assets_folder=assets_path)
app.title = 'March Madness Prediction Explorer'


# App layout
app.layout = html.Div(
    children=[
        html.Div(
            className='app-header',
            children=[
                html.H1(children="March Madness Prediction Explorer"),
                html.P(children="Look at match predictions for each game of March Madness."),
            ]
        ),
        html.Div(
            className='app-settings',
            children=[
                html.Div(children=[
                    html.Label(children="Competition"),
                    dcc.Dropdown(
                        id='competition-gender',
                        options=[{'label': "Men's", 'value':0}, {'label': "Women's", 'value':1}],
                        value=0
                    )
                ], className="app-single-setting-container"),
                html.Div(children=[
                    html.Label(children="Team"),
                    dcc.Dropdown(
                        id='home_team',
                        options=mens_teams,
                        value=1173
                    )
                ], className="app-single-setting-container"),
                html.Div(children=[
                    html.Label(children="Team"),
                    dcc.Dropdown(
                        id='away_team',
                        options=mens_teams,
                        value=1187
                    ),
                ], className="app-single-setting-container")
            ]
        ),
        html.Div(
            className='app-body',
            children=[
                html.Img(id="shap-game-prediction", className='shap-image'),
                html.Div(
                    className='team-name',
                    children=[
                        html.P(className='team-win-ratio', children="0-0", id='team-a-win-ratio'),
                        html.P(className='team-full-name', children="Team A", id='team-a-full-name'),
                        html.Div(
                            className='team-form-container',
                            children=[
                                html.Span(className='team-form-win', id='team-a-form-1'),
                                html.Span(className='team-form-loss', id='team-a-form-2'),
                                html.Span(className='team-form-win', id='team-a-form-3'),
                                html.Span(className='team-form-win', id='team-a-form-4'),
                                html.Span(className='team-form-win', id='team-a-form-5')
                            ]
                        ),
                        html.P(className='team-win-prob', children="50%", id='team-a-win-prob')
                    ]
                ),
                html.Div(
                    className='team-name',
                    children=[
                        html.P(className='team-win-ratio', children="0-0", id='team-b-win-ratio'),
                        html.P(className='team-full-name', children="Team B", id='team-b-full-name'),
                        html.Div(
                            className='team-form-container',
                            children=[
                                html.Span(className='team-form-win', id='team-b-form-1'),
                                html.Span(className='team-form-loss', id='team-b-form-2'),
                                html.Span(className='team-form-loss', id='team-b-form-3'),
                                html.Span(className='team-form-win', id='team-b-form-4'),
                                html.Span(className='team-form-loss', id='team-b-form-5')
                            ]
                        ),
                        html.P(className='team-win-prob', children="50%", id='team-b-win-prob')
                    ]
                ),
                dash_table.DataTable(id='main-stats-table', data=[])
            ]
        ),
        html.Div(
            className='app-footer',
            children=[
                html.P(children="Curtis Thompson | https://github.com/CurtisThompson/march-madness")
            ]
        ),
    ]
)


@app.callback(
        [Output('main-stats-table', 'data'), Output('main-stats-table', 'columns')],
        [Input('competition-gender', 'value'), Input('home_team', 'value'), Input('away_team', 'value')]
)
def update_main_stats_table(gender, teama, teamb):
    """Callback to update match features table."""

    if app_utils.cannot_update_visuals(gender, teama, teamb):
        return no_update

    # Get teams in ID order, and construct ID
    year = 2023
    match_id, home_team, away_team = app_utils.get_match_id(year, teama, teamb)

    # Get team names
    name_dict = mens_id_name_dict if gender == 0 else womens_id_name_dict
    home_name = name_dict[home_team]
    away_name = name_dict[away_team]

    # If not a row in shap dataset then do not update graph
    if match_id not in test_set.ID.values:
        return no_update
    
    df = app_utils.get_feature_table(test_set,
                                     ['Seed', 'Elo', 'WinRatio'],
                                     match_id).to_dict('records')

    columns = [{'id': 'TeamA', 'name': home_name},
               {'id': 'Statistics', 'name': 'Statistics'},
               {'id': 'TeamB', 'name': away_name}]

    return df, columns


@app.callback(
        [Output('team-a-form-1', 'className'), Output('team-a-form-2', 'className'),
         Output('team-a-form-3', 'className'), Output('team-a-form-4', 'className'),
         Output('team-a-form-5', 'className'),
         Output('team-b-form-1', 'className'), Output('team-b-form-2', 'className'),
         Output('team-b-form-3', 'className'), Output('team-b-form-4', 'className'),
         Output('team-b-form-5', 'className')],
        [Input('competition-gender', 'value'), Input('home_team', 'value'), Input('away_team', 'value')]
)
def update_team_forms(gender, teama, teamb):
    """Callback to update team form."""

    if app_utils.cannot_update_visuals(gender, teama, teamb):
        return no_update
    
    # Get home and away team, and forms
    match_id, home_team, away_team = app_utils.get_match_id(2023, teama, teamb)
    home_form = team_forms.loc[team_forms.Team == home_team].reset_index(drop=True).iloc[0]
    away_form = team_forms.loc[team_forms.Team == away_team].reset_index(drop=True).iloc[0]
    
    # Function to get CSS form class
    form_class = lambda x: 'team-form-win' if x == 'W' else 'team-form-loss'

    return (form_class(home_form.G1), form_class(home_form.G2), form_class(home_form.G3), form_class(home_form.G4),
            form_class(home_form.G5), form_class(away_form.G1), form_class(away_form.G2), form_class(away_form.G3),
            form_class(away_form.G4), form_class(away_form.G5))


@app.callback(
        [Output('team-a-full-name', 'children'), Output('team-a-win-prob', 'children'),
         Output('team-b-full-name', 'children'), Output('team-b-win-prob', 'children'),
         Output('team-a-win-ratio', 'children'), Output('team-b-win-ratio', 'children')],
        [Input('competition-gender', 'value'), Input('home_team', 'value'), Input('away_team', 'value')]
)
def update_team_win_probs(gender, teama, teamb):
    """Callback to update team names and win probabilities."""

    if app_utils.cannot_update_visuals(gender, teama, teamb):
        return no_update

    # Get teams in ID order, and construct ID
    year = 2023
    match_id, home_team, away_team = app_utils.get_match_id(year, teama, teamb)

    # If not a row in shap dataset then do not update graph
    if match_id not in data_preds.ID.values:
        return no_update

    # Get win probabilities
    home_win_prob, away_win_prob = app_utils.get_match_win_probs(data_preds, match_id)

    # Get team names
    name_dict = mens_id_name_dict if gender == 0 else womens_id_name_dict
    home_name = name_dict[home_team]
    away_name = name_dict[away_team]

    # Get team win ratios
    home_win_ratio = app_utils.get_team_win_record(home_team, team_win_records)
    away_win_ratio = app_utils.get_team_win_record(away_team, team_win_records)

    return home_name, home_win_prob, away_name, away_win_prob, home_win_ratio, away_win_ratio


@app.callback(
        [Output('home_team', 'options'), Output('home_team', 'value'),
         Output('away_team', 'options'), Output('away_team', 'value')],
        [Input('competition-gender', 'value')]
)
def update_dashboard_gender(gender):
    """Callback to change the team choices in app settings bar."""
    if gender == 1:
        return womens_teams, 3247, womens_teams, 3348
    else:
        return mens_teams, 1173, mens_teams, 1187


@app.callback(
    Output('shap-game-prediction', 'src'),
    [Input('competition-gender', 'value'), Input('home_team', 'value'), Input('away_team', 'value')]
)
def update_figure(gender, teama, teamb):
    """Callback to change the shap force plot of important prediction features."""

    if app_utils.cannot_update_visuals(gender, teama, teamb):
        return no_update

    # Get teams in ID order, and construct ID
    year = 2023
    match_id, home_team, away_team = app_utils.get_match_id(year, teama, teamb)

    # If not a row in shap dataset then do not update graph
    data_shap = data_shap_men if gender == 0 else data_shap_women
    if match_id not in data_shap.ID.values:
        return no_update

    # Find match predictions in shap dataset
    shapset = data_shap.loc[data_shap.ID == match_id].reset_index(drop=True).iloc[0]

    # Create plot and save to memory
    data = app_utils.create_force_plot(shapset)

    return "data:image/png;base64,{}".format(data)


# Run app by running this script
if __name__ == '__main__':
    app.run_server(debug=True)