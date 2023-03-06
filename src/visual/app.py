import io
import base64
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from dash import Dash, dcc, html, no_update
from dash.dependencies import Input, Output


# Datasets
data_shap_men = pd.read_csv('./data/explain/shap_men.csv')
data_shap_women = pd.read_csv('./data/explain/shap_women.csv')
mens_teams = pd.read_csv('./data/kaggle/MTeams.csv').rename(columns={'TeamID':'value', 'TeamName':'label'})
mens_teams = mens_teams[['label', 'value']].to_dict(orient='records')


# Main app
assets_path = os.getcwd() + '/viz/assets'
print(assets_path)
app = Dash(__name__, assets_folder=assets_path)


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
                html.H2(children="Predictor Factors"),
                html.Img(id="shap-game-prediction", className='shap-image')
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
    Output('shap-game-prediction', 'src'),
    [Input('home_team', 'value'), Input('away_team', 'value')]
)
def update_figure(teama, teamb):
    # If no team selected, do not update images
    if teama == None or teamb == None:
        return no_update

    # Get teams in ID order, and construct ID
    year = 2023
    home_team = min(teama, teamb)
    away_team = max(teama, teamb)
    match_id = str(year) + '_' + str(home_team) + '_' + str(away_team)

    # If not a row in shap dataset then do not update graph
    if match_id not in data_shap_men.ID.values:
        return no_update

    # Find match predictions in shap dataset
    shapset = data_shap_men.loc[data_shap_men.ID == match_id].reset_index(drop=True).iloc[0]

    # Create plot and save to memory
    data = create_force_plot(shapset)

    return "data:image/png;base64,{}".format(data)


# Build shap plot
def create_force_plot(shapset):
    # Plot values
    columns = ['SeedDiff', 'EloWinProbA', 'WinRatioA', 'WinRatioB', 'ClutchRatioA', 'ClutchRatioB']
    base_val = shapset['shap_BaseVal']
    shap_vals = shapset[['shap_'+c for c in columns]].values
    column_vals = shapset[columns].values
    out_name = 'Team A Win Probability'

    # Round column values for display
    column_vals = np.array(["{:.2f}".format(c) if type(c) == np.float64 else c for c in column_vals])

    # Allocate memory buffer for image
    buffer = io.BytesIO()

    # Plot and save
    shap.plots.force(base_val, shap_values=shap_vals, features=column_vals, feature_names=columns, out_names='',
                     matplotlib=True, show=False, contribution_threshold=0)
    #plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()

    # Get image data from buffer, clear and return
    buffer_data = base64.b64encode(buffer.getbuffer()).decode("utf8")
    buffer.close()
    return buffer_data


# Run app by running this script
if __name__ == '__main__':
    app.run_server(debug=True)