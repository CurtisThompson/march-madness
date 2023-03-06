import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from dash import Dash, dcc, html, no_update
from dash.dependencies import Input, Output

# Datasets
data_shap_men = pd.read_csv('./data/explain/shap_men.csv')
data_shap_women = pd.read_csv('./data/explain/shap_women.csv')


# Main app
app = Dash(__name__)


# App layout
app.layout = html.Div(
    children=[
        html.H1(children="March Madness Prediction Explorer"),
        html.P(children="Look at match predictions for each game of March Madness."),
        dcc.Dropdown(
            id='home_team',
            options=[{'label': '1173', 'value': 1173}, {'label': '1127', 'value': 1127}],
            value=1173
        ),
        dcc.Dropdown(
            id='away_team',
            options=[{'label': '1187', 'value': 1187}, {'label': '1416', 'value': 1416}],
            value=1187
        ),
        html.Img(id="shap-game-prediction")
    ]
)


@app.callback(
    Output('shap-game-prediction', 'src'),
    [Input('home_team', 'value'), Input('away_team', 'value')]
)
def update_figure(teama, teamb):
    # Get teams in ID order, and construct ID
    year = 2023
    home_team = min(teama, teamb)
    away_team = max(teama, teamb)
    match_id = str(year) + '_' + str(home_team) + '_' + str(away_team)
    print(match_id)

    # If not a row in shap dataset then do not update graph
    if match_id not in data_shap_men.ID.values:
        print('no update')
        return no_update

    print('will update')

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