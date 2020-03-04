import numpy as np
import pandas as pd
import tensorflow as tf
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import plotly.express as px
from plotly.colors import DEFAULT_PLOTLY_COLORS
from dash.dependencies import Input, Output
tf.keras.backend.set_floatx('float64')

from mllib.data import PandasHdfSource
from mllib.model import KerasSurrogate

from invertible_network.invertible_neural_network import InvertibleNetworkSurrogate

#########################################
# constants of the machine
#########################################
cavity_locations = [
    (1., 1.8),
    (3., 4.5),
    (5., 6.),
    (7., 8.2),
    (8.4, 9.5)
]

YAG_screens = [
    0.5,
    2.93,
    6.22,
    9.47,
    11.36,
    11.57,
    15.13,
    16.70,
    19.26,
    20.69,
    22.99,
    25.27,
    #27.78
]

solenoid_locations = [
    2.08,
    4.65,
    6.69,
]

# colors for the beamline elements
machine_element_colors = {
    'cavity': px.colors.qualitative.Safe[0],
    'YAG': px.colors.qualitative.Safe[1],
    'solenoid': px.colors.qualitative.Safe[2],
}

element_shade_opacity = 0.7


#########################################
# functions to build widgets and plots
#########################################
def build_dvar_control(dvar, minimum, maximum, label):
    numeric_input = daq.NumericInput(min=minimum,
                        max=maximum,
                        value=minimum,
                        id='{}_numeric_input'.format(dvar),
                        #size=100,
                        className='numericInput',
                        )

    label = html.Div(label,
                    className='dvarLabel',
                    id='{}_label'.format(dvar))
    range_label = html.Div('[{}, {}]'.format(minimum, maximum), className='dvarLabel')
    labelled_textfield = html.Div([label, numeric_input, range_label],
                                    id=dvar, className='dvarContainer')

    return labelled_textfield

def build_graph_dict(s, qoi, y_label, y_ranges, uncertainty):
    # for details about the error bars, see:
    # https://plot.ly/python/v3/continuous-error-bars/
    return {
        'data': [
            # lower uncertainty "bar"
            {
                'x': uncertainty.index,
                'y': qoi - uncertainty.values,
                'line': {'width': 0},
            },
            # prediction
            {
                'x': s,
                'y': qoi,
                'line': {'color': DEFAULT_PLOTLY_COLORS[0]},
                'mode': 'lines',
                'name': 'prediction',
                'fill': 'tonexty',
                'fillcolor': 'LightGray',
            },
            # upper uncertainty "bar"
            {
                'x': uncertainty.index,
                'y': qoi + uncertainty.values,
                'fill': 'tonexty',
                'fillcolor': 'LightGray',
                'line': {'width': 0},
            },
        ],
        'layout': {
            'showlegend': False,
            'xaxis': {
                'title': {
                    'text': 's [m]',
                    'font': {
                            'size': 30
                    },
                },
                'tickfont': {
                    'size': 20,
                }
            },
            'yaxis': {
                'title': {
                    'text': y_label,
                    'font': {
                        'size': 30
                    }
                },
                'tickfont': {
                    'size': 20,
                },
                'range': y_ranges
            },
            # shaded areas
            'shapes': [
                # cavitites
                {
                    'type': 'rect',
                    'xref': 'x',
                    'yref': 'paper',
                    'x0': x_start,
                    'y0': 0.,
                    'x1': x_end,
                    'y1': 1.,
                    'layer': 'below',
                    'opacity': element_shade_opacity,
                    'line': {
                        'color': machine_element_colors['cavity'],
                        'width': 0,
                    },
                    'fillcolor': machine_element_colors['cavity'],
                } for (x_start, x_end) in cavity_locations
            ] + [
                # YAG screens
                {
                    'type': 'line',
                    'xref': 'x',
                    'yref': 'paper',
                    'x0': x,
                    'y0': 0.,
                    'x1': x,
                    'y1': 1.,
                    'layer': 'below',
                    'opacity': element_shade_opacity,
                    'line': {
                        'color': machine_element_colors['YAG'],
                        'width': 2,
                    }
                } for x in YAG_screens
            ] + [
                # solenoids
                {
                    'type': 'line',
                    'xref': 'x',
                    'yref': 'paper',
                    'x0': x,
                    'y0': 0.,
                    'x1': x,
                    'y1': 1.,
                    'layer': 'below',
                    'opacity': element_shade_opacity,
                    'line': {
                        'color': machine_element_colors['solenoid'],
                        'width': 2,
                    }
                } for x in solenoid_locations
            ]
        }
    }

###################
# Data structure
###################
qoi_columns = [
    'Mean Bunch Energy',
    'RMS Beamsize in x',
    'RMS Beamsize in y',
    #'RMS Beamsize in s',
    'Normalized Emittance x',
    'Normalized Emittance y',
    #'Normalized Emittance s',
    #'RMS Normalized Momenta in x',
    #'RMS Normalized Momenta in y',
    #'RMS Normalized Momenta in s',
    'energy spread of the beam',
    'Correlation xpx',
    'Correlation ypy',
    #'Correlation zpz',
]

columns_to_keep = [
    'Mean Bunch Energy',
    'RMS Beamsize in x',
    'RMS Beamsize in y',
    #'RMS Beamsize in s',
    'Normalized Emittance x',
    'Normalized Emittance y',
    #'Normalized Emittance s',
    #'RMS Normalized Momenta in x',
    #'RMS Normalized Momenta in y',
    #'RMS Normalized Momenta in s',
    'energy spread of the beam',
    'Correlation xpx',
    'Correlation ypy',
    #'Correlation zpz',
    'Path length'
]

dvar_ranges = {
    'IBF': (450, 550),
    'IM': (150, 260),
    'GPHASE': (-50, 10),
    'ILS1': (0, 250),
    'ILS2': (0, 200),
    'ILS3': (0, 200),
    'Bunch charge': (1, 5),
    'lambda': (0.3, 2),
    'SIGXY': (1.5, 12.5)
}

# values taken from the single_sample_vs_ground_truth.py dashboard
# (by eye, not from the test set range!)
qoi_ranges = {
    'RMS Beamsize in x': (0, 15),
    'RMS Beamsize in y': (0, 15),
    #'RMS Beamsize in s': (0, 1.5),
    'Normalized Emittance x': (0, 0.3),
    'Normalized Emittance y': (0, 0.3),
    #'Normalized Emittance s': (0, 0.15),
    'Correlation xpx': (-1, 1),
    'Correlation ypy': (-1, 1),
    #'Correlation zpz': (-1, 1),
    'Mean Bunch Energy': (0, 70),
    'energy spread of the beam': (0., 1000.)
}

dvars = [
    'IBF',
    'IM',
    'GPHASE',
    'ILS1',
    'ILS2',
    'ILS3',
    'Bunch charge',
    'lambda',
    'SIGXY'
]

dvar_labels = {
    'IBF': 'IBF [A]',
    'IM': 'IM [A]',
    'GPHASE': 'GPHASE [°]',
    'ILS1': 'ILS1 [A]',
    'ILS2': 'ILS2 [A]',
    'ILS3': 'ILS3 [A]',
    'Bunch charge': 'Bunch charge [nC]',
    'lambda': 'lambda [ps]',
    'SIGXY': 'Laser radius [mm]'
}

###########################
# load model
###########################
is_invertible = False

model_name = 'hiddenLayers_8_unitsPerLayer_500_activation_relu_batch_size_128_learning_rate_0.0001_optimizer_adam_epochs_12_awa_range_dense_filtered_8x500_0_to_26m_4peak_distr_12_epochs'

if is_invertible:
    model = InvertibleNetworkSurrogate.load('.', model_name)
else:
    model = KerasSurrogate.load('.', model_name)

app = dash.Dash(__name__)

components = []

##########################
# load model uncertainty
##########################
# "Uncertainty" here means the quantiles of the residuals on the test set
# at each longitudinal position.
uncertainties = pd.read_csv('residual_quantiles_by_longitudinal_pos.csv', index_col=(0, 1))
uncertainty_quantile_levels = uncertainties.index.get_level_values(1).unique()

##########################
# DVAR sliders
##########################
table_cells = []
n_cols = 2

for dvar, label in dvar_labels.items():
    minimum = dvar_ranges[dvar][0]
    maximum = dvar_ranges[dvar][1]

    labelled_textfield = build_dvar_control(dvar, minimum, maximum, label)
    table_cells.append(labelled_textfield)

rows = []

for i in range((len(table_cells) // n_cols) + 1):
    row = table_cells[i*n_cols:(i+1)*n_cols]
    row = html.Tr([html.Td(element) for element in row])
    rows.append(row)

table = html.Table(rows, id='dvar_table')

# color labels
color_labels = []
for key, value in machine_element_colors.items():
    color_labels.append(html.P(key, style={
        'color': value,
        'font-weight': 'bold',
        'font-size': 30,
    }))
color_labels = html.Div(color_labels)

# quantiles for the uncertainty
uncertainty_drowpdown = html.Div([
    html.P('Quantile of the residuals to use as uncertainty:'),
    dcc.Dropdown(
        id='uncertainty_dropdown',
        options=[{
            'label': lvl,
            'value': lvl,
        } for lvl in uncertainty_quantile_levels],
        value=0.95
    ),
])

container = html.Div([table, uncertainty_drowpdown, color_labels], id='right_panel_container')

components.append(container)

##########################
# Graphs
##########################

# beam sizes
components.append(dcc.Markdown('# Beam sizes'))
beam_sizes = html.Div([
    dcc.Graph(id='sigma_x'),
    dcc.Graph(id='sigma_y'),
    #dcc.Graph(id='sigma_s'),
], id='beam_sizes')
components.append(beam_sizes)

# emittances
components.append(dcc.Markdown('# Emittances'))
emittances = html.Div([
    dcc.Graph(id='epsilon_x'),
    dcc.Graph(id='epsilon_y'),
    #dcc.Graph(id='epsilon_s'),
])
components.append(emittances)

# energy & energy spread
components.append(dcc.Markdown('# E & dE'))
energy = html.Div([
    dcc.Graph(id='energy'),
    dcc.Graph(id='energy_spread'),
])
components.append(energy)

# correlations
components.append(dcc.Markdown('# Correlations'))
corrs = html.Div([
    dcc.Graph(id='correlation_x'),
    dcc.Graph(id='correlation_y'),
    #dcc.Graph(id='correlation_s'),
])
components.append(corrs)

# add all components
app.layout = html.Div(components)

server = app.server


####################################
# Update graphs
####################################
@app.callback(
    [
        Output('sigma_x', 'figure'),
        Output('sigma_y', 'figure'),
        #Output('sigma_s', 'figure'),
        Output('epsilon_x', 'figure'),
        Output('epsilon_y', 'figure'),
        #Output('epsilon_s', 'figure'),
        Output('correlation_x', 'figure'),
        Output('correlation_y', 'figure'),
        #Output('correlation_s', 'figure'),
        Output('energy', 'figure'),
        Output('energy_spread', 'figure'),
    ],
        [Input(f'{dvar}_numeric_input', 'value') for dvar in dvars] + [Input('uncertainty_dropdown', 'value')]
    )
def update_graphs(IBF, IM, GPHASE, ILS1, ILS2, ILS3, bunch_charge, cavityVoltage, SIGXY, uncertainty_lvl):
    s_values = uncertainties.index.get_level_values(0).unique()

    X = [np.array([float(IBF),
                    float(IM),
                    float(GPHASE),
                    float(ILS1),
                    float(ILS2),
                    float(ILS3),
                    float(bunch_charge),
                    float(cavityVoltage),
                    float(SIGXY),
                    s]).reshape(1, 10) for s in s_values]

    X = np.vstack(X)
    
    prediction = model.predict(X)
    if is_invertible:
        # the invertible network predicts "s" as the last column --> remove it
        prediction = prediction[:, :-1]
    prediction = pd.DataFrame(data=prediction, columns=qoi_columns)

    to_return = []

    # get only rows corresponding to the selected residual quantile
    uncertainties_to_plot = uncertainties.loc[(s_values, uncertainty_lvl), :]
    print(uncertainties_to_plot.shape)
    # remove the now unnecessary quantile subindex
    uncertainties_to_plot.index = uncertainties_to_plot.index.droplevel(1)

    print(uncertainties_to_plot.shape)

    # beam sizes
    to_return.append(build_graph_dict(s_values, prediction['RMS Beamsize in x'] * 1000., 'sigma_x [mm]', qoi_ranges['RMS Beamsize in x'], uncertainties_to_plot['RMS Beamsize in x']))
    to_return.append(build_graph_dict(s_values, prediction['RMS Beamsize in y'] * 1000., 'sigma_y [mm]', qoi_ranges['RMS Beamsize in y'], uncertainties_to_plot['RMS Beamsize in y']))
    #to_return.append(build_graph_dict(s_values, prediction['RMS Beamsize in s'] * 1000., 'sigma_s [mm]', qoi_ranges['RMS Beamsize in s']))

    # emittances
    to_return.append(build_graph_dict(s_values, prediction['Normalized Emittance x'] * 1000., 'epsilon_x [mm rad]', qoi_ranges['Normalized Emittance x'], uncertainties_to_plot['Normalized Emittance x']))
    to_return.append(build_graph_dict(s_values, prediction['Normalized Emittance y'] * 1000., 'epsilon_y [mm rad]', qoi_ranges['Normalized Emittance y'], uncertainties_to_plot['Normalized Emittance y']))
    #to_return.append(build_graph_dict(s_values, prediction['Normalized Emittance s'] * 1000., 'epsilon_s [mm rad]', qoi_ranges['Normalized Emittance s']))

    # correlations
    to_return.append(build_graph_dict(s_values, prediction['Correlation xpx'], 'corr(x, px)', qoi_ranges['Correlation xpx'], uncertainties_to_plot['Correlation xpx']))
    to_return.append(build_graph_dict(s_values, prediction['Correlation ypy'], 'corr(y, py)', qoi_ranges['Correlation ypy'], uncertainties_to_plot['Correlation ypy']))
    #to_return.append(build_graph_dict(s_values, prediction['Correlation zpz'], 'corr(s, ps)', qoi_ranges['Correlation zpz']))

    # E & dE
    to_return.append(build_graph_dict(s_values, prediction['Mean Bunch Energy'], 'E [MeV]', qoi_ranges['Mean Bunch Energy'], uncertainties_to_plot['Mean Bunch Energy']))
    to_return.append(build_graph_dict(s_values, prediction['energy spread of the beam'] * 1000., 'dE [keV]', qoi_ranges['energy spread of the beam'], uncertainties_to_plot['energy spread of the beam']))

    return to_return


if __name__ == '__main__':
    app.run_server(debug=True)
