import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output

from mllib.data import PandasHdfSource
from mllib.model import KerasSurrogate

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

def build_graph_dict(s, qoi, y_label, y_ranges):
    return {
        'data': [
            {
                'x': s,
                'y': qoi,
                'mode': 'lines',
                'name': 'prediction'
            }
        ],
        'layout': {
            'xaxis': {'title': 's [m]'},
            'yaxis': {
                'title': y_label,
                'range': y_ranges
            }
        }
    }

###################
# Data structure
###################
qoi_columns = [
    'Number of Macro Particles',
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
    'Number of Macro Particles',
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
    'IBF': (200, 450),
    'IM': (210, 260),
    'GPHASE': (-25, 0),
    'ILS1': (50, 250),
    'ILS2': (100, 200),
    'ILS3': (150, 200),
    'Bunch charge': (1, 5),
    'cavityVoltage': (12, 25),
    'SIGXY': (9, 27)
}

# values taken from the single_sample_vs_ground_truth.py dashboard
# (by eye, not from the test set range!)
qoi_ranges = {
    'RMS Beamsize in x': (0, 12),
    'RMS Beamsize in y': (0, 12),
    #'RMS Beamsize in s': (0, 1.5),
    'Normalized Emittance x': (0, 0.3),
    'Normalized Emittance y': (0, 0.3),
    #'Normalized Emittance s': (0, 0.15),
    'Correlation xpx': (-1, 1),
    'Correlation ypy': (-1, 1),
    #'Correlation zpz': (-1, 1),
    'Mean Bunch Energy': (0, 70),
    'energy spread of the beam': (0, 0.15)
}

dvars = [
    'IBF',
    'IM',
    'GPHASE',
    'ILS1',
    'ILS2',
    'ILS3',
    'Bunch charge',
    'cavityVoltage',
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
    'cavityVoltage': 'Cavity voltage [MV]',
    'SIGXY': 'Laser radius [mm]'
}

###########################
# load model
###########################
model_name = 'hiddenLayers_8_unitsPerLayer_300_activation_relu_batch_size_128_learning_rate_0.001_optimizer_adam_epochs_700_double_varying_radius_new_scaling_final'

model = KerasSurrogate.load('.', model_name)

app = dash.Dash(__name__)

components = []

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
components.append(table)

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
    [Input('{}_numeric_input'.format(dvar), 'value') for dvar in dvars])
def update_graphs(IBF, IM, GPHASE, ILS1, ILS2, ILS3, bunch_charge, cavityVoltage, SIGXY):
    s_values = np.linspace(1., 14., 1000)

    X = [np.array([float(IBF),
                    float(IM),
                    float(GPHASE),
                    float(ILS1),
                    float(ILS2),
                    float(ILS3),
                    float(bunch_charge),
                    float(cavityVoltage),
                    float(SIGXY) / 1000.,
                    s]).reshape(1, 10) for s in s_values]

    X = np.vstack(X)
    
    prediction = model.predict(X)
    prediction = pd.DataFrame(data=prediction, columns=qoi_columns)

    to_return = []

    # beam sizes
    to_return.append(build_graph_dict(s_values, prediction['RMS Beamsize in x'] * 1000., 'sigma_x [mm]', qoi_ranges['RMS Beamsize in x']))
    to_return.append(build_graph_dict(s_values, prediction['RMS Beamsize in y'] * 1000., 'sigma_y [mm]', qoi_ranges['RMS Beamsize in y']))
    #to_return.append(build_graph_dict(s_values, prediction['RMS Beamsize in s'] * 1000., 'sigma_s [mm]', qoi_ranges['RMS Beamsize in s']))

    # emittances
    to_return.append(build_graph_dict(s_values, prediction['Normalized Emittance x'] * 1000., 'epsilon_x [mm rad]', qoi_ranges['Normalized Emittance x']))
    to_return.append(build_graph_dict(s_values, prediction['Normalized Emittance y'] * 1000., 'epsilon_y [mm rad]', qoi_ranges['Normalized Emittance y']))
    #to_return.append(build_graph_dict(s_values, prediction['Normalized Emittance s'] * 1000., 'epsilon_s [mm rad]', qoi_ranges['Normalized Emittance s']))

    # correlations
    to_return.append(build_graph_dict(s_values, prediction['Correlation xpx'], 'corr(x, px)', qoi_ranges['Correlation xpx']))
    to_return.append(build_graph_dict(s_values, prediction['Correlation ypy'], 'corr(y, py)', qoi_ranges['Correlation ypy']))
    #to_return.append(build_graph_dict(s_values, prediction['Correlation zpz'], 'corr(s, ps)', qoi_ranges['Correlation zpz']))

    # E & dE
    to_return.append(build_graph_dict(s_values, prediction['Mean Bunch Energy'], 'E [MeV]', qoi_ranges['Mean Bunch Energy']))
    to_return.append(build_graph_dict(s_values, prediction['energy spread of the beam'], 'dE [MeV]', qoi_ranges['energy spread of the beam']))

    return to_return


if __name__ == '__main__':
    app.run_server(debug=True)
