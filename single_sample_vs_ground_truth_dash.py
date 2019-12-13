import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from mllib.data import PandasHdfSource
from mllib.model import KerasSurrogate

#########################################
# functions to build widgets and plots
#########################################
def build_table_rows(design_variables):
    rows = []
    for design_var in dvar.drop(columns='Path length').columns:
        value = '{:4>.1f}'.format(design_variables[design_var])
        label = html.B(design_var)
        row = html.Tr([html.Td(label),
                        html.Td(value)])
        rows.append(row)
    return rows

def build_graph_dict(s, qoi_true, qoi_pred, y_label, y_range=(0, 1)):
    return {
        'data': [
            {
                'x': s,
                'y': qoi_true,
                'mode': 'lines',
                'name': 'ground truth'
            },
            {
                'x': s,
                'y': qoi_pred,
                'mode': 'lines',
                'name': 'prediction'
            }
        ],
        'layout': {
            'xaxis': {'title': 's [m]'},
            'yaxis': {
                'title': y_label,
                'range': y_range
            }
        }
    }


#################
# load dataset
#################
test_set_path = 'random_sample_with_linac_solenoids_charge_sweep 1k slinear_sample_id 2D cleaned including momenta.hdf5'

src = PandasHdfSource('source', test_set_path)
dvar, qoi = src.get_data()
qoi['Path length'] = dvar['Path length']

columns_to_keep = [
    'Number of Macro Particles',
    'Mean Bunch Energy',
    'RMS Beamsize in x',
    'RMS Beamsize in y',
    'RMS Beamsize in s',
    'Normalized Emittance x',
    'Normalized Emittance y',
    'Normalized Emittance s',
    #'RMS Normalized Momenta in x',
    #'RMS Normalized Momenta in y',
    #'RMS Normalized Momenta in s',
    'energy spread of the beam',
    'Correlation xpx',
    'Correlation ypy',
    'Correlation zpz',
    'Path length'
]

qoi = qoi[columns_to_keep]

num_samples = dvar['sample_id'].max()

y_ranges = qoi.describe().loc[['min', 'max']]

###############
# load model
###############
model_name = 'hiddenLayers_12_unitsPerLayer_100_activation_relu_L2regularizer_0.0_dropout_0.0_batch_size_128_learning_rate_0.001_optimizer_adam_epochs_10_cycles_15_epochs_per_cycle_150_random_sample_with_linac_solenoids_charge_sweep_more_qois_more_qoi_final'

model = KerasSurrogate.load('.', model_name)


#########################
# Build widgets & plots
#########################
app = dash.Dash(__name__)

components = []

# 'Random' button
components.append(html.Button('Select random sample', id='selectRandom'))

# DVAR table
rows = build_table_rows(dvar.loc[0])
table = html.Table(rows, id='dvar_table')
components.append(table)

# beam sizes
components.append(dcc.Markdown('# Beam sizes'))
beam_sizes = html.Div([
    dcc.Graph(id='sigma_x'),
    dcc.Graph(id='sigma_y'),
    dcc.Graph(id='sigma_s'),
], id='beam_sizes')
components.append(beam_sizes)

# emittances
components.append(dcc.Markdown('# Emittances'))
emittances = html.Div([
    dcc.Graph(id='epsilon_x'),
    dcc.Graph(id='epsilon_y'),
    dcc.Graph(id='epsilon_s'),
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
    dcc.Graph(id='correlation_s'),
])
components.append(corrs)

# add all components
app.layout = html.Div(components)


##################################################
# Callback for selecting and plotting new sample
##################################################
@app.callback(
    [
        Output('sigma_x', 'figure'),
        Output('sigma_y', 'figure'),
        Output('sigma_s', 'figure'),
        Output('epsilon_x', 'figure'),
        Output('epsilon_y', 'figure'),
        Output('epsilon_s', 'figure'),
        Output('correlation_x', 'figure'),
        Output('correlation_y', 'figure'),
        Output('correlation_s', 'figure'),
        Output('energy', 'figure'),
        Output('energy_spread', 'figure'),
        Output('dvar_table', 'children'),
    ],
    [Input('selectRandom', 'n_clicks')],
    [dash.dependencies.State('selectRandom', 'value')])
def update_graphs(n_clicks, value):
    sample_index = np.random.randint(num_samples, size=1)[0]

    # collect all the samples with this ID
    rows_to_keep = (dvar['sample_id'] == sample_index)
    dvar_sample = dvar[rows_to_keep].sort_values(by='Path length')
    qoi_true = qoi[rows_to_keep].sort_values(by='Path length')

    prediction_input = dvar_sample.drop(columns=['sample_id'])
    cols_of_prediction = qoi_true.drop(columns='Path length').columns

    # predict
    qoi_pred = model.predict(prediction_input.values)
    qoi_pred = pd.DataFrame(data=qoi_pred, columns=cols_of_prediction)

    to_return = []

    # sigma_x
    to_return.append(build_graph_dict(qoi_true['Path length'],
                                    qoi_true['RMS Beamsize in x'] * 1000.,
                                    qoi_pred['RMS Beamsize in x'] * 1000.,
                                    r'sigma_x [mm]',
                                    y_ranges['RMS Beamsize in x'].values * 1000.))

    # sigma_y
    to_return.append(build_graph_dict(qoi_true['Path length'],
                                    qoi_true['RMS Beamsize in y'] * 1000.,
                                    qoi_pred['RMS Beamsize in y'] * 1000.,
                                    r'sigma_y [mm]',
                                    y_ranges['RMS Beamsize in y'].values * 1000.))

    # sigma_s
    to_return.append(build_graph_dict(qoi_true['Path length'],
                                    qoi_true['RMS Beamsize in s'] * 1000.,
                                    qoi_pred['RMS Beamsize in s'] * 1000.,
                                    r'sigma_s [mm]',
                                    y_ranges['RMS Beamsize in s'].values * 1000.))

    # epsilon_x
    to_return.append(build_graph_dict(qoi_true['Path length'],
                                    qoi_true['Normalized Emittance x'] * 1000.,
                                    qoi_pred['Normalized Emittance x'] * 1000.,
                                    r'epsilon_x [mm rad]',
                                    y_ranges['Normalized Emittance x'] * 1000.))

    # epsilon_y
    to_return.append(build_graph_dict(qoi_true['Path length'],
                                    qoi_true['Normalized Emittance y'] * 1000.,
                                    qoi_pred['Normalized Emittance y'] * 1000.,
                                    r'epsilon_y [mm rad]',
                                    y_ranges['Normalized Emittance y'] * 1000.))

    # epsilon_s
    to_return.append(build_graph_dict(qoi_true['Path length'],
                                    qoi_true['Normalized Emittance s'] * 1000.,
                                    qoi_pred['Normalized Emittance s'] * 1000.,
                                    r'epsilon_s [mm rad]',
                                    y_ranges['Normalized Emittance s'] * 1000.))

    # corr(x, px)
    to_return.append(build_graph_dict(qoi_true['Path length'],
                                    qoi_true['Correlation xpx'],
                                    qoi_pred['Correlation xpx'],
                                    r'Corr(x, px)',
                                    y_ranges['Correlation xpx']))

    # corr(y, py)
    to_return.append(build_graph_dict(qoi_true['Path length'],
                                    qoi_true['Correlation ypy'],
                                    qoi_pred['Correlation ypy'],
                                    r'Corr(y, py)',
                                    y_ranges['Correlation ypy']))

    # corr(s, ps)
    to_return.append(build_graph_dict(qoi_true['Path length'],
                                    qoi_true['Correlation zpz'],
                                    qoi_pred['Correlation zpz'],
                                    r'Corr(s, ps)',
                                    y_ranges['Correlation zpz']))

    # energy
    to_return.append(build_graph_dict(qoi_true['Path length'],
                                    qoi_true['Mean Bunch Energy'],
                                    qoi_pred['Mean Bunch Energy'],
                                    r'E [MeV]',
                                    y_ranges['Mean Bunch Energy']))

    # energy spread
    to_return.append(build_graph_dict(qoi_true['Path length'],
                                    qoi_true['energy spread of the beam'],
                                    qoi_pred['energy spread of the beam'],
                                    r'dE [MeV]',
                                    y_ranges['energy spread of the beam']))

    # design variable table
    rows = build_table_rows(dvar_sample.iloc[0, :])
    to_return.append(rows)

    return to_return


if __name__ == '__main__':
    app.run_server(debug=True)
