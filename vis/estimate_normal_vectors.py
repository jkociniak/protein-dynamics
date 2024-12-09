import yaml
from base64 import b64decode
import io
import yaml
from dash.exceptions import PreventUpdate

import numpy as np
import plotly.graph_objs as go
import torch

from dash import Dash, dcc, html, callback_context, no_update
from dash.dependencies import Input, Output, State

from src.datasets.euclidean import DatasetRegistry
from src.datasets.euclidean.ambient_2d import *
from src.datasets.euclidean.mnist import *

from src.utils.normal_estimation import SmoothDistanceFunction, NormalEstimator
from src.utils.callbacks import SineExperimentsLogger

# mapping from dataset name to tuple (class, dim, params)
# the datasets are instantiated after selection from dropdown list
# DATASETS = {
#     'Custom3DDataset': (Custom3DDataset, dict()),
#     'Sphere3DDataset': (Sphere3DDataset, dict(n_points=100, polar_angle_range=(0, torch.pi/2), azimuth_range=(0, torch.pi/2))),
#     'Torus3D_100': (Torus3DDataset, dict(n_points=100)),
#     'Torus3D_500': (Torus3DDataset, dict(n_points=500)),
#     'StanfordBunny3D': (StanfordBunny3DDataset, dict()),
# }

print('Hello')
print(DatasetRegistry._datasets)
DATASETS = {}
for _, cls in DatasetRegistry._datasets.items():
    # assert cls has attribute param_sets
    for name, param_set in cls.param_sets.items():
        DATASETS[name] = (cls, param_set)

print(DATASETS)

app = Dash(__name__)

app.layout = html.Div([
    html.H1("SDF Level Set and Sample Optimization Visualization"),
    html.Div([
        html.Div([
            html.Label("SDF Radius (log scale)"),
            dcc.Slider(
                id='sdf-radius-slider',
                min=-2,
                max=-0.5,
                step=0.1,
                value=-1.6,
                marks={-2: '0.01', -1: '0.1', -0.5: '10**{-0.5}'},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 10px'}),
    ], style={'display': 'flex', 'justifyContent': 'center'}),
    html.Div([
        html.Div([
            html.Label("Noise scale for tubular samples"),
            dcc.Slider(
                id='tub-noise-scale-slider',
                min=0.01,
                max=1.,
                step=0.01,
                value=0.1,
                marks={i: str(i) for i in [0.01, 0.1, 0.5, 1.]}
            )
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 10px'}),
        html.Div([
            html.Label("Number of Noisy Samples per Point"),
            dcc.Slider(
                id='tub-num-noisy-samples-slider',
                min=1,
                max=50,
                step=1,
                value=5,
                marks={i: str(i) for i in range(1, 21, 4)}
            )
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 10px'})
    ], style={'display': 'flex', 'justifyContent': 'center'}),
    html.Div([
        html.Div([
            html.Label("Learning Rate"),
            dcc.Slider(
                id='tub-lr-slider',
                min=0.01,
                max=1.,
                step=0.01,
                value=0.1,
                marks={i: str(i) for i in [0.01, 0.1, 0.5, 1.]}
            )
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 10px'}),
        html.Div([
            html.Label("Optimization Steps"),
            dcc.Slider(
                id='tub-optimization-steps-slider',
                min=0,
                max=300,
                step=50,
                value=50,
                marks={i: str(i) for i in range(0, 301, 50)}
            )
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 10px'})
    ], style={'display': 'flex', 'justifyContent': 'center'}),
    html.Div([
        html.Div([
            html.Label("SDF Level Set Value"),
            dcc.Slider(
                id='tub-sdf-level-set-slider',
                min=-0.5,
                max=0.5,
                step=0.01,
                value=0,
                marks={i / 10: str(i / 10) for i in range(-5, 6, 1)}
            )
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 10px'}),
        html.Div([
            html.Label("SDF Level Set loss weight"),
            dcc.Slider(
                id='tub-sdf-loss-weight-slider',
                min=0.1,
                max=0.9,
                step=0.1,
                value=0.9,
                marks={i: str(i) for i in [0.1, 0.9]}
            )
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 10px'})
    ], style={'display': 'flex', 'justifyContent': 'center'}),
    html.Div([
        html.Div([
            html.Label("Boundary classification SDF threshold"),
            dcc.Slider(
                id='boundary-sdf-thr-slider',
                min=-0.5,
                max=0.5,
                step=0.01,
                value=-0.007,
                marks={i / 10: str(i / 10) for i in range(-5, 6, 1)}
            )
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 10px'}),
        html.Div([
            html.Label("Boundary classification neighbor threshold"),
            dcc.Slider(
                id='boundary-neighbor-thr-slider',
                min=0.1,
                max=0.9,
                step=0.1,
                value=0.9,
                marks={i: str(i) for i in [0.1, 0.9]}
            )
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 10px'})
    ], style={'display': 'flex', 'justifyContent': 'center'}),

    dcc.Dropdown(list(DATASETS.keys()), 'Custom3DDataset', id='datasets-dropdown'),
    html.Div([
        dcc.Upload(
            id='upload-config',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Config File (.yaml)')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px 0'
            },
            accept='.yaml,.yml'
        ),
        html.Div(id='upload-status', style={'margin': '10px 0', 'textAlign': 'center'})
    ]),
    html.Div([
        html.Button('Save Current Settings', id='save-button', n_clicks=0),
        html.Div(id='save-status', children='')
    ], style={'margin-top': '20px', 'text-align': 'center'}),

    html.Div([
        dcc.Graph(id='sdf-visualization-plot', style={'height': '60vh'}),
        dcc.Graph(id='normals-plot', style={'height': '60vh'}),
        # dcc.Graph(id='corrected-frames-plot', style={'height': '60vh'}),
        dcc.Graph(id='orthonormal-frames-plot', style={'height': '60vh'}),
        dcc.Graph(id='loss-curve-plot', style={'height': '30vh'}),
        # dcc.Graph(id='singular-values-plot', style={'height': '30vh'}),
        # dcc.Graph(id='explained-variance-plot', style={'height': '30vh'}),
        # dcc.Graph(id='estimated-dimension-plot', style={'height': '30vh'}),
        # dcc.Graph(id='categorized-points-plot', style={'height': '60vh'})
    ]),
    dcc.Store(id='points'),
    dcc.Store(id='tub-samples'),
    dcc.Store(id='tub-optimization-log')
])

@app.callback(
[
    Output('upload-status', 'children'),
    Output('sdf-radius-slider', 'value'),
    Output('tub-noise-scale-slider', 'value'),
    Output('tub-num-noisy-samples-slider', 'value'),
    Output('tub-lr-slider', 'value'),
    Output('tub-optimization-steps-slider', 'value'),
    Output('tub-sdf-level-set-slider', 'value'),
    Output('tub-sdf-loss-weight-slider', 'value'),
    Output('boundary-sdf-thr-slider', 'value'),
    Output('boundary-neighbor-thr-slider', 'value'),
    Output('points', 'data')],
[
    Input('datasets-dropdown', 'value'),
    Input('upload-config', 'contents'),
    State('upload-config', 'filename'),
]
)
def new_update_graph(dataset_name, contents, filename):
    print('RUNNING NEW UPDATE GRAPH')
    print(f'Dataset name: {dataset_name}')
    ctx = callback_context

    if not ctx.triggered or dataset_name is None:
        print('NO CONTEXT, FALLBACK TO DEFAULT')
        print('RUNNING BUILD DATASET "Default sine"')
        dataset_cls, params = DATASETS['Default sine']
        print(f'PARAMS {params}')
        # Create the dataset
        dataset = dataset_cls(**params)
        points = dataset[0].pos  # (n_points, ambient_dim)
        print(f'UWAGA UWAGA POINST SHAPE {points.shape}')

        values = [
            -1.6, 0.1, 5, 0.1, 50, 0, 0.9, -0.007, 0.9, points
        ]
        return [f"Successfully set default settings for SineDataset"] + values
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == "datasets-dropdown":
            # code for graph-selector original function goes here
            print(f'RUNNING BUILD DATASET {dataset_name}')
            dataset_cls, params = DATASETS[dataset_name]
            print(f'PARAMS {params}')
            # Create the dataset
            dataset = dataset_cls(**params)
            points = dataset[0].pos  # (n_points, ambient_dim)
            print(f'UWAGA UWAGA POINST SHAPE {points.shape}')

            values = [
                -1.6, 0.1, 5, 0.1, 50, 0, 0.9, -0.007, 0.9, points
            ]
            return [f"Successfully loaded default for {dataset_name}"] + values
        else:
            print('RUNNING UPDATE FROM CONFIG')
            if contents is None:
                raise PreventUpdate
            try:
                # Decode and parse the uploaded file
                content_type, content_string = contents.split(',')
                decoded = b64decode(content_string)
                config = yaml.safe_load(io.StringIO(decoded.decode('utf-8')))

                # Extract dataset information
                dataset_name = config.pop('_target_').split('.')[-1]
                if dataset_name not in DATASETS:
                    return [f"Error: Unknown dataset type {dataset_name}"] + [no_update] * 10
                dataset_cls, _ = DATASETS[dataset_name]

                # Extract normal estimation config
                ne_cfg = config.pop('normal_estimation_cfg', {})
                boundary_cfg = ne_cfg.pop('boundary_cfg', {})
                print(f'PARAMS {config}')
                # Create the dataset
                dataset = dataset_cls(**config)
                points = dataset[0].pos  # (n_points, ambient_dim)

                # Prepare all values, using current slider values as defaults
                values = [
                    ne_cfg.get('log_sdf_radius', -1.6),  # Default from original slider
                    ne_cfg.get('noise_scale', 0.1),
                    ne_cfg.get('num_noisy_samples', 5),
                    ne_cfg.get('lr', 0.1),
                    ne_cfg.get('optimization_steps', 50),
                    ne_cfg.get('sdf_level_set', 0),
                    ne_cfg.get('sdf_loss_weight', 0.9),
                    boundary_cfg.get('boundary_sdf_thr', -0.007),
                    boundary_cfg.get('boundary_neighbor_thr', 0.9),
                    points
                ]

                return [f"Successfully loaded config from {filename}"] + values

            except Exception as e:
                return [f"Error loading config: {str(e)}"] + [no_update] * 10


def plot_level_set_3d(points, sdf, target_value):
    X, Y, Z = compute_mesh_3d(points)
    grid_points = torch.tensor(np.column_stack((X.ravel(), Y.ravel(), Z.ravel())), dtype=torch.float32)
    sdf_values = sdf(grid_points).detach().numpy().reshape(X.shape)
    # Create the level set plot
    level_set = go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=sdf_values.flatten(),
        isomin=target_value,
        isomax=target_value,
        opacity=0.5,
        colorscale='Viridis',
        name=f'SDF Level Set ({target_value:.2f})'
    )
    return level_set


def plot_level_set_2d(points, sdf, target_value):
    X, Y = compute_mesh_2d(points)
    grid_points = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)
    sdf_values = sdf(grid_points).detach().numpy().reshape(X.shape)

    contour = go.Contour(
        x=X.flatten(),
        y=Y.flatten(),
        z=sdf_values.flatten(),
        contours=dict(
            start=target_value,
            end=target_value,
            size=0,
            coloring='lines'
        ),
        name=f'SDF Level Set ({target_value:.2f})'
    )
    return contour


def compute_mesh_3d(points, margin=0.1):
    assert isinstance(margin, float) and margin >= 0
    # Create a grid of points for the level set
    x = np.linspace(points[:, 0].min() - margin, points[:, 0].max() + margin, 50)
    y = np.linspace(points[:, 1].min() - margin, points[:, 1].max() + margin, 50)
    z = np.linspace(points[:, 2].min() - margin, points[:, 2].max() + margin, 50)
    X, Y, Z = np.meshgrid(x, y, z)
    return X, Y, Z


def compute_mesh_2d(points, margin=0.1):
    assert isinstance(margin, float) and margin >= 0
    # Create a grid of points for the level set
    x = np.linspace(points[:, 0].min() - margin, points[:, 0].max() + margin, 50)
    y = np.linspace(points[:, 1].min() - margin, points[:, 1].max() + margin, 50)
    X, Y = np.meshgrid(x, y)
    return X, Y


@app.callback(
    Output('save-status', 'children'),
    Input('save-button', 'n_clicks'),
    [State('sdf-radius-slider', 'value'),
     State('tub-noise-scale-slider', 'value'),
     State('tub-num-noisy-samples-slider', 'value'),
     State('tub-lr-slider', 'value'),
     State('tub-optimization-steps-slider', 'value'),
     State('tub-sdf-level-set-slider', 'value'),
     State('tub-sdf-loss-weight-slider', 'value'),
     State('boundary-sdf-thr-slider', 'value'),
     State('boundary-neighbor-thr-slider', 'value'),
     State('datasets-dropdown', 'value')]
)
def save_settings(n_clicks, log_sdf_radius, tub_noise_scale, tub_num_noisy_samples,
                  tub_lr, tub_optimization_steps, tub_sdf_level_set, tub_sdf_loss_weight,
                  boundary_sdf_thr, boundary_neighbor_thr, dataset_name):
    if n_clicks > 0 and callback_context.triggered[0]['prop_id'] == 'save-button.n_clicks':
        config = {
            '_target_': f'src.datasets.euclidean.{dataset_name}',
            'normal_estimation_cfg': {
                'log_sdf_radius': log_sdf_radius,
                'noise_scale': tub_noise_scale,
                'num_noisy_samples': tub_num_noisy_samples,
                'optimization_steps': tub_optimization_steps,
                'lr': tub_lr,
                'sdf_level_set': tub_sdf_level_set,
                'sdf_loss_weight': tub_sdf_loss_weight,
                'boundary_cfg': {
                    'boundary_sdf_thr': boundary_sdf_thr,
                    'boundary_neighbor_thr': boundary_neighbor_thr
                }
            }
        }

        # Add dataset-specific parameters if needed
        if dataset_name in ['Disk2DDataset']:
            config.update({
                'n_points': 100,
                'center': True,
                'seed': 42
            })

        filename = f'{dataset_name.lower()}_config.yaml'
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return f'Settings saved to {filename}'

    return ''


# @app.callback(
#     Output('save-status', 'children'),
#     Input('save-button', 'n_clicks'),
#     [State('sdf-radius-slider', 'value'),
#      State('tub-noise-scale-slider', 'value'),
#      State('tub-num-noisy-samples-slider', 'value'),
#      State('tub-lr-slider', 'value'),
#      State('tub-optimization-steps-slider', 'value'),
#      State('tub-sdf-level-set-slider', 'value'),
#      State('tub-sdf-loss-weight-slider', 'value'),
#      State('boundary-sdf-thr-slider', 'value'),
#      State('boundary-neighbor-thr-slider', 'value')]
# )
# def save_settings(n_clicks, log_sdf_radius, tub_noise_scale, tub_num_noisy_samples,
#                   tub_lr, tub_optimization_steps, tub_sdf_level_set, tub_sdf_loss_weight,
#                   boundary_sdf_thr, boundary_neighbor_thr):
#     if n_clicks > 0 and callback_context.triggered[0]['prop_id'] == 'save-button.n_clicks':
#         normal_estimation_cfg = dict(
#             log_sdf_radius=log_sdf_radius,
#             noise_scale=tub_noise_scale,
#             num_noisy_samples=tub_num_noisy_samples,
#             optimization_steps=tub_optimization_steps,
#             lr=tub_lr,
#             sdf_level_set=tub_sdf_level_set,
#             sdf_loss_weight=tub_sdf_loss_weight,
#             boundary_cfg=dict(boundary_sdf_thr=boundary_sdf_thr,
#                               boundary_neighbor_thr=boundary_neighbor_thr)
#             # how many % of tubular samples must lie "outside" to categorize point a boundary)
#         )
#
#         filename = 'normal_estimation_settings.yaml'
#         with open(filename, 'w') as f:
#             yaml.dump(normal_estimation_cfg, f, default_flow_style=False)
#
#         return f'Settings saved to {filename}'
#
#     return ''


@app.callback(
    [Output('sdf-visualization-plot', 'figure'),
     Output('tub-samples', 'data'),
     Output('tub-optimization-log', 'data')],
    [Input('sdf-radius-slider', 'value'),
     Input('tub-noise-scale-slider', 'value'),
     Input('tub-num-noisy-samples-slider', 'value'),
     Input('tub-lr-slider', 'value'),
     Input('tub-optimization-steps-slider', 'value'),
     Input('tub-sdf-level-set-slider', 'value'),
     Input('tub-sdf-loss-weight-slider', 'value'),
     Input('points', 'data')
     ]
)
def update_plot(log_sdf_radius,
                tub_noise_scale, tub_num_noisy_samples,
                tub_lr, tub_optimization_steps, tub_sdf_level_set, tub_sdf_loss_weight, points):
    print('RUNNING UPDATE PLOT')
    points = torch.from_numpy(np.array(points, dtype=np.float32))

    # Convert log scale to actual radius
    sdf_radius = 10 ** log_sdf_radius
    sdf = SmoothDistanceFunction(points, radius=sdf_radius)

    normal_estimation_cfg = dict(
        noise_scale=tub_noise_scale,
        num_noisy_samples=tub_num_noisy_samples,
        lr=tub_lr,
        optimization_steps=tub_optimization_steps,
        sdf_level_set=tub_sdf_level_set,
        sdf_loss_weight=tub_sdf_loss_weight,
    )

    tubular_samples, tub_optimization_log = NormalEstimator.estimate_tubular_points(sdf, points,
                                                                                     **normal_estimation_cfg)

    dim = points.shape[-1]
    if dim == 3:
        scatter = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(color='black', size=3),
            name='Dataset Points'
        )
        level_set = plot_level_set_3d(points, sdf, tub_sdf_level_set)

        bs = tubular_samples.shape
        tubular_samples = tubular_samples.reshape(-1, tubular_samples.shape[-1])

        # Create scatter plot for tubular samples
        tubular_scatter = go.Scatter3d(
            x=tubular_samples[:, 0].numpy(),
            y=tubular_samples[:, 1].numpy(),
            z=tubular_samples[:, 2].numpy(),
            mode='markers',
            marker=dict(color='red', size=2),
            name='Optimized Samples'
        )

        # Combine all traces
        fig = go.Figure(data=[scatter, level_set, tubular_scatter])

        # Update layout
        fig.update_layout(
            title=f'SDF Level Set and Sample Optimization (Radius: {sdf_radius:.2e}, Level: {tub_sdf_level_set:.2f})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            legend=dict(
                x=1.05,
                y=0.5,
                xanchor='left',
                yanchor='middle',
                bgcolor='rgba(255, 255, 255, 0.5)'
            ),
            margin=dict(r=150, l=10, t=50, b=10)
        )

        # Adjust colorbar position
        fig.update_traces(
            selector=dict(type='isosurface'),
            colorbar=dict(
                x=1,
                y=0.5,
                len=0.5,
                yanchor='middle'
            )
        )
        tubular_samples = tubular_samples.reshape(*bs)
    elif dim == 2:
        scatter = go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers',
            marker=dict(color='black', size=3),
            name='Dataset Points'
        )
        level_set = plot_level_set_2d(points, sdf, tub_sdf_level_set)

        bs = tubular_samples.shape
        tubular_samples = tubular_samples.reshape(-1, tubular_samples.shape[-1])

        # Create scatter plot for tubular samples
        tubular_scatter = go.Scatter(
            x=tubular_samples[:, 0].numpy(),
            y=tubular_samples[:, 1].numpy(),
            mode='markers',
            marker=dict(color='red', size=2),
            name='Optimized Samples'
        )

        # Combine all traces
        fig = go.Figure(data=[scatter, level_set, tubular_scatter])

        # Update layout
        fig.update_layout(
            title=f'SDF Level Set and Sample Optimization (Radius: {sdf_radius:.2e}, Level: {tub_sdf_level_set:.2f})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                aspectmode='data'
            ),
            legend=dict(
                x=1.05,
                y=0.5,
                xanchor='left',
                yanchor='middle',
                bgcolor='rgba(255, 255, 255, 0.5)'
            ),
            margin=dict(r=150, l=10, t=50, b=10)
        )

        # # Adjust colorbar position
        # fig.update_traces(
        #     selector=dict(type='isosurface'),
        #     colorbar=dict(
        #         x=1,
        #         y=0.5,
        #         len=0.5,
        #         yanchor='middle'
        #     )
        # )
        tubular_samples = tubular_samples.reshape(*bs)
    else:
        raise ValueError('only 2d and 3d data is supported')

    return fig, tubular_samples, tub_optimization_log


@app.callback(
    [Output('loss-curve-plot', 'figure')],
    [Input('tub-optimization-log', 'data')]
)
def update_tubular_loss_plot(optimization_info):
    sdf_losses = optimization_info['sdf_losses']
    l2_losses = optimization_info['l2_losses']
    total_losses = optimization_info['total_losses']

    # Create loss curve plot with both losses and their weighted sum
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=list(range(len(sdf_losses))), y=sdf_losses, mode='lines', name='SDF Loss'))
    loss_fig.add_trace(go.Scatter(x=list(range(len(l2_losses))), y=l2_losses, mode='lines', name='L2 Loss'))
    loss_fig.add_trace(go.Scatter(x=list(range(len(total_losses))), y=total_losses, mode='lines', name='Total Loss'))

    loss_fig.update_layout(
        title=f'Optimization Loss Curves (SDF Weight: 0.9)',
        xaxis_title='Optimization Step',
        yaxis_title='Loss',
        yaxis_type='log'  # Using log scale for better visualization
    )
    return loss_fig,


@app.callback(
    [Output('normals-plot', 'figure'),
     Output('orthonormal-frames-plot', 'figure')],
    [Input('tub-samples', 'data'),
     Input('sdf-radius-slider', 'value'),
     Input('boundary-sdf-thr-slider', 'value'),
     Input('boundary-neighbor-thr-slider', 'value'),
     Input('points', 'data')]
)
def update_normals_fig(tubular_samples, log_sdf_radius, boundary_sdf_thr, boundary_neighbor_thr, points):
    print('RUNNING UPDATE NORMALS FIG')
    points = np.array(points, dtype=np.float32)
    torch.save(points, 'points_est.pt')

    # Convert log scale to actual radius
    if tubular_samples is None:
        return go.Figure(), go.Figure()

    sdf_radius = 10 ** log_sdf_radius
    sdf = SmoothDistanceFunction(torch.tensor(points, dtype=torch.float32), radius=sdf_radius)

    tubular_samples = torch.from_numpy(np.array(tubular_samples, dtype=np.float32))

    boundary_mask = NormalEstimator.estimate_boundary(sdf, tubular_samples, boundary_sdf_thr, boundary_neighbor_thr)
    print(boundary_mask)
    b_normals = NormalEstimator.extract_normal_vectors(sdf, tubular_samples[boundary_mask])
    b_points = points[boundary_mask]

    # Create arrow plot for normal vectors
    normals_fig = go.Figure()

    on_frames, singular_values, estimated_dimensions = NormalEstimator.estimate_dimensionality(b_normals, threshold=0.85)
    print(estimated_dimensions)

    # frames = {i: row for i, row in enumerate(normalized_gradients[0])}
    # singular_values_all, estimated_dimensions = estimate_dimensionality(frames)
    #
    # # Create color array based on estimated dimensions
    color_map = {1: 'red', 2: 'green', 3: 'blue'}
    colors = [color_map[i] for i in estimated_dimensions]

    # Repeat colors for each noisy sample
    colors = np.repeat(colors, b_normals.shape[1])

    b_points_expanded = np.broadcast_to(b_points[:, None, :], b_normals.shape)
    dim = b_points.shape[-1]
    if dim == 3:
        # Add unsmoothed normal vectors with colors
        unsmoothed_arrows = SineExperimentsLogger.create_arrow_plot_3d(
            b_points_expanded.reshape(-1, b_points.shape[-1]),
            b_normals.reshape(-1, b_normals.shape[-1]),
            colors,
            name='Normal Vectors'
        )
        normals_fig.add_traces(unsmoothed_arrows)

        # Add the dataset points
        normals_fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(color='black', size=3),
            name='Dataset Points'
        ))

        normals_fig.update_layout(
            title='Normal Vectors Visualization (Colored by Estimated Dimension)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            legend=dict(x=1.05, y=0.5)
        )

        # Create the orthonormal frames plot
        orthonormal_frames_fig = go.Figure()

        colors = ['red', 'green', 'blue']
        for i in range(3):
            arrows = SineExperimentsLogger.create_arrow_plot_3d(
                b_points,
                on_frames[:, i, :],
                [colors[i]] * b_points.shape[0],
                name=f'Frame Component {i + 1}'
            )
            orthonormal_frames_fig.add_traces(arrows)

        orthonormal_frames_fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(color='black', size=3),
            name='Dataset Points'
        ))

        orthonormal_frames_fig.update_layout(
            title='Orthonormal Frames Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            legend=dict(x=1.05, y=0.5)
        )
    elif dim == 2:
        # Add unsmoothed normal vectors with colors
        unsmoothed_arrows = SineExperimentsLogger.create_arrow_plot_2d(
            b_points_expanded.reshape(-1, b_points.shape[-1]),
            b_normals.reshape(-1, b_normals.shape[-1]),
            colors,
            name='Normal Vectors'
        )
        normals_fig.add_traces(unsmoothed_arrows)

        # Add the dataset points
        normals_fig.add_trace(go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers',
            marker=dict(color='black', size=3),
            name='Dataset Points'
        ))

        normals_fig.update_layout(
            title='Normal Vectors Visualization (Colored by Estimated Dimension)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                aspectmode='data'
            ),
            legend=dict(x=1.05, y=0.5)
        )

        # Create the orthonormal frames plot
        orthonormal_frames_fig = go.Figure()

        colors = ['red', 'green']
        for i in range(2):
            arrows = SineExperimentsLogger.create_arrow_plot_2d(
                b_points,
                on_frames[:, i, :],
                [colors[i]] * b_points.shape[0],
                name=f'Frame Component {i + 1}'
            )
            orthonormal_frames_fig.add_traces(arrows)

        orthonormal_frames_fig.add_trace(go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers',
            marker=dict(color='black', size=3),
            name='Dataset Points'
        ))

        orthonormal_frames_fig.update_layout(
            title='Orthonormal Frames Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                aspectmode='data'
            ),
            legend=dict(x=1.05, y=0.5)
        )
    else:
        raise ValueError('only 2d and 3d is supported')

    return normals_fig, orthonormal_frames_fig


if __name__ == '__main__':
    app.run_server(debug=True)
