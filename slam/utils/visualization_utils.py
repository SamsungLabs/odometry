import os
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot, iplot


def make_trace_and_start(xyz, is_gt, showlegend=True, is_3d=True):
    trace_color = 'orange' if is_gt else 'blue'
    if is_3d:
        trace_pred = go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode='lines+markers',
            marker={'size': 1, 'color': trace_color},
            line={'width': 1, 'color': trace_color},
            name='ground truth' if is_gt else 'prediction',
            showlegend=showlegend)

        start_pred = go.Scatter3d(
            x=[xyz[0, 0]],
            y=[xyz[0, 1]],
            z=[xyz[0, 2]],
            mode='markers',
            marker={'size': 3, 'color': 'red'},
            showlegend=False)
    else:
        trace_pred = go.Scatter(
            x=xyz[:, 0],
            y=xyz[:, 2],
            mode='lines+markers',
            marker={'size': 3, 'color': trace_color},
            line={'width': 3, 'color': trace_color},
            name='ground truth' if is_gt else 'prediction',
            showlegend=showlegend)

        start_pred = go.Scatter(
            x=[xyz[0, 0]],
            y=[xyz[0, 2]],
            mode='markers',
            marker={'size': 6, 'color': 'red'},
            showlegend=False)

    return trace_pred, start_pred


def init_figure(cols=3, is_3d=True):
    fig = plotly.tools.make_subplots(
        rows=1,
        cols=cols,
        specs=[[{'is_3d': is_3d}] * cols],
        print_grid=False
    )
    return fig


def get_scene_id(index):
    if index == 0:
        return 'scene'
    return f'scene{index + 1}'


def get_axis_id(index):
    if index == 0:
        return 'axis'
    return f'axis{index + 1}'
    

def update_figure(fig, values, cols=3, is_3d=True):
    left_coord, right_coord = np.min(values), np.max(values)
    border = np.abs(right_coord - left_coord) / 10

    if is_3d:
        axis_dict = {'xaxis': {'range': [left_coord - border, right_coord + border]},
                     'yaxis': {'range': [left_coord - border, right_coord + border]},
                     'zaxis': {'range': [left_coord - border, right_coord + border]}}
    else:
        axis_dict = {'range': [left_coord - border, right_coord + border],
                     'showgrid': False,
                     'zeroline': False}

    if is_3d:
        scenes = [get_scene_id(col) for col in range(cols)]
        fig['layout'].update({scene: axis_dict for scene in scenes})
        for scene in scenes:
            fig['layout'][scene].update(go.layout.Scene(aspectmode='cube'))
    else:
        axes = [get_axis_id(col) for col in range(cols)]
        for ax in axes:
            for axis_type in ('x', 'y'):
                fig['layout'][axis_type + ax].update(axis_dict)

    return fig


def save_figure(fig, title, cols=3, is_3d=True, file_path=None):
    fig['layout'].update(title=title, height=1000)
    
    if is_3d:
        scenes = [get_scene_id(index) for index in range(cols)]
        x_dom = [fig['layout'][scene]['domain']['x'] for scene in scenes]
        y_dom = [fig['layout'][scene]['domain']['y'] for scene in scenes]
    else:
        axes = [get_axis_id(index) for index in range(cols)]
        x_dom = [fig['layout']['x' + ax]['domain'] for ax in axes]
        y_dom = [fig['layout']['y' + ax]['domain'] for ax in axes]

    if file_path is not None:
        plot(fig, filename=file_path)
    else:
        init_notebook_mode(connected=True)
        iplot(fig)


def append_multiple_traces_to_figure(fig, traces, row, col):
    for trace in traces:
        fig.append_trace(trace, row, col)
    return fig


def visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory, title='', is_3d=True, file_path=None):
    predicted_aligned_trajectory = predicted_trajectory.align_with(gt_trajectory, by='start')
    
    gt_trajectory_points = gt_trajectory.points
    predicted_trajectory_points = predicted_trajectory.points
    predicted_aligned_trajectory_points = predicted_aligned_trajectory.points
    
    # predicted trajectory
    predicted_trace, predicted_start = make_trace_and_start(predicted_trajectory_points, is_gt=False, is_3d=is_3d)
    # groundtruth trajectory
    gt_trace, gt_start = make_trace_and_start(gt_trajectory_points, is_gt=True, is_3d=is_3d)
    # both, aligned
    predicted_aligned_trace, predicted_aligned_start = \
        make_trace_and_start(predicted_aligned_trajectory_points, is_gt=False, is_3d=is_3d)

    fig = init_figure(cols=3, is_3d=is_3d)
    
    fig = append_multiple_traces_to_figure(fig, [predicted_trace, predicted_start], 1, 1)
    fig = append_multiple_traces_to_figure(fig, [gt_trace, gt_start], 1, 2)
    fig = append_multiple_traces_to_figure(fig, [gt_trace, gt_start, predicted_aligned_trace, predicted_aligned_start], 1, 3)
    
    values = np.stack([gt_trajectory_points, predicted_trajectory_points, predicted_aligned_trajectory_points])
    if not is_3d:
        values = values[:, np.array((0, 2, 3, 5))] # select values corresponding to 2d motion 
    fig = update_figure(fig, values, is_3d=is_3d)
    save_figure(fig, title, cols=3, is_3d=is_3d, file_path=file_path)


def visualize_trajectory(trajectory, title='', is_gt=False, is_3d=True, file_path=None):
    points = trajectory.points
    trace, start = make_trace_and_start(points, is_gt=False, is_3d=is_3d)
    
    fig = init_figure(cols=1, is_3d=is_3d)
    fig.append_trace(trace, 1, 1)
    fig.append_trace(start, 1, 1)
    fig = update_figure(fig, values=points, cols=1, is_3d=is_3d)
    save_figure(fig, title, cols=1, is_3d=is_3d, file_path=file_path)
