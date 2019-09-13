import os
import numpy as np
import pandas as pd

import __init_path__
import env

angles_columns = ['euler_x', 'euler_y', 'euler_z']
translations_columns = ['t_x', 't_y', 't_z']


def generate_vertex(from_index,
                    to_index,
                    translation=(0, 0, 0),
                    translation_std=(1, 1, 1),
                    rotation=(0, 0, 0),
                    rotation_std=(1, 1, 1)):
    data = dict()
    for column_index, column in enumerate(angles_columns):
        data[column] = [rotation[column_index]]
        data[column + '_confidence'] = [rotation_std[column_index]]

    for column_index, column in enumerate(translations_columns):
        data[column] = [translation[column_index]]
        data[column + '_confidence'] = [translation_std[column_index]]

    data['path_to_rgb'] = [f'{from_index}.png']
    data['path_to_rgb_next'] = [f'{to_index}.png']
    return pd.DataFrame(data)


def generate_rectangle(edge_lengths, translations, translations_std):

    df = pd.DataFrame()
    for edge_index in range(4):
        start_index = edge_index * edge_lengths[edge_index]
        stop_index = (edge_index + 1) * edge_lengths[edge_index] - 1

        for vertex_index in range(start_index, stop_index):
            df = df.append(generate_vertex(from_index=vertex_index,
                                           to_index=vertex_index+1,
                                           translation=translations[edge_index],
                                           translation_std=translations_std[edge_index]))

        df = df.append(generate_vertex(from_index=stop_index,
                                       to_index=stop_index+1,
                                       translation=translations[edge_index],
                                       translation_std=translations_std[edge_index],
                                       rotation=(0, -np.pi/2, 0),
                                       rotation_std=(1, 1, 1)))
    return df


def generate_square_loop():
    edge_lengths = [300, 300, 300, 300]
    translations = [(1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)]
    translations_std = [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)]
    gt_df = generate_rectangle(edge_lengths, translations, translations_std)
    gt_df.to_csv(os.path.join(env.PROJECT_PATH, 'tests/minidataset/toy/square_loop_gt.csv'))

    translations = [(1, 0, 0), (1, 0, -0.1), (1, 0, 0), (1, 0, 0)]
    translations_std = [(1, 1, 1), (2, 2, 2), (1, 1, 1), (1, 1, 1)]
    noised_df = generate_rectangle(edge_lengths, translations, translations_std)
    noised_df = noised_df.append(generate_vertex(from_index=sum(edge_lengths) - 1,
                                                 to_index=0,
                                                 rotation=(0, -np.pi/2, 0)))
    noised_df.to_csv(os.path.join(env.PROJECT_PATH, 'tests/minidataset/toy/square_loop_predict.csv'))


if __name__ == "__main__":
    generate_square_loop()
