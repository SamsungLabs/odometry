import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from pyquaternion import Quaternion
from statistics import mean, median

from slam.linalg import (convert_euler_angles_to_rotation_matrix,
                         GlobalTrajectory,
                         RelativeTrajectory,
                         euler_to_quaternion)


class DatasetStat:
    def __init__(self):
        self.rotation_columns = ['euler_x', 'euler_y', 'euler_z']
        self.translation_columns = ['t_x', 't_y', 't_z']
        self.mean_cols = self.rotation_columns + self.translation_columns
        self.std_cols = [c + '_confidence' for c in self.mean_cols]
        self.vis_columns = ['translation_distance',
                            'rotation_distance'] + self.translation_columns + self.rotation_columns
        self.dataframe_types = ['all', 'consecutive_fr', 'consecutive_kfr', 'loops']
        colors = ['b', 'r', 'g', 'y', 'c']
        self.colors = {k: v for k, v in zip(self.dataframe_types + ['pred'], colors)}
        self.summary_keys = ['frames_total', 'cons_fr_num', 'loop_fr_num', 'unique_loop_fr_num', 'loops_num']
        self.percentiles = [100, 95, 75]

    def append_translation_stat(self, relative_df, global_df):
        to_translation = global_df[self.translation_columns].values[relative_df.to_index.values.astype('int')]
        from_translation = global_df[self.translation_columns].values[relative_df.from_index.values.astype('int')]
        distance = np.linalg.norm(to_translation - from_translation, axis=1) * np.sign(relative_df['t_z'].values)
        relative_df['translation_distance'] = distance
        return relative_df

    def append_rotation_stat(self, relative_df, global_df):
        to_rotation = global_df[self.rotation_columns].values[relative_df.to_index.values.astype('int')]
        from_rotation = global_df[self.rotation_columns].values[relative_df.from_index.values.astype('int')]
        quaternions_from = [Quaternion(euler_to_quaternion(euler)).unit for euler in from_rotation]
        quaternions_to = [Quaternion(euler_to_quaternion(euler)).unit for euler in to_rotation]
        quaternions_pairs = zip(quaternions_to, quaternions_from)
        relative_df['rotation_distance'] = [Quaternion.distance(*pair) for pair in quaternions_pairs]
        return relative_df

    def get_pair_frame_stat(self, relative_df):
        is_adjustment = (relative_df.to_index - relative_df.from_index) <= 1
        consecutive_measurements = relative_df[is_adjustment].reset_index(drop=True)
        gt_trajectory = RelativeTrajectory.from_dataframe(consecutive_measurements).to_global()
        global_df = gt_trajectory.to_dataframe()
        relative_df = self.append_translation_stat(relative_df, global_df)
        relative_df = self.append_rotation_stat(relative_df, global_df)
        return relative_df

    def df2slam_predict(self, gt):
        predict = gt[self.mean_cols]

        for std_col in self.std_cols:
            if std_col in gt.columns:
                predict[std_col] = gt[std_col]
            else:
                predict[std_col] = 1

        if 'to_path' in gt.columns:
            predict['to_index'] = gt['to_index']
            predict['from_index'] = gt['from_index']
        elif 'path_to_rgb_next' in gt.columns:
            stem_function = lambda x: int(Path(x).stem)
            predict['to_index'] = gt['path_to_rgb_next'].apply(stem_function)
            predict['from_index'] = gt['path_to_rgb'].apply(stem_function)
        else:
            predict['to_index'] = np.arange(1, len(gt) + 1)
            predict['from_index'] = np.arange(0, len(gt))
        return predict

    def get_trajectory_stat(self, path_to_csv, loop_threshold, keyframe_period, trajectory_id):
        stat = self.init_data()

        pair_frame_df = pd.read_csv(path_to_csv)
        if len(pair_frame_df.columns) < 6:
            return stat

        pair_frame_df = self.df2slam_predict(pair_frame_df)
        pair_frame_df = pair_frame_df.drop_duplicates(subset=['to_index', 'from_index'])
        pair_frame_df = self.get_pair_frame_stat(pair_frame_df)
        pair_frame_df['trajectory_id'] = trajectory_id
        stat['all'] = pair_frame_df

        index_difference = pair_frame_df.to_index - pair_frame_df.from_index
        consecutive_fr = pair_frame_df[index_difference == 1].reset_index(drop=True)
        stat['consecutive_fr'] = consecutive_fr

        is_loop = index_difference >= loop_threshold
        loops = pair_frame_df[is_loop].reset_index(drop=True)
        stat['loops'] = loops

        is_periodic = (index_difference >= keyframe_period) & (index_difference < loop_threshold)
        consecutive_kfr = pair_frame_df[is_periodic].reset_index(drop=True)
        stat['consecutive_kfr'] = consecutive_kfr

        stat['summary']['frames_total'] = len(pair_frame_df)
        stat['summary']['cons_fr_num'] = len(consecutive_fr)

        if len(loops) == 0:
            return stat

        stat['summary']['loop_fr_num'] = len(loops)
        stat['summary']['unique_loop_fr_num'] = len(loops.to_index.unique()) if len(loops) > 0 else 0

        loops_num = 1
        previous_index = loops['to_index'].values[0]
        for index, row in loops.iterrows():
            if row.to_index - previous_index > 3 * keyframe_period:
                loops_num += 1
            previous_index = row.to_index

        stat['summary']['loops_num'] = loops_num
        return stat

    def filter_outlier(self, x, percentile, get_indices=False):
        if len(x) == 0:
            if get_indices:
                return np.array(()), np.array(())
            else:
                return np.array(())

        min_percentile, max_percentile = (100 - percentile) / 2, (100 + percentile) / 2
        min_threshold = np.percentile(x, min_percentile)
        if np.isnan(min_threshold):
            min_threshold = min(x)

        max_threshold = np.percentile(x, max_percentile)
        if np.isnan(max_threshold):
            max_threshold = max(x)

        indices = np.squeeze((min_threshold <= x) & (x <= max_threshold))

        if get_indices:
            return x[np.where(indices)], indices
        else:
            return x[np.where(indices)]

    def filter_pairs(self, stat, other_stat):
        stat = stat.set_index(['from_index', 'to_index', 'trajectory_id'])
        other_stat = other_stat.set_index(['from_index', 'to_index', 'trajectory_id'])
        index = [ix for ix in stat.index if ix in other_stat.index]
        stat = stat.loc[index].reset_index()
        other_stat = other_stat.loc[index].reset_index()
        return stat, other_stat

    def plot_hist(self, ax, gt_stat, pred_stat, column, percentile, title):

        y = list()
        legend = list()
        colors = list()
        total_max = float('-inf')
        total_min = float('inf')

        for index, key in enumerate(self.dataframe_types):
            gt = self.filter_outlier(gt_stat[key][column].values, percentile)
            if len(gt) > 0:
                y_median = median(gt)
                y_max = max(gt)
                y_min = min(gt)
                #                 print('gt', gt.shape)
                #                 print('gt_median', y_median.shape)
                #                 print('gt_max', y_max.shape)
                #                 print('gt_min', y_min.shape)
                legend.append(f'gt {key}. [{y_min:.2}, {y_max:.2}] m={y_median:.2}.')
                colors.append(self.colors[key])
                total_max = max([y_max, total_max])
                total_min = min([y_min, total_min])
                y.append(gt)

        pred = self.filter_outlier(pred_stat['all'][column].values, percentile)
        if len(pred) > 0:
            y_median = median(pred)
            y_max = max(pred)
            y_min = min(pred)
            legend.append(f'Pred all. [{y_min:.2}, {y_max:.2}] m={y_median:.2}.')
            colors.append(self.colors['pred'])
            total_max = max([y_max, total_max])
            total_min = min([y_min, total_min])
            y.append(pred)

        if len(y) == 0:
            return None

        _, bins, _ = ax.hist(y, normed=True, color=colors)
        ax.legend(legend)
        ax.set_title(title)
        ax.set_ylabel('normalized number of samples')
        ax.set_xlabel('x')
        ax.grid()
        if np.isinf(total_min) or np.isinf(total_max):
            return bins

        return bins

    def plot_error(self, ax, gt_stat, pred_stat, bins, column, percentile, title):

        if bins is None:
            return

        markers = ['bo', 'r+', 'gx', 'y8']
        legend = list()
        for df_index, key in enumerate(self.dataframe_types):
            gt_stat[key], pred_stat[key] = self.filter_pairs(pred_stat[key], gt_stat[key])
            gt = gt_stat[key][column].values
            pred = pred_stat[key][column].values

            assert len(gt) == len(pred), f'len(gt)={len(gt)}, len(pred)={len(pred)}'

            __, normal_indices = self.filter_outlier(gt, percentile, get_indices=True)

            if len(normal_indices) == 0:
                continue

            abs_diff = np.abs(gt - pred)
            bin_indices = np.digitize(gt, bins[:-2])

            if key == 'loops' and column == 'translation_distance' and percentile == 75:
                print('bin_indices', bin_indices)
            line = np.zeros(len(bins) - 1)
            for i in range(0, len(bins)):
                indices = (bin_indices == i) & normal_indices

                if np.any(indices):
                    line[i] = median(abs_diff[indices])

            indices = np.where(line != 0)

            x = bins[:-1] + (bins[1:] - bins[:-1]) / 2
            ax.plot(x[indices], line[indices], markers[df_index], linestyle='solid')
            ax.set_yscale('log')
            legend.append(key)

        ax.legend(legend)
        ax.set_title(title)
        ax.set_ylabel('median of |error|')
        ax.set_xlabel('gt value')
        ax.axis(xmin=bins[0], xmax=bins[-1])
        ax.grid()

    def plot(self, gt_stat, pred_stat, title):
        num_of_cols = len(self.vis_columns)
        num_of_rows = len(self.percentiles) * 2

        f, axes = plt.subplots(num_of_rows, num_of_cols, figsize=(6 * num_of_cols, 32))
        f.suptitle(title, fontsize=16)

        for index, column in enumerate(self.vis_columns):
            for p_index, p in enumerate(self.percentiles):
                ax_1 = axes[(index // num_of_cols) * 6 + p_index * 2, index % num_of_cols]
                ax_2 = axes[(index // num_of_cols) * 6 + p_index * 2 + 1, index % num_of_cols]
                bins = self.plot_hist(ax_1, gt_stat, pred_stat, column, p, title=f'{column}_{p}_percentile')
                self.plot_error(ax_2, gt_stat, pred_stat, bins, column, p, title=f'{column}_error_{p}_percentile')
        plt.show()

    def print_stat(self, stat):
        print('Total frames numbers', stat['summary']['frames_total'])
        print('Consecutive frames number', stat['summary']['cons_fr_num'])
        matches_per_frame = stat['summary']['frames_total'] / stat['summary']['cons_fr_num']
        print('matches_per_frame: ', matches_per_frame)

        print('Frames in loops number', stat['summary']['loop_fr_num'])
        print('Unique frames in loops number', stat['summary']['unique_loop_fr_num'])
        if stat['summary']['unique_loop_fr_num'] != 0:
            matches_per_loop = stat['summary']['loop_fr_num'] / stat['summary']['unique_loop_fr_num']
        else:
            matches_per_loop = 0

        print('Matches_per_loop: ', matches_per_loop)
        print('Number of loops: ', stat['summary']['loops_num'])

    def append_to_history(self, history, stat):
        for key in self.dataframe_types:
            history[key] = history[key].append(stat[key]).reset_index(drop=True)

        for k, v in stat['summary'].items():
            history['summary'][k] += v

        return history

    def init_data(self):
        history = defaultdict(pd.DataFrame)
        history['summary'] = defaultdict(int)
        return history

    def find_gt(self, dataset_root, predict_path):
        trajectory_name = predict_path.stem

        subsets = ['train_', 'test_', 'val_']
        for subset in subsets:
            if trajectory_name[:len(subset)] == subset:
                trajectory_name = trajectory_name[len(subset):]

        found_files = list(dataset_root.glob(trajectory_name))
        found_files.extend(list((dataset_root / 'train').glob(trajectory_name)))
        found_files.extend(list((dataset_root / 'test').glob(trajectory_name)))
        found_files.extend(list((dataset_root / 'val').glob(trajectory_name)))
        if len(found_files) == 0:
            print(f'Skipping {trajectory_name}. Gt trajectory directory not found')
            return None

        gt_path = (found_files[0] / 'df.csv').as_posix()
        return gt_path

    def get_dataset_stat(self, dataset_root, loop_threshold, keyframe_period, predict_root, plot_traj_stat=False):
        dataset_root = Path(dataset_root)
        predict_root = Path(predict_root)

        gt_history = self.init_data()
        predict_history = self.init_data()

        for trajectory_id, predict_path in enumerate(predict_root.iterdir()):

            if predict_path.suffix != '.csv':
                continue

            print('Prediction_path', predict_path)
            gt_path = self.find_gt(dataset_root, predict_path)
            if not gt_path:
                continue

            print('Gt path', gt_path)
            gt_stat = self.get_trajectory_stat(gt_path, loop_threshold, keyframe_period, trajectory_id)

            if len(gt_stat['all']) == 0:
                print(f'Skipping {directory.as_posix()}. Gt not found')
                continue

            predict_stat = self.get_trajectory_stat(predict_path, loop_threshold, keyframe_period, trajectory_id)

            assert len(gt_stat) == len(predict_stat), f'GT len: {len(gt_stat)}, Predict len ={len(predict_stat)}'

            gt_history = self.append_to_history(gt_history, gt_stat)
            predict_history = self.append_to_history(predict_history, predict_stat)

            if plot_traj_stat:
                self.plot(gt_stat, predict_stat, f'Trajectory {directory.name}')
                self.print_stat(gt_stat)

        self.plot(gt_history, predict_history, 'Dataset stat')
        self.print_stat(gt_history)
