import pandas as pd
from statistics import mean, median
from pathlib import Path
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from slam.linalg import (convert_euler_angles_to_rotation_matrix, 
                         GlobalTrajectory,
                         RelativeTrajectory,
                         euler_to_quaternion,
                         shortest_path_with_normalization)

class DatasetStat:
    def __init__(self):
        self.rotation_columns = ['euler_x', 'euler_y', 'euler_z']
        self.translation_columns =  ['t_x', 't_y', 't_z']
        self.mean_cols = self.rotation_columns + self.translation_columns 
        self.std_cols = [c + '_confidence' for c in self.mean_cols] 
        self.vis_columns = ['translation_distance', 'rotation_distance'] + self.translation_columns + self.rotation_columns
        self.dataframe_types = ['all', 'consecutive_fr', 'consecutive_kfr', 'loops']
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
            if std_col not in gt.columns:
                predict[std_col] = 1

        if 'to_path' in gt.columns:
            predict['to_index'] = gt['to_index']
            predict['from_index'] = gt['from_index']
        elif 'path_to_rgb_next' in gt.columns:
            predict['to_index'] = gt['path_to_rgb_next'].apply(lambda x: int(Path(x).stem))
            predict['from_index'] = gt['path_to_rgb'].apply(lambda x: int(Path(x).stem))
        else:
            predict['to_index'] = np.arange(1, len(gt) + 1)
            predict['from_index'] = np.arange(0, len(gt))
        return predict

    def get_trajectory_stat(self, path_to_csv, loop_threshold, keyframe_period):
        stat = self.init_data()

        pair_frame_df = pd.read_csv(path_to_csv)
        if len(pair_frame_df.columns) < 6:
            return stat

        pair_frame_df = self.df2slam_predict(pair_frame_df)    
        pair_frame_df = self.get_pair_frame_stat(pair_frame_df)
        stat['all'] = pair_frame_df

        index_difference = pair_frame_df.to_index - pair_frame_df.from_index
        consecutive_fr = pair_frame_df[index_difference == 1].reset_index(drop=True)
        stat['consecutive_fr'] = consecutive_fr

        is_loop = index_difference >= loop_threshold
        loops = pair_frame_df[is_loop].reset_index(drop=True)
        stat['loops'] = loops

        is_periodic = ( index_difference >= keyframe_period) & (index_difference < loop_threshold)
        consecutive_kfr = pair_frame_df[is_periodic].reset_index(drop=True)
        stat['consecutive_kfr'] = consecutive_kfr
        
        stat['summary']['frames_total'] = len(pair_frame_df)
        stat['summary']['cons_fr_num'] = len(consecutive_fr)

        if len(loops) == 0:
            return stat

        stat['summary']['loop_fr_num'] = len(loops)
        if stat['summary']['loop_fr_num'] > 0:
            stat['summary']['unique_loop_fr_num'] = len(loops.to_index.unique())

        loops_num = 1
        previous_index = loops['to_index'].values[0]
        for index, row in loops.iterrows():
            if row.to_index - previous_index > 3*keyframe_period:
                loops_num += 1
            previous_index = row.to_index

        stat['summary']['loops_num'] = loops_num
        return stat

    def filter_outlier(self, x, percentile):
        if len(x) > 0:
            return x[np.where(np.abs(x) < np.percentile(x, percentile))]
        else:
            return list()

    def plot_hist(self, ax, gt_stat, pred_stat, column, percentile, title):

        y = list()
        for key in self.dataframe_types:
            gt = self.filter_outlier(gt_stat[key][column].values, percentile)  
            y.append(gt)

        y.append(self.filter_outlier(pred_stat['all'][column].values, percentile)) 

        colors = ['b', 'r', 'g', 'y', 'c']

        _, bins, _ = ax.hist(y, normed=True, color=colors)

        pair_frame_median = median(y[0]) if len(y[0]) else 0.0
        adjustment_median = median(y[1]) if len(y[1]) else 0.0
        periodic_median = median(y[2]) if len(y[2]) else 0.0
        loop_median = median(y[3]) if len(y[3]) else 0.0
        predict_median = median(y[4])  if len(y[4]) else 0.0


        pair_frame_max = max(y[0]) if len(y[0]) > 0 else float('-inf')
        adjustment_max = max(y[1]) if len(y[1]) > 0 else float('-inf')
        periodic_max = max(y[2]) if len(y[2]) > 0 else float('-inf')
        loop_max = max(y[3]) if len(y[3]) > 0 else float('-inf')
        predict_max = max(y[4]) if len(y[4]) > 0 else float('-inf')

        pair_frame_min = min(y[0]) if len(y[0]) > 0 else float('inf')
        adjustment_min = min(y[1]) if len(y[1]) > 0 else float('inf')
        periodic_min = min(y[2]) if len(y[2]) > 0 else float('inf')
        loop_min = min(y[3]) if len(y[3]) > 0 else float('inf')
        predict_min = min(y[4]) if len(y[4]) > 0 else float('inf')

        legend = [f'gt all. [{pair_frame_min:.2}, {pair_frame_max:.2}] m={pair_frame_median:.2}.',
                  f'gt cons. fr.. [{adjustment_min:.2}, {adjustment_max:.2}], m={adjustment_median:.2}',
                  f'gt cons. kfr. [{periodic_min:.2}, {periodic_max:.2}], m={periodic_median:.2}',
                  f'gt loops. [{loop_min:.2}, {loop_max:.2}], m={loop_median:.2}',
                  f'predict all. [{predict_min:.2}, {predict_max:.2}], m={predict_median:.2}']



        ax.legend(legend)
        ax.set_title(title)
        ax.set_ylabel('normalized number of samples')
        ax.set_xlabel('x')
        ax.grid()

        total_min = min([pair_frame_min, adjustment_min, loop_min, periodic_min, predict_min])
        total_max = max([pair_frame_max, adjustment_max, loop_max, periodic_max, predict_max])

        if np.isinf(total_min) or np.isinf(total_max):
            return bins

        ax.axis(xmin=total_min, xmax=total_max)

        return bins

    def plot_error(self, ax, gt_stat, pred_stat, bins, column, percentile, title):
        markers = ['bo', 'r+', 'gx', 'y8']
        legend = ['all','consecutive frames', 'consective keyframes', 'loops']
        for index, key in enumerate(self.dataframe_types):
            gt = gt_stat[key][column].values
            pred = pred_stat[key][column].values
            assert len(gt) == len(pred), f'len(gt)={len(gt)}, len(pred)={len(pred)}'

            gt = self.filter_outlier(gt, percentile)

            line = np.zeros(len(bins) - 1)
            for i in range(1, bins.shape[0]):
                indices = np.where((bins[i-1] <= gt) & (gt < bins[i]))
                if np.sum(indices) == 0:
                    line[i - 1] = 0
                else:
                    error = median(np.abs(gt[indices] - pred[indices]))
                    line[i - 1] = error
            indices = np.where(line != 0)
            x = bins[:-1] + (bins[1:] - bins[:-1]) / 2
            ax.plot(x[indices], line[indices], markers[index], linestyle='solid')
            ax.set_yscale('log')
        ax.legend(legend)
        ax.set_title(title)
        ax.set_ylabel('median of |error|')
        ax.set_xlabel('gt value')
        ax.axis(xmin=bins[0], xmax=bins[-1])
        ax.grid()

    def plot_pair(self, ax_1, ax_2, gt_stat, pred_stat, column, percentile):
        bins = self.plot_hist(ax_1, gt_stat, pred_stat, column, percentile, title=f'{column}_{percentile}_percentile')
        self.plot_error(ax_2, gt_stat, pred_stat, bins, column, percentile, title=f'{column}_error_{percentile}_percentile')

    def plot_subplot(self, gt_stat, predict_stat, title):
        num_of_cols = len(self.vis_columns)
        num_of_rows = len(self.percentiles) * 2
        
        f, axes = plt.subplots(num_of_rows, num_of_cols, figsize=(6 * num_of_cols, 32))
        f.suptitle(title, fontsize=16)

        for index, column in enumerate(self.vis_columns):
            for p_index, p in enumerate(self.percentiles):
                ax_1 = axes[(index // num_of_cols)*6 + p_index * 2, index % num_of_cols]
                ax_2 = axes[(index // num_of_cols)*6 + p_index * 2 + 1, index % num_of_cols]
                self.plot_pair(ax_1, ax_2, gt_stat, predict_stat, column, p)
        plt.show()

    def print_stat(self, stat):
        print('Total frames numbers', stat['summary']['frames_total'])
        print('Consecutive frames number' , stat['summary']['cons_fr_num'])
        matches_per_frame  = stat['summary']['frames_total'] / stat['summary']['cons_fr_num']
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

        for k, v in  stat['summary'].items():
             history['summary'][k] += v

        return history

    def init_data(self):
        history = dict()
        for key in self.dataframe_types:
            history[key] = pd.DataFrame()

        history['summary'] = dict()
        
        for key in self.summary_keys:
            history['summary'][key] = 0
        return history

    def get_dataset_stat(self, dataset_root, loop_threshold, keyframe_period, predict_root):
        dataset_root = Path(dataset_root)

        gt_history = self.init_data()
        predict_history = self.init_data()

        for directory in dataset_root.iterdir():
            if not directory.is_dir():
                continue
            gt_stat = self.get_trajectory_stat(directory/'df.csv', loop_threshold, keyframe_period)

            if len(gt_stat['all']) == 0:
                print(f'Skipping {directory.as_posix()}. Gt not found')
                continue


            predict_root = Path(predict_root)
            found_files = list(predict_root.rglob(f'{directory.name}.csv'))
            
            if len(found_files) == 0:
                print(f'Skipping {directory.as_posix()}. Prediction not found')
                continue

            predict_path = found_files[0].as_posix()
            
            print('Prediction_path', predict_path)
            predict_stat = self.get_trajectory_stat(predict_path, loop_threshold, keyframe_period)
            
            assert len(gt_stat) == len(predict_stat), f'GT len: {len(gt_stat)}, Predict len ={len(predict_stat)}'

            gt_history = self.append_to_history(gt_history, gt_stat)
            predict_history = self.append_to_history(predict_history, predict_stat)          

            self.plot_subplot(gt_stat, predict_stat, f'Trajectory {directory.name}')
            self.print_stat(gt_stat)

        self.plot_subplot(gt_history, predict_history, 'Dataset stat')
        self.print_stat(gt_history)
