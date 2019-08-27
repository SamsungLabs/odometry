import os


def _create_file_path(save_dir, trajectory_id, ext, prediction_id='', subset=''):

    trajectory_name = trajectory_id.replace('/', '_')
    file_path = os.path.join(save_dir,
                             prediction_id,
                             subset,
                             trajectory_name + '.' + ext)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    os.chmod(os.path.dirname(file_path), 0o777)
    return file_path


def create_vis_file_path(save_dir, trajectory_id, prediction_id='', subset='', sub_dir=''):
    return _create_file_path(save_dir=os.path.join(save_dir, 'visuals', sub_dir),
                             trajectory_id=trajectory_id,
                             prediction_id=prediction_id,
                             subset=subset,
                             ext='html')


def create_prediction_file_path(save_dir, trajectory_id, prediction_id='', subset=''):
    return _create_file_path(save_dir=os.path.join(save_dir, 'predictions'),
                             trajectory_id=trajectory_id,
                             prediction_id=prediction_id,
                             subset=subset,
                             ext='csv')    
