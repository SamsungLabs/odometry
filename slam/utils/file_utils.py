import os
import shutil


def chmod(path):
    mode = 0o755 if os.path.isdir(path) else 0o644
    os.chmod(path, mode)


def _copy(src, dst):
    are_dirs = os.path.isdir(src) and os.path.isdir(src)
    are_files = os.path.isfile(dst) and os.path.isfile(src)
    assert are_dirs or are_files
    if are_dirs:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copyfile(src, dst)


def symlink(src, dst):
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        elif os.path.isfile(dst):
            os.remove(dst)
        else:
            raise ValueError(f'{dst} should be a path to a directory or a file')

    os.symlink(src, dst)


def _create_file_path(save_dir, trajectory_id, ext, prediction_id='', subset=''):
    trajectory_name = trajectory_id.replace('/', '_')
    file_path = os.path.join(save_dir,
                             prediction_id,
                             subset,
                             trajectory_name + '.' + ext)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    chmod(os.path.dirname(file_path))
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
