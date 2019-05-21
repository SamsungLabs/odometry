import os
import shutil
import copy
import functools
import tqdm
import numpy as np
import pandas as pd
import PIL

from estimator import (
    PWCOpticalFlowEstimator,
    Struct2DepthEstimator,
    SENetDepthEstimator
)
from image_manager import ImageManager
from data_parser import DISCOMANParser, TUMParser
from video_parser import VideoParser
from computation_utils import make_memory_safe, set_computation


VIDEO = 'VIDEO'
DIRECTORY = 'DIRECTORY'
CSV = 'CSV'

DISCOMAN = 'DISCOMAN'
TUM = 'TUM'
KITTI = 'KITTI'


class BaseDatasetBuilder():
    TRAIN = 'train'
    TEST = 'test'

    PWC = 'pwc'
    OPTICAL_FLOW_ESTIMATOR = {
        PWC: PWCOpticalFlowEstimator,
    }
    OPTICAL_FLOW_CHECKPOINT = {
        PWC: '/Vol0/user/f.konokhov/tfoptflow/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'
    }

    STRUCT2DEPTH = 'struct2depth'
    SENET = 'senet'
    DEPTH_ESTIMATOR = {
        STRUCT2DEPTH: Struct2DepthEstimator,
        SENET: SENetDepthEstimator,
    }
    DEPTH_CHECKPOINT = {
        STRUCT2DEPTH: 'struct2depth/model/model-199160',
        SENET: 'depth_pred/senet154_5'
    }

    def __init__(self,
                 sequence_directory,
                 build_from,
                 image_directory=None,
                 depth_directory=None,
                 csv_filename='df.csv',
                 csv_path=None,
                 txt_path=None,
                 video_path=None,
                 json_path=None,
                 mode=None,
                 height=None,
                 width=None,
                 stride=1,
                 fps=30,
                 estimate_optical_flow=True,
                 optical_flow_estimator_name=PWC,
                 optical_flow_checkpoint=None,
                 estimate_depth=True,
                 depth_estimator_name=STRUCT2DEPTH,
                 depth_checkpoint=None,
                 memory_safe=False,
                 **computation_kwargs):

        self.sequence_directory = sequence_directory
        self.build_from = build_from
        self.image_directory = image_directory
        self.depth_directory = depth_directory
        self.csv_filename = csv_filename
        self.video_path = video_path
        self.json_path = json_path
        self.csv_path = csv_path
        self.txt_path = txt_path
        self.mode = mode
        self.height = height
        self.width = width
        self.stride = stride
        self.fps = fps

        self.estimate_optical_flow = estimate_optical_flow
        self.optical_flow_estimator_name = optical_flow_estimator_name
        self.optical_flow_checkpoint = optical_flow_checkpoint

        self.estimate_depth = estimate_depth
        self.depth_estimator_name = depth_estimator_name
        self.depth_checkpoint = depth_checkpoint

        self.memory_safe = memory_safe
        self.computation_kwargs = computation_kwargs

        os.makedirs(self.sequence_directory, exist_ok=True)

        if self.estimate_optical_flow:
            assert self.optical_flow_estimator_name is not None
            self.optical_flow_checkpoint = (
                self.optical_flow_checkpoint
                or BaseDatasetBuilder.OPTICAL_FLOW_CHECKPOINT[self.optical_flow_estimator_name])
            self.optical_flow_estimator = self.OPTICAL_FLOW_ESTIMATOR[self.optical_flow_estimator_name]
            self.optical_flow_directory = os.path.join(self.sequence_directory, 
                                                       'optical_flow_stride{}'.format(self.stride))

        if self.estimate_depth:
            assert self.depth_directory is None
            assert self.depth_estimator_name is not None
            self.depth_checkpoint = (
                self.depth_checkpoint
                or BaseDatasetBuilder.DEPTH_CHECKPOINT[self.depth_estimator_name])
            self.depth_estimator = self.DEPTH_ESTIMATOR[self.depth_estimator_name]
            self.depth_directory = os.path.join(self.sequence_directory, 'depth')

    def _create_image_directory(self):
        self.image_directory = os.path.join(self.sequence_directory, 'images')
        if os.path.exists(self.image_directory):
            shutil.rmtree(self.image_directory)

    def _create_image_manager(self, image_filenames=None, stride=None):
        self.image_manager = ImageManager(self.image_directory,
                                          image_filenames,
                                          height=self.height,
                                          width=self.width,
                                          stride=stride if stride is not None else self.stride,
                                          sample=(self.mode == BaseDatasetBuilder.TEST))

    def _configure(self):
        raise NotImplemented

    def _load_dataframe(self):
        self.dataframe = pd.read_csv(os.path.join(self.sequence_directory, self.csv_filename))
    
    def _save_dataframe(self):
        self.dataframe.to_csv(os.path.join(self.sequence_directory, self.csv_filename), 
                              index=False)

    def _create_dataframe(self):
        self.dataframe = pd.DataFrame(index=range(self.image_manager.num_images))
        self.dataframe['path_to_rgb'] = self.image_manager.image_filepaths
        self.dataframe['x'] = 0
        self.dataframe['y'] = 0
        self.dataframe['z'] = 0
        self.dataframe['euler_x'] = 0
        self.dataframe['euler_y'] = 0
        self.dataframe['euler_z'] = 0

        assert self.fps is not None
        print('Using FPS rate = {}'.format(self.fps))
        self.timestamps = np.arange(self.image_manager.num_images) / self.fps

        self.dataframe['timestamps'] = self.timestamps

    def _run_estimator_wrapper(self, Estimator, checkpoint, directory, dataframe_col, next_dataframe_col=None):
        def _run_estimator():
            estimator = Estimator(directory,
                                  self.image_manager,
                                  checkpoint)
            estimator.run()
            self.dataframe[dataframe_col] = (self.dataframe.path_to_rgb.apply(os.path.basename)).map(estimator.mapping)
            if next_dataframe_col is not None:
                self.dataframe[next_dataframe_col] = (self.dataframe.path_to_next_rgb.apply(os.path.basename)).map(estimator.mapping)
            self._save_dataframe()

        if self.memory_safe:
            make_memory_safe(_run_estimator, **self.computation_kwargs)()
            self._load_dataframe()
        else:
            _run_estimator()
            
    def _add_next_item(self):
        columns = self.dataframe.columns
        add_next_rgb = not 'path_to_next_rgb' in columns
        add_next_depth = 'path_to_depth' in columns and not 'path_to_next_rgb' in columns
        if not (add_next_rgb or add_next_depth):
            return
        
        if add_next_rgb: 
            path_to_next_rgb = self.dataframe.path_to_rgb.values[self.stride:]
        if add_next_depth:
            path_to_next_depth = self.dataframe.path_to_depth[self.stride:]
    
        self.dataframe = self.dataframe[:-self.stride]
        
        if add_next_rgb:
            self.dataframe['path_to_next_rgb'] = path_to_next_rgb
        if add_next_depth:
            self.dataframe['path_to_next_depth'] = path_to_next_depth

    def build(self):
        self._configure()

        if (self.estimate_optical_flow or self.estimate_depth) and not self.memory_safe:
            set_computation(**self.computation_kwargs)
   
        if self.estimate_depth:
            print('Estimate depth')
            print('Model:   {}'.format(self.depth_estimator))
            print('Weights: {}'.format(self.depth_checkpoint))
            print('Output:  {}'.format(self.depth_directory))
            next_dataframe_col = None
            if 'path_to_next_depth' in self.dataframe.columns and self.estimate_depth:
                next_dataframe_col = 'path_to_next_depth'
            self._run_estimator_wrapper(self.depth_estimator, self.depth_checkpoint, self.depth_directory, 
                                        dataframe_col='path_to_depth', next_dataframe_col=next_dataframe_col)
    
        if self.estimate_optical_flow:
            print('Estimate optical flow')
            print('Model:   {}'.format(self.optical_flow_estimator))
            print('Weights: {}'.format(self.optical_flow_checkpoint))
            print('Output:  {}'.format(self.optical_flow_directory))
            self._add_next_item()
            self._run_estimator_wrapper(self.optical_flow_estimator, self.optical_flow_checkpoint, self.optical_flow_directory,
                                       dataframe_col='path_to_optical_flow')
    
        self._save_dataframe()

    def __repr__(self):
        return '''DatasetBuilder(dir={}, build_from={},
            image_dir={}, depth_dir={}, csv_filename={},
            video_path={}, json_path={}, csv_path={},
            mode={}, image_height={}, image_width={}, stride={}, fps={},
            optical_flow={}, optical_flow_estimator={}, optical_flow_checkpoint={},
            depth={}, depth_estimator={}, depth_checkpoint={},
            {}'''.format(
            self.sequence_directory, self.build_from,
            self.image_directory, self.depth_directory, self.csv_filename, self.video_path, self.json_path, self.csv_path,
            self.mode, self.height, self.width, self.stride, self.fps,
            self.estimate_optical_flow, self.optical_flow_estimator_name, self.optical_flow_checkpoint,
            self.estimate_depth, self.depth_estimator_name, self.depth_checkpoint,
            ','.join(['{}={}'.format(k, v) for k, v in self.computation_kwargs.items()]))
    
    
class VideoDatasetBuilder(BaseDatasetBuilder):
    def __init__(self, *args, **kwargs):
        super(VideoDatasetBuilder, self).__init__(*args, **kwargs)

        assert self.video_path is not None
        assert self.mode == BaseDatasetBuilder.TEST

    def _configure(self):
        self._create_image_directory()
        self._create_image_manager()

        self.video_parser = VideoParser(self.image_manager, video_path=self.video_path)
        self.video_parser.parse()
        self.fps = self.video_parser.fps or self.fps

        self.image_manager.reset()
        self._create_dataframe()
        
        
class ImagesDatasetBuilder(BaseDatasetBuilder):
    def __init__(self, *args, **kwargs):
        super(ImagesDatasetBuilder, self).__init__(*args, **kwargs)
        
        assert self.image_directory is not None
        assert self.csv_filename is not None
        
    def _configure(self):
        self._create_image_manager()
        self._create_dataframe()
        if not self.estimate_depth and self.depth_directory is not None:
            depth_ext = os.path.splitext(os.listdir(self.depth_directory)[0])[-1]
            create_path_to_depth = lambda x: os.path.join(
                self.depth_directory,
                ''.join((os.path.splitext(os.path.basename(x))[0], depth_ext)))
            self.dataframe['path_to_depth'] = self.dataframe.path_to_rgb.apply(create_path_to_depth)
            

class CSVDatasetBuilder(BaseDatasetBuilder):
    def __init__(self, *args, **kwargs):
        super(CSVDatasetBuilder, self).__init__(*args, **kwargs)

        assert self.csv_path is not None

    def _configure(self):
        self._create_image_manager()
        self.dataframe = pd.read_csv(self.csv_path)
        

KITTIDatasetBuilder = ImagesDatasetBuilder        


class ParserDatasetBuilder(BaseDatasetBuilder):
    def __init__(self, *args, **kwargs):
        super(ParserDatasetBuilder, self).__init__(*args, **kwargs)
        self.parser = None

    def _configure(self):
        self.parser.parse()

        self.dataframe = self.parser.relative_dataframe
        self.image_directory = self.parser.image_directory
        self._create_image_manager(self.dataframe.path_to_rgb.apply(os.path.basename).values,
                                   stride=1)
        self.dataframe = self.dataframe[self.dataframe.path_to_rgb.apply(os.path.exists)]
        
        
class DISCOMANDatasetBuilder(ParserDatasetBuilder):
    def __init__(self, *args, **kwargs):
        super(DISCOMANDatasetBuilder, self).__init__(*args, **kwargs)

        assert self.json_path is not None
        self.parser = DISCOMANParser(self.sequence_directory, json_path=self.json_path)
        

class TUMDatasetBuilder(ParserDatasetBuilder):
    def __init__(self, *args, **kwargs):
        super(TUMDatasetBuilder, self).__init__(*args, **kwargs)

        assert self.txt_path is not None
        self.parser = TUMParser(self.sequence_directory, txt_path=self.txt_path)
