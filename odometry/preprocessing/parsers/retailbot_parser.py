import os

from odometry.preprocessing.parsers.tum_parser import TUMParser


class RetailBotParser(TUMParser):
    
    def __init__(self,
                 trajectory_dir,
                 src_dir):
        super(RetailBotParser, self).__init__(trajectory_dir, src_dir)
        self.gt_txt_path = os.path.join(self.src_dir, 'pose.txt')
        self.depth_txt_path = os.path.join(self.src_dir, 'depth.txt')
        self.rgb_txt_path = os.path.join(self.src_dir, 'rgb.txt')
        self.skiprows = 0

    def __repr__(self):
        return 'RetailBotParser(dir={}, txt_path={})'.format(self.dir, self.gt_txt_path)

    
    
class SAICOfficeParser(RetailBotParser):
    
    def __init__(self,
                 trajectory_dir,
                 src_dir):
        super(SAICOfficeParser, self).__init__(trajectory_dir, src_dir)
        self.cols = ['path_to_rgb']
        
    def _load_data(self):
        self.dataframes = [ self._load_rgb_txt(), self._load_gt_txt()]
        self.timestamp_cols = ['timestamp_rgb', 'timestamp_gt']
        
    def _load_depth_txt(self):
        pass
        
    def __repr__(self):
        return 'SAICOfficeParser(dir={}, txt_path={})'.format(self.dir, self.gt_txt_path)