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
