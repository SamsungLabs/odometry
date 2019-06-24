import os

from odometry.preprocessing.parsers.tum_parser import TUMParser


class RetailBotParser(TUMParser):

    def __init__(self, src_dir):
        try:
            super(RetailBotParser, self).__init__(src_dir)
        except RuntimeError:
            self.gt_txt_path = os.path.join(self.src_dir, 'pose.txt')
            if not os.path.exists(self.gt_txt_path):
                raise RuntimeError(f"Couldn't find groundtruth.txt: {self.gt_txt_path}")

            self.depth_txt_path = os.path.join(self.src_dir, 'depth.txt')
            if not os.path.exists(self.depth_txt_path):
                raise RuntimeError(f"Couldn't find depth.txt: {self.depth_txt_path}")

            self.rgb_txt_path = os.path.join(self.src_dir, 'rgb.txt')
            if not os.path.exists(self.rgb_txt_path):
                raise RuntimeError(f"Couldn't find rgb.txt: {self.rgb_txt_path}")
        self.skiprows = 0

    def __repr__(self):
        return 'RetailBotParser(txt_path={})'.format(self.gt_txt_path)


class SAICOfficeParser(RetailBotParser):

    def __init__(self, src_dir):
        super(SAICOfficeParser, self).__init__(src_dir)
        self.cols = ['path_to_rgb']

    def _load_data(self):
        self.dataframes = [self._load_rgb_txt(), self._load_gt_txt()]
        self.timestamp_cols = ['timestamp_rgb', 'timestamp_gt']

    def _load_depth_txt(self):
        pass

    def __repr__(self):
        return 'SAICOfficeParser(txt_path={})'.format(self.gt_txt_path)
