import os

from odometry.preprocessing.parsers.tum_parser import TUMParser


class RetailBotParser(TUMParser):

    def __init__(self, src_dir):
        gt_txt_path = os.path.join(src_dir, 'pose.txt')
        depth_txt_path = os.path.join(src_dir, 'depth.txt')
        rgb_txt_path = os.path.join(src_dir, 'rgb.txt')
        super(RetailBotParser, self).__init__(src_dir,
                                              gt_txt_path=gt_txt_path,
                                              depth_txt_path=depth_txt_path,
                                              rgb_txt_path=rgb_txt_path
                                              )

        self.name = 'RetailBotParser'
        self.skiprows = 0