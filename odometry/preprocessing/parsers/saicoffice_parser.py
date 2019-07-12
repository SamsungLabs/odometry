from .retailbot_parser import RetailBotParser


class SAICOfficeParser(RetailBotParser):

    def __init__(self, src_dir):
        super(SAICOfficeParser, self).__init__(src_dir)
        self.name = 'SAICOfficeParser'
        self.cols = ['path_to_rgb']

    def _load_data(self):
        self.dataframes = [self._load_rgb_txt(), self._load_gt_txt()]
        self.timestamp_cols = ['timestamp_rgb', 'timestamp_gt']
