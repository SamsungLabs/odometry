import pandas as pd

from .base_parser import BaseParser


class ElementwiseParser(BaseParser):

    def __init__(self, src_dir):
        super(ElementwiseParser, self).__init__(src_dir)
        self.trajectory = None

    def _parse_item(self, item):
        raise NotImplementedError

    def _load_data(self):
        raise NotImplementedError

    def _create_dataframe(self):
        trajectory_parsed = [self._parse_item(item) for item in self.trajectory]
        self.df = pd.DataFrame.from_records(trajectory_parsed)
