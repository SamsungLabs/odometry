import pandas as pd

from odometry.preprocessing.parsers.base_parser import BaseParser


class ElementwiseParser(BaseParser):

    def _parse_item(self, item):
        raise NotImplementedError

    def _create_dataframe(self):
        trajectory_parsed = [self._parse_item(item) for item in self.trajectory]
        self.df = pd.DataFrame.from_dict(trajectory_parsed)
