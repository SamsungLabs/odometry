import os

class BaseParser:

    def __init__(self):
        self.cols = ['path_to_rgb', 'path_to_depth']

    def _load_data(self):
        raise NotImplementedError

    def _make_absolute_filepath(self):
        for col in self.cols:
            self.df[col] = self.df[col].apply(
                lambda filename: os.path.abspath(os.path.join(self.src_dir, filename)))

    def _create_dataframe(self):
        raise NotImplementedError

    def run(self):
        self._load_data()
        self._create_dataframe()
        self._make_absolute_filepath()
        return self.df
