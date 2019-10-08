import os


class BaseParser:

    def __init__(self, src_dir):

        self.src_dir = src_dir
        if not os.path.exists(self.src_dir):
            raise RuntimeError(f'Could not find trajectory dir {src_dir}')

        self.cols = ['path_to_rgb', 'path_to_rgb_right', 'path_to_depth']
        self.df = None
        self.name = None

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

    def __repr__(self):
        return f'{self.name} (src_dir={self.src_dir})'
