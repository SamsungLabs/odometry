class BaseAggregator:
    def append(self, df):
        raise RuntimeError('Not implemented')

    def get_trajectory(self):
        raise RuntimeError('Not implemented')
