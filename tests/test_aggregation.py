import __init_path__
import env

from slam.aggregation import DummyAverager
from tests.base_aggragation_testing import BaseTest


class TestDummyAverager(BaseTest):
    def setUp(self) -> None:
        self.algorithm = DummyAverager()
