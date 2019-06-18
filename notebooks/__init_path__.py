import os
import sys
from pathlib import Path

cur_path = Path(os.path.realpath(__file__)).parent
project_path = cur_path

while len(list(project_path.glob('.gitmodules'))) == 0:
    project_path = project_path.parent

sys.path.insert(0, str(project_path))