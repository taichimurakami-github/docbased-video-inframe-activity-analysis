import os
from pathlib import Path


def get_data_dir():
    return os.path.join(
        Path(__file__).parent.parent.parent.parent.absolute(), ".data"
    )
