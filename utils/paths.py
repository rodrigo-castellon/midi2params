import os
import pathlib

LIB_DIR = os.path.dirname(os.path.join(pathlib.Path(__file__).absolute()))

REPO_DIR = os.path.dirname(LIB_DIR)
PARAMS_DIR = os.path.join(REPO_DIR, 'params')
TEST_DATA_DIR = os.path.join(REPO_DIR, 'test_data')
