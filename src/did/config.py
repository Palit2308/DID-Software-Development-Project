"""All the general configuration of the project."""
from pathlib import Path

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()

METHODS = ["ols" , "crse" , "res_agg" , "wbtest", "fgls"]
DATA_GEN_LIST = ['homogenous_AR1', 'heterogenous_AR1', 'homogenous_MA1']


__all__ = ["BLD", "SRC", "TEST_DIR", "METHODS", "DATA_GEN_LIST"]
