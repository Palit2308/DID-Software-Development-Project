import pytest
import pandas as pd
import numpy as np
from src.did.data_management.empirical_clean_data import clean_empirical_data, treatment_assignment
from src.did.config import SRC

def test_clean_empirical_data_valid_file():
    file_path = SRC/ "data" / "Trade_Indices_E_All_Data_NOFLAG.csv.gz"
    result = clean_empirical_data(file_path)
    assert isinstance(result, pd.DataFrame)

def test_clean_empirical_data_invalid_file():
    file_path = "invalid_file.csv" 
    with pytest.raises(ValueError):
        clean_empirical_data(file_path)


def test_clean_empirical_data_valid_execution():
    file_path = SRC/ "data" / "Trade_Indices_E_All_Data_NOFLAG.csv.gz"
    # Assume that all conditions are met for a successful execution
    result = clean_empirical_data(file_path)
    assert isinstance(result, pd.DataFrame)


# TREATMENT ASSIGNMENT IN EMPIRICAL WORK

@pytest.fixture
def sample_data():
    return pd.DataFrame({'Area': ['A','A','A','A', 'B','B','B','B', 'C','C','C','C', 'D', 'D', 'D', 'D'], 
                         'Year': [1990,1991,1992,1993,1990,1991,1992,1993,1990,1991,1992,1993,1990,1991,1992,1993]})

# Test case to check if treatment is assigned correctly for valid inputs
def test_treatment_assignment_valid(sample_data):
    sample_dict = {'A': 1992, 'B': 1991, 'C': 1990, 'D': 1993}
    result = treatment_assignment(sample_data.copy(), sample_dict)
    assert result['TREATMENT'].tolist() == [0,0,1,1,0,1,1,1,1,1,1,1,0,0,0,1]


# Test case to check if TypeError is raised when Year values or dictionary values are not compatible for comparison
def test_treatment_assignment_typeerror(sample_data):
    sample_dict = {'A': '1992', 'B': 1991, 'C': 1990, 'D': 1993}
    with pytest.raises(ValueError, match="The 'Year' values or dictionary values are not compatible for comparison"):
        treatment_assignment(sample_data.copy(), sample_dict)