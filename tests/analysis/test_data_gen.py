import numpy as np
import pytest
import pandas as pd
from src.did.monte_carlo_analysis.synthetic_data_gen import _generate_homogenous_ar1_data
from src.did.monte_carlo_analysis.synthetic_data_gen import _generate_heterogenous_ar1_data
from src.did.monte_carlo_analysis.synthetic_data_gen import _generate_homogenous_ma1_data




# To check for homogenous AR 1 data generation process

def test_homogenous_ar1_data_output_shape():
    N = 5
    T = 10
    num_individuals = 3
    df = _generate_homogenous_ar1_data(N, T, rho=0.5, num_individuals=num_individuals)
    assert df.shape == ((N * num_individuals * T), 4)  # Check if the shape of the DataFrame is correct

def test_homogenous_ar1_data_column_names():
    N = 5
    T = 10
    num_individuals = 3
    df = _generate_homogenous_ar1_data(N, T, rho=0.5, num_individuals=num_individuals)
    expected_columns = ['state', 'individual', 'time', 'value']
    assert all(col in df.columns for col in expected_columns)  # Check if all expected columns are present

def test_homogenous_ar1_data_invalid_input():
    with pytest.raises(ValueError):
        _generate_homogenous_ar1_data(N=0, T=10, rho=0.5, num_individuals=3)  # N is not positive
    with pytest.raises(ValueError):
        _generate_homogenous_ar1_data(N=5, T=0, rho=0.5, num_individuals=3)  # T is not positive
    with pytest.raises(ValueError):
        _generate_homogenous_ar1_data(N=5, T=10, rho=-0.5, num_individuals=3)  # rho is not in [0, 1]
    with pytest.raises(ValueError):
        _generate_homogenous_ar1_data(N=5, T=10, rho=0.5, num_individuals=-3)  # num_individuals is not positive


# To check for heterogenous AR 1 data generation process
        
def test_heterogenous_ar1_data_output_shape():
    N = 5
    T = 10
    num_individuals = 3
    df = _generate_heterogenous_ar1_data(N, T, num_individuals=num_individuals)
    assert df.shape == ((N * num_individuals * T), 4)  # Check if the shape of the DataFrame is correct

def test_heterogenous_ar1_data_column_names():
    N = 5
    T = 10
    num_individuals = 3
    df = _generate_heterogenous_ar1_data(N, T, num_individuals=num_individuals)
    expected_columns = ['state', 'individual', 'time', 'value']
    assert all(col in df.columns for col in expected_columns)  # Check if all expected columns are present

def test_heterogenous_ar1_data_invalid_input():
    with pytest.raises(ValueError):
        _generate_heterogenous_ar1_data(N=0, T=10, num_individuals=3)  # N is not positive
    with pytest.raises(ValueError):
        _generate_heterogenous_ar1_data(N=5, T=0, num_individuals=3)  # T is not positive
    with pytest.raises(ValueError):
        _generate_heterogenous_ar1_data(N=5, T=10, num_individuals=-3)  # num_individuals is not positive



# To check for homogenous MA 1 data generation process

def test_homogenous_ma1_data_output_shape():
    N = 5
    T = 10
    num_individuals = 3
    df = _generate_homogenous_ma1_data(N, T, theta=0.5, num_individuals=num_individuals)
    assert df.shape == ((N * num_individuals * T), 4)  # Check if the shape of the DataFrame is correct

def test_homogenous_ma1_data_column_names():
    N = 5
    T = 10
    num_individuals = 3
    df = _generate_homogenous_ma1_data(N, T, theta=0.5, num_individuals=num_individuals)
    expected_columns = ['state', 'individual', 'time', 'value']
    assert all(col in df.columns for col in expected_columns)  # Check if all expected columns are present

def test_homogenous_ma1_data_invalid_input():
    with pytest.raises(ValueError):
        _generate_homogenous_ma1_data(N=0, T=10, theta=0.5, num_individuals=3)  # N is not positive
    with pytest.raises(ValueError):
        _generate_homogenous_ma1_data(N=5, T=0, theta=0.5, num_individuals=3)  # T is not positive
    with pytest.raises(ValueError):
        _generate_homogenous_ma1_data(N=5, T=10, theta=-0.5, num_individuals=3)  # rho is not in [0, 1]
    with pytest.raises(ValueError):
        _generate_homogenous_ma1_data(N=5, T=10, theta=0.5, num_individuals=-3)  # num_individuals is not positive


