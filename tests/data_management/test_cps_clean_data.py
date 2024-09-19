import pytest
import pandas as pd
import os
import gzip
from src.did.data_management.cps_clean_data import process_cps_data, aggregate_cps_data


# TESTS FOR CPS DATA CLEANING 

@pytest.fixture
def cps_sample_data(tmp_path):

    data = {
        'INCWAGE': [10000, 40, 999, 20000, 30000, 10000, 800, 1000, 20000, 30000],
        'EDUC': [0, 125, 1, 90, 80, 9, 35, 56, 60, 65],
        'YEAR': [1985, 1990, 1995, 2000, 2005, 1985, 1990, 1995, 1999, 1998],
        'STATEFIP': [5, 10, 20, 30, 50, 5, 10, 20, 30, 50],
        'AGE': [20, 25, 30, 45, 50 ,30, 35, 48, 34, 29],
        'SEX': [1, 2, 1, 2, 2, 2, 2, 2, 2, 2]
    }
    df = pd.DataFrame(data)
    
    gzip_file_path = os.path.join(tmp_path, "sample_data.csv.gz")
    with gzip.open(gzip_file_path, 'wt', encoding='utf-8') as f:
        df.to_csv(f, index=False)

    return gzip_file_path

def test_process_cps_data_columns_exist(cps_sample_data):
    result = process_cps_data(cps_sample_data)
    assert 'INCWAGE' in result.columns
    assert 'EDUC' in result.columns
    assert 'YEAR' in result.columns
    assert 'STATEFIP' in result.columns
    assert 'AGE' in result.columns
    assert 'SEX' in result.columns

def test_process_cps_data_invalid_incwage(cps_sample_data):
    result = process_cps_data(cps_sample_data)
    assert result['INCWAGE'].isin([0, 999]).sum() == 0

def test_process_cps_data_invalid_educ(cps_sample_data):
    result = process_cps_data(cps_sample_data)
    assert result['EDUC'].isin([0, 1]).sum() == 0

def test_process_cps_data_year_range(cps_sample_data):
    result = process_cps_data(cps_sample_data)
    assert result['YEAR'].between(1980, 2000).all()

def test_process_cps_data_education_category(cps_sample_data):
    result = process_cps_data(cps_sample_data)
    assert 'Up to Grade 10' in result.columns
    assert 'High School' in result.columns
    assert "Master's Degree" in result.columns

def test_process_cps_data_boolean_conversion(cps_sample_data):
    result = process_cps_data(cps_sample_data)
    assert result['High School'].isin([0, 1]).all()
    assert result["Master's Degree"].isin([0, 1]).all()
    assert result['Up to Grade 10'].isin([0, 1]).all()

def test_process_cps_data_state_filter(cps_sample_data):
    result = process_cps_data(cps_sample_data)
    assert not ((result['STATEFIP'] > 56) | (result['STATEFIP'] == 11)).any()

def test_process_cps_data_age_and_sex_filter(cps_sample_data):
    result = process_cps_data(cps_sample_data)
    assert result['AGE'].between(25, 50).all()
    assert (result['SEX'] == 2).all()

# TESTS FOR CPS DATA AGGREGATION

@pytest.fixture
def sample_cps_data():
    data = {
        'High School': [100, 200, 150, 180],
        "Master's Degree": [50, 70, 60, 80],
        'Up to Grade 10': [80, 90, 100, 110],
        'AGE': [30, 40, 35, 45],
        'INCWAGE': [50000, 60000, 55000, 58000],
        'STATEFIP': [1, 1, 2, 2],
        'YEAR': [2019, 2020, 2019, 2020]
    }
    return pd.DataFrame(data)

def test_aggregate_cps_data_returns_dataframe(sample_cps_data):
    aggregated_data = aggregate_cps_data(sample_cps_data)
    assert isinstance(aggregated_data, pd.DataFrame)

def test_aggregate_cps_data_missing_INCWAGE(sample_cps_data):
    sample_cps_data.drop(columns=['INCWAGE'], inplace=True)
    with pytest.raises(ValueError):
        aggregate_cps_data(sample_cps_data)

def test_aggregate_cps_data_missing_High_School(sample_cps_data):
    sample_cps_data.drop(columns=['High School'], inplace=True)
    with pytest.raises(ValueError):
        aggregate_cps_data(sample_cps_data)

def test_aggregate_cps_data_missing_Masters_Degree(sample_cps_data):
    sample_cps_data.drop(columns=["Master's Degree"], inplace=True)
    with pytest.raises(ValueError):
        aggregate_cps_data(sample_cps_data)

def test_aggregate_cps_data_missing_Up_to_Grade_10(sample_cps_data):
    sample_cps_data.drop(columns=['Up to Grade 10'], inplace=True)
    with pytest.raises(ValueError):
        aggregate_cps_data(sample_cps_data)

def test_aggregate_cps_data_missing_AGE(sample_cps_data):
    sample_cps_data.drop(columns=['AGE'], inplace=True)
    with pytest.raises(ValueError):
        aggregate_cps_data(sample_cps_data)
