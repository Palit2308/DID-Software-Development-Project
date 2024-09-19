import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import warnings
from src.did.monte_carlo_analysis.subfunctions import _random_placebo_treatment, _power_function, _calculate_statistics, _ols_regression_function, _crse_regression_function, _boot_regression_function, _res_agg_regression_function, _fgls_regression_function

# RANDOM PLACEBO TREATMENT ASSIGNMENT FUNCTION TESTS

@pytest.fixture
def sample_data_placebo():
    return pd.DataFrame({
        'State': ['A', 'A', 'B', 'B'],
        'Time': [1, 2, 1, 2]
    })

def test_state_variable_type(sample_data_placebo):
    with pytest.raises(ValueError):
        _random_placebo_treatment(sample_data_placebo, state_variable=1, time_variable='Time')

def test_time_variable_type(sample_data_placebo):
    with pytest.raises(ValueError):
        _random_placebo_treatment(sample_data_placebo, state_variable='State', time_variable=1)

def test_state_variable_missing(sample_data_placebo):
    with pytest.raises(ValueError):
        _random_placebo_treatment(sample_data_placebo, state_variable='States', time_variable='Time')

def test_time_variable_missing(sample_data_placebo):
    with pytest.raises(ValueError):
        _random_placebo_treatment(sample_data_placebo, state_variable='State', time_variable='Year')

def test_insufficient_unique_states(sample_data_placebo):
    with pytest.raises(ValueError):
        _random_placebo_treatment(sample_data_placebo.iloc[:2], state_variable='State', time_variable='Time')

def test_insufficient_unique_time_periods(sample_data_placebo):
    with pytest.raises(ValueError):
        _random_placebo_treatment(sample_data_placebo.iloc[:2], state_variable='State', time_variable='Time')

def test_starting_period_type(sample_data_placebo):
    with pytest.raises(ValueError):
        _random_placebo_treatment(sample_data_placebo, state_variable='State', time_variable='Time', treatment_starting_period='1')

def test_starting_period_less_than_min(sample_data_placebo):
    with pytest.raises(ValueError):
        _random_placebo_treatment(sample_data_placebo, state_variable='State', time_variable='Time', treatment_starting_period=0)

def test_ending_period_type(sample_data_placebo):
    with pytest.raises(ValueError):
        _random_placebo_treatment(sample_data_placebo, state_variable='State', time_variable='Time', treatment_ending_period='2')

def test_ending_period_greater_than_max(sample_data_placebo):
    with pytest.raises(ValueError):
        _random_placebo_treatment(sample_data_placebo, state_variable='State', time_variable='Time', treatment_ending_period=3)

def test_ending_period_less_than_starting(sample_data_placebo):
    with pytest.raises(ValueError):
        _random_placebo_treatment(sample_data_placebo, state_variable='State', time_variable='Time', treatment_starting_period=2, treatment_ending_period=1)


# POWER FUNCTION TESTS
        
@pytest.fixture
def sample_data_power():
    return pd.DataFrame({
        'Dependent_Variable': [1, 2, 3],
        'Treatment_Var': [0, 1, 1],
        'Invalid_Treatment_Var' : [0, 1, 3]
    })

def test_depvar_type(sample_data_power):
    with pytest.raises(ValueError):
        _power_function(sample_data_power, depvar=1, treatment_var='Treatment_Var', effect=0.5)

def test_treatment_var_type(sample_data_power):
    with pytest.raises(ValueError):
        _power_function(sample_data_power, depvar='Dependent_Variable', treatment_var=1, effect=0.5)

def test_depvar_not_found(sample_data_power):
    with pytest.raises(ValueError):
        _power_function(sample_data_power, depvar='Outcome', treatment_var='Treatment_Var', effect=0.5)

def test_treatment_var_not_found(sample_data_power):
    with pytest.raises(ValueError):
        _power_function(sample_data_power, depvar='Dependent_Variable', treatment_var='Treatment', effect=0.5)

def test_treatment_var_values(sample_data_power):
    with pytest.raises(ValueError):
        _power_function(sample_data_power, depvar='Dependent_Variable', treatment_var='Invalid_Treatment_Var', effect=0.5)

def test_effect_type(sample_data_power):
    with pytest.raises(ValueError):
        _power_function(sample_data_power, depvar='Dependent_Variable', treatment_var='Treatment_Var', effect='0.5') 

# CALCULATE STATISTICS TESTS

@pytest.fixture
def sample_data_stats():
    num_simulations = 1000
    bias_values = [0.1, 0.2, 0.3]
    squared_error_values = [0.01, 0.04, 0.09]
    standard_error_values = [0.05, 0.06, 0.07]
    beta1_estimates = [1.2, 1.3, 1.4]
    reject_count_1 = 150
    reject_count_power = 180
    return num_simulations, bias_values, squared_error_values, standard_error_values, beta1_estimates, reject_count_1, reject_count_power

def test_stats_valid_input(sample_data_stats):
    warnings.filterwarnings("ignore")
    num_simulations, bias_values, squared_error_values, standard_error_values, beta1_estimates, reject_count_1, reject_count_power = sample_data_stats
    result = _calculate_statistics(num_simulations, bias_values, squared_error_values, standard_error_values, beta1_estimates, reject_count_1, reject_count_power)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 9  # Number of output rows
    assert all(isinstance(value, (int, float, str)) for value in result['Value'])

def test_stats_empty_lists():
    with pytest.raises(ValueError):
        _calculate_statistics(1000, [], [0.01, 0.04, 0.09], [0.05, 0.06, 0.07], [1.2, 1.3, 1.4], 150, 180) 

def test_stats_invalid_num_simulations_type(sample_data_stats):
    warnings.filterwarnings("ignore")
    num_simulations, bias_values, squared_error_values, standard_error_values, beta1_estimates, reject_count_1, reject_count_power = sample_data_stats
    with pytest.raises(ValueError):
        _calculate_statistics('1000', bias_values, squared_error_values, standard_error_values, beta1_estimates, reject_count_1, reject_count_power)

def test_stats_invalid_reject_counts_type(sample_data_stats):
    warnings.filterwarnings("ignore")
    num_simulations, bias_values, squared_error_values, standard_error_values, beta1_estimates, reject_count_1, _ = sample_data_stats
    with pytest.raises(ValueError):
        _calculate_statistics(num_simulations, bias_values, squared_error_values, standard_error_values, beta1_estimates, '150', '180')
    with pytest.raises(ValueError):
        _calculate_statistics(num_simulations, bias_values, squared_error_values, standard_error_values, beta1_estimates, reject_count_1, 180.0)

def test_stats_invalid_list_types(sample_data_stats):
    warnings.filterwarnings("ignore")
    num_simulations, bias_values, squared_error_values, standard_error_values, beta1_estimates, reject_count_1, reject_count_power = sample_data_stats
    with pytest.raises(ValueError):
        _calculate_statistics(num_simulations, bias_values, np.array(squared_error_values), standard_error_values, beta1_estimates, reject_count_1, reject_count_power)
    with pytest.raises(ValueError):
        _calculate_statistics(num_simulations, np.array(bias_values), squared_error_values, standard_error_values, beta1_estimates, reject_count_1, reject_count_power)
    with pytest.raises(ValueError):
        _calculate_statistics(num_simulations, bias_values, squared_error_values, standard_error_values, np.array(beta1_estimates), reject_count_1, reject_count_power)
    with pytest.raises(ValueError):
        _calculate_statistics(num_simulations,bias_values, squared_error_values, np.array(standard_error_values), beta1_estimates, reject_count_1, reject_count_power)

def test_stats_negative_reject_counts(sample_data_stats):
    warnings.filterwarnings("ignore")
    num_simulations, bias_values, squared_error_values, standard_error_values, beta1_estimates, _, _ = sample_data_stats
    with pytest.raises(ValueError):
        _calculate_statistics(num_simulations, bias_values, squared_error_values, standard_error_values, beta1_estimates, -1, 180)
    with pytest.raises(ValueError):
        _calculate_statistics(num_simulations, bias_values, squared_error_values, standard_error_values, beta1_estimates, 150, -1)                            

# OLS REGRESSION FUNCTION TESTS
        
@pytest.fixture
def sample_ols_data():
    data = {
        'dependent_var': [1, 2, 3, 4, 5],
        'independent_var1': [2, 3, 4, 5, 6],
        'independent_var2': [3, 4, 5, 6, 7],
        'state': [1, 2, 1, 2, 1],
        'time_period': [1, 2, 3, 4, 5],
        'treat_var': [1, 0, 1, 1, 0] ,
        'invalid_treat_var': [0, 1, 2, 1, 0] 
    }
    return pd.DataFrame(data)

def test_ols_valid_input(sample_ols_data):
    result = _ols_regression_function(sample_ols_data, 'dependent_var', 'treat_var', ['independent_var1', 'independent_var2'], 'state', 'time_period')
    assert isinstance(result, sm.regression.linear_model.RegressionResultsWrapper)

def test_ols_depvar_not_string(sample_ols_data):
    with pytest.raises(ValueError):
        _ols_regression_function(sample_ols_data, 123, 'treat_var', ['independent_var1'])

def test_ols_indvar_not_list(sample_ols_data):
    with pytest.raises(ValueError):
        _ols_regression_function(sample_ols_data, 'dependent_var', 'treat_var', 'independent_var1')

def test_ols_indvar_element_not_string(sample_ols_data):
    with pytest.raises(ValueError):
        _ols_regression_function(sample_ols_data, 'dependent_var', 'treat_var', [1, 'independent_var1'])

def test_ols_indvar_column_not_exist(sample_ols_data):
    with pytest.raises(ValueError):
        _ols_regression_function(sample_ols_data, 'dependent_var', 'treat_var', ['not_exist'])

def test_ols_depvar_column_not_exist(sample_ols_data):
    with pytest.raises(ValueError):
        _ols_regression_function(sample_ols_data, 'not_exist', 'treat_var', ['independent_var1'])

def test_ols_treat_var_not_string(sample_ols_data):
    with pytest.raises(ValueError):
        _ols_regression_function(sample_ols_data, 'dependent_var', 123, ['independent_var1'])

def test_ols_treat_var_column_not_exist(sample_ols_data):
    with pytest.raises(ValueError):
        _ols_regression_function(sample_ols_data, 'dependent_var', 'not_exist', ['independent_var1'])

def test_ols_treat_var_values_not_binary(sample_ols_data):
    with pytest.raises(ValueError):
        _ols_regression_function(sample_ols_data, 'dependent_var', 'invalid_treat_var', ['independent_var1'])

def test_ols_state_variable_column_not_exist(sample_ols_data):
    with pytest.raises(ValueError):
        _ols_regression_function(sample_ols_data, 'dependent_var', 'treat_var', ['independent_var1'], state_variable='not_exist')

def test_ols_time_variable_column_not_exist(sample_ols_data):
    with pytest.raises(ValueError):
        _ols_regression_function(sample_ols_data, 'dependent_var', 'treat_var', ['independent_var1'], time_variable='not_exist')

def test_ols_valid_output(sample_ols_data):
    result = _ols_regression_function(sample_ols_data, 'dependent_var', 'treat_var',  ['independent_var1', 'independent_var2'], 'state', 'time_period')
    assert isinstance(result, sm.regression.linear_model.RegressionResultsWrapper)
# CRSE REGRESSION FUNCTION TESTS

@pytest.fixture
def sample_data_crse():
    data = {
        'dependent_var': [1, 2, 3, 4, 5],
        'independent_var1': [2, 3, 4, 5, 6],
        'independent_var2': [3, 4, 5, 6, 7],
        'state': [1, 2, 1, 2, 1],
        'time_period': [1, 2, 3, 4, 5],
        'treat_var': [1, 0, 1, 0, 1],
        'cluster_var': ['A', 'B', 'A', 'B', 'A']
    }
    return pd.DataFrame(data)

def test_crse_valid_input(sample_data_crse):
    result = _crse_regression_function(sample_data_crse, 'dependent_var', 'treat_var', 'cluster_var', ['independent_var1', 'independent_var2'], 'state', 'time_period')
    assert isinstance(result, sm.regression.linear_model.RegressionResultsWrapper)

def test_crse_depvar_not_string(sample_data_crse):
    with pytest.raises(ValueError):
        _crse_regression_function(sample_data_crse, 123, 'treat_var', 'cluster_var', ['independent_var1'])

def test_crse_indvar_not_list(sample_data_crse):
    with pytest.raises(ValueError):
        _crse_regression_function(sample_data_crse, 'dependent_var', 'treat_var', 'cluster_var', 'independent_var1')

def test_crse_indvar_element_not_string(sample_data_crse):
    with pytest.raises(ValueError):
        _crse_regression_function(sample_data_crse, 'dependent_var', 'treat_var', 'cluster_var', [1, 'independent_var1'])

def test_crse_indvar_column_not_exist(sample_data_crse):
    with pytest.raises(ValueError):
        _crse_regression_function(sample_data_crse, 'dependent_var', 'treat_var', 'cluster_var', ['not_exist'])

def test_crse_depvar_column_not_exist(sample_data_crse):
    with pytest.raises(ValueError):
        _crse_regression_function(sample_data_crse, 'not_exist', 'treat_var', 'cluster_var', ['independent_var1'])

def test_crse_treat_var_not_string(sample_data_crse):
    with pytest.raises(ValueError):
        _crse_regression_function(sample_data_crse, 'dependent_var', 123, 'cluster_var', ['independent_var1'])

def test_crse_treat_var_column_not_exist(sample_data_crse):
    with pytest.raises(ValueError):
        _crse_regression_function(sample_data_crse, 'dependent_var', 'not_exist', 'cluster_var', ['independent_var1'])

def test_crse_treat_var_values_not_binary(sample_data_crse):
    sample_data_crse['treat_var'] = [1, 0, 1, 2, 1]
    with pytest.raises(ValueError):
        _crse_regression_function(sample_data_crse, 'dependent_var', 'treat_var', 'cluster_var', ['independent_var1'])

def test_crse_cluster_var_not_string(sample_data_crse):
    with pytest.raises(ValueError):
        _crse_regression_function(sample_data_crse, 'dependent_var', 'treat_var', 123, ['independent_var1'])

def test_crse_cluster_var_column_not_exist(sample_data_crse):
    with pytest.raises(ValueError):
        _crse_regression_function(sample_data_crse, 'dependent_var', 'treat_var', 'not_exist', ['independent_var1'])

def test_crse_state_variable_column_not_exist(sample_data_crse):
    with pytest.raises(ValueError):
        _crse_regression_function(sample_data_crse, 'dependent_var', 'treat_var', 'cluster_var', ['independent_var1'], state_variable='not_exist')

def test_crse_time_variable_column_not_exist(sample_data_crse):
    with pytest.raises(ValueError):
        _crse_regression_function(sample_data_crse, 'dependent_var', 'treat_var', 'cluster_var', ['independent_var1'], time_variable='not_exist')



# RESIDUAL AGGREGATION REGRESSION FUNCTION

@pytest.fixture
def sample_data_res_agg():
    data = pd.DataFrame({
        'Dependent_Var': np.random.randn(100),
        'Treatment_Var': np.random.randint(0, 2, 100),
        'State_Var': np.random.randint(1, 11, 100),
        'Time_Var': np.random.randint(1, 6, 100),
        'Covariate_1': np.random.randn(100),
        'Covariate_2': np.random.randn(100),
        'Invalid_Treatment_Var' : np.random.randn(100)
    })
    return data

def test_res_agg_valid_input(sample_data_res_agg):
    try:
        _res_agg_regression_function(sample_data_res_agg, 'Dependent_Var', 'Treatment_Var', 'State_Var', 'Time_Var')
    except ValueError:
        pytest.fail("Unexpected ValueError raised")

def test__res_agg_invalid_depvar_input(sample_data_res_agg):
    with pytest.raises(ValueError):
        _res_agg_regression_function(sample_data_res_agg, 'Nonexistent_Var', 'Treatment_Var', 'State_Var', 'Time_Var')

def test_res_agg_invalid_covar_input(sample_data_res_agg):
    with pytest.raises(ValueError):
        _res_agg_regression_function(sample_data_res_agg, 'Dependent_Var', 'Treatment_Var', 'State_Var', 'Time_Var', indvar=['Nonexistent_Covar'])

def test_res_agg_invalid_treatment_input(sample_data_res_agg):
    with pytest.raises(ValueError):
        _res_agg_regression_function(sample_data_res_agg, 'Dependent_Var', 'Nonexistent_Var', 'State_Var', 'Time_Var')


def test_res_agg_treat_var_values_not_binary(sample_data_res_agg):
    with pytest.raises(ValueError):
        _res_agg_regression_function(sample_data_res_agg, 'Dependent_Var', 'Invalid_Treatment_Var', 'State_Var', 'Time_Var', ['Covariate_1'])


def test_res_agg_invalid_state_input(sample_data_res_agg):
    with pytest.raises(ValueError):
        _res_agg_regression_function(sample_data_res_agg, 'Dependent_Var', 'Treatment_Var', 'Nonexistent_Var', 'Time_Var')

def test_res_agg_invalid_time_input(sample_data_res_agg):
    with pytest.raises(ValueError):
        _res_agg_regression_function(sample_data_res_agg, 'Dependent_Var', 'Treatment_Var', 'State_Var', 'Nonexistent_Var')

def test_res_agg_treatment_var_type(sample_data_res_agg):
    model = _res_agg_regression_function(sample_data_res_agg, 'Dependent_Var', 'Treatment_Var', 'State_Var', 'Time_Var')
    assert isinstance(model, sm.regression.linear_model.RegressionResultsWrapper)

def test_res_agg_constant_term(sample_data_res_agg):
    model = _res_agg_regression_function(sample_data_res_agg, 'Dependent_Var', 'Treatment_Var', 'State_Var', 'Time_Var')
    assert 'const' in model.params.index

def test_res_agg_regression_model_fit(sample_data_res_agg):
    model = _res_agg_regression_function(sample_data_res_agg, 'Dependent_Var', 'Treatment_Var', 'State_Var', 'Time_Var')
    assert isinstance(model, sm.regression.linear_model.RegressionResultsWrapper)

def test_res_agg_residuals_calculation(sample_data_res_agg):
    model = _res_agg_regression_function(sample_data_res_agg, 'Dependent_Var', 'Treatment_Var', 'State_Var', 'Time_Var')
    assert 'Residuals' in sample_data_res_agg.columns

def test_res_agg_structure(sample_data_res_agg):
    model = _res_agg_regression_function(sample_data_res_agg, 'Dependent_Var', 'Treatment_Var', 'State_Var', 'Time_Var')
    assert set(model.params.index) == {'const', 'TREATMENT'}

def test_res_agg_model_summary(sample_data_res_agg):
    model = _res_agg_regression_function(sample_data_res_agg, 'Dependent_Var', 'Treatment_Var', 'State_Var', 'Time_Var')
    assert isinstance(model.summary(), sm.iolib.summary.Summary)

# WILD BOOTTEST REGRESSION FUNCTION TESTS

@pytest.fixture
def sample_data_wbtest():
    return pd.DataFrame({
        'Dependent_Var': [1, 2, 3],
        'treat_var': [0, 1, 0],
        'Cluster_Var': ['A', 'B', 'A'],
        'indvar': [4, 5, 6],
        'indvar2': [7, 8, 9],
        'State_Variable': ['X', 'Y', 'Z'],
        'Time_Variable': [1, 2, 3],
    })

def test_wbtest_depvar_type(sample_data_wbtest):
    with pytest.raises(ValueError):
        _boot_regression_function(sample_data_wbtest, depvar=1, treat_var='treat_var', cluster_var='Cluster_Var', indvar=['indvar'])

def test_wbtest_treat_var_type(sample_data_wbtest):
    with pytest.raises(ValueError):
        _boot_regression_function(sample_data_wbtest, depvar='Dependent_Var', treat_var=1, cluster_var='Cluster_Var', indvar=['indvar'])

def test_wbtest_cluster_var_type(sample_data_wbtest):
    with pytest.raises(ValueError):
        _boot_regression_function(sample_data_wbtest, depvar='Dependent_Var', treat_var='treat_var', cluster_var=1, indvar=['indvar'])

def test_wbtest_indvar_type(sample_data_wbtest):
    with pytest.raises(ValueError):
        _boot_regression_function(sample_data_wbtest, depvar='Dependent_Var', treat_var='treat_var', cluster_var='Cluster_Var', indvar=1)

def test_wbtest_depvar_not_found(sample_data_wbtest):
    with pytest.raises(ValueError):
        _boot_regression_function(sample_data_wbtest, depvar='Outcome', treat_var='treat_var', cluster_var='Cluster_Var', indvar=['indvar'])

def test_wbtest_treat_var_not_found(sample_data_wbtest):
    with pytest.raises(ValueError):
        _boot_regression_function(sample_data_wbtest, depvar='Dependent_Var', treat_var='Treatment', cluster_var='Cluster_Var', indvar=['indvar'])

def test_wbtest_cluster_var_not_found(sample_data_wbtest):
    with pytest.raises(ValueError):
        _boot_regression_function(sample_data_wbtest, depvar='Dependent_Var', treat_var='treat_var', cluster_var='Cluster', indvar=['indvar'])

def test_wbtest_indvar_not_found(sample_data_wbtest):
    with pytest.raises(ValueError):
        _boot_regression_function(sample_data_wbtest, depvar='Dependent_Var', treat_var='treat_var', cluster_var='Cluster_Var', indvar=['indvar1', 'indvar2'])

def test_wbtest_duplicate_variables(sample_data_wbtest):
    with pytest.raises(ValueError):
        _boot_regression_function(sample_data_wbtest, depvar='Dependent_Var', treat_var='treat_var', cluster_var='Cluster_Var', indvar=['indvar', 'treat_var'])
        
def test_wbtest_regression_results_type(sample_data_wbtest):
    results = _boot_regression_function(sample_data_wbtest, depvar='Dependent_Var', treat_var='treat_var', cluster_var='Cluster_Var', indvar=['indvar'])
    assert isinstance(results, dict)

def test_wbtest_result_keys(sample_data_wbtest):
    results = _boot_regression_function(sample_data_wbtest, depvar='Dependent_Var', treat_var='treat_var', cluster_var='Cluster_Var', indvar=['indvar'])
    expected_keys = ['Observed Coefficient', 'Mean Bootstrapped Coefficient', 'Bias', 'MSE', 'RMSE', 'Standard Error', 'P-Value']
    assert all(key in results for key in expected_keys) 

# FGLS REGRESSION FUNCTION TESTS
    
@pytest.fixture
def sample_data_fgls():
    # Sample panel data
    data = pd.DataFrame({
        'Dependent_Var': np.random.rand(100),
        'Independent_Var1': np.random.rand(100),
        'Independent_Var2': np.random.rand(100),
        'State_Variable': np.random.randint(1, 5, 100),
        'Time_Variable': np.random.randint(1, 10, 100),
        'Treat_Var' : np.random.randint(0, 2, size=100)
    })
    return data

def test_fgls_valid_input(sample_data_fgls):
    data = sample_data_fgls
    model = _fgls_regression_function(data, 'Dependent_Var', 'Treat_Var', ['Independent_Var1', 'Independent_Var2'], state_variable='State_Variable', time_variable='Time_Variable')
    assert isinstance(model, sm.regression.linear_model.RegressionResultsWrapper)

def test_fgls_invalid_depvar_type(sample_data_fgls):
    data = sample_data_fgls
    with pytest.raises(ValueError):
        _fgls_regression_function(data, 123, 'Treat_Var', ['Independent_Var1', 'Independent_Var2'], state_variable='State_Variable', time_variable='Time_Variable')

def test_fgls_invalid_indvar_type(sample_data_fgls):
    data = sample_data_fgls
    with pytest.raises(ValueError):
        _fgls_regression_function(data, 'Dependent_Var', 'Treat_Var', 'Invalid_Ind_Var', state_variable='State_Variable', time_variable='Time_Variable')

def test_fgls_missing_depvar(sample_data_fgls):
    data = sample_data_fgls.drop(columns=['Dependent_Var'])
    with pytest.raises(ValueError):
        _fgls_regression_function(data, 'Dependent_Var', 'Treat_Var', ['Independent_Var1', 'Independent_Var2'], state_variable='State_Variable', time_variable='Time_Variable')

def test_fgls_missing_indvar(sample_data_fgls):
    data = sample_data_fgls.drop(columns=['Independent_Var1'])
    with pytest.raises(ValueError):
        _fgls_regression_function(data, 'Dependent_Var', 'Treat_Var', ['Independent_Var1', 'Independent_Var2'], state_variable='State_Variable', time_variable='Time_Variable')

def test_fgls_invalid_state_var_type(sample_data_fgls):
    data = sample_data_fgls
    with pytest.raises(ValueError):
        _fgls_regression_function(data, 'Dependent_Var', 'Treat_Var', ['Independent_Var1', 'Independent_Var2'], state_variable=123, time_variable='Time_Variable')

def test_fgls_invalid_time_var_type(sample_data_fgls):
    data = sample_data_fgls
    with pytest.raises(ValueError):
        _fgls_regression_function(data, 'Dependent_Var', 'Treat_Var', ['Independent_Var1', 'Independent_Var2'], state_variable='State_Variable', time_variable=123)    

def test_fgls_no_state_variable(sample_data_fgls):
    data = sample_data_fgls
    model = _fgls_regression_function(data, 'Dependent_Var', 'Treat_Var', ['Independent_Var1', 'Independent_Var2'], time_variable='Time_Variable')
    assert isinstance(model, sm.regression.linear_model.RegressionResultsWrapper)

def test_fgls_no_time_variable(sample_data_fgls):
    data = sample_data_fgls
    model = _fgls_regression_function(data, 'Dependent_Var', 'Treat_Var', ['Independent_Var1', 'Independent_Var2'], state_variable='State_Variable')
    assert isinstance(model, sm.regression.linear_model.RegressionResultsWrapper)

def test_fgls_no_ind_var(sample_data_fgls):
    data = sample_data_fgls
    model = _fgls_regression_function(data, 'Dependent_Var', 'Treat_Var', state_variable='State_Variable', time_variable='Time_Variable')
    assert isinstance(model, sm.regression.linear_model.RegressionResultsWrapper)

def test_fgls_invalid_dummy_indvar_type(sample_data_fgls):
    data = sample_data_fgls
    with pytest.raises(ValueError):
        _fgls_regression_function(data, 'Dependent_Var', 'Treat_Var', dummy_indvar=123, state_variable='State_Variable', time_variable='Time_Variable')

def test_fgls_invalid_dummy_indvar_list_type(sample_data_fgls):
    data = sample_data_fgls
    with pytest.raises(ValueError):
        _fgls_regression_function(data, 'Dependent_Var', 'Treat_Var', dummy_indvar=['Dummy_Var'], state_variable='State_Variable', time_variable='Time_Variable')

def test_fgls_invalid_dummy_indvar_columns(sample_data_fgls):
    data = sample_data_fgls
    with pytest.raises(ValueError):
        _fgls_regression_function(data, 'Dependent_Var', 'Treat_Var', dummy_indvar=['Invalid_Dummy_Var'], state_variable='State_Variable', time_variable='Time_Variable')