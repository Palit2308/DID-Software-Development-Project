import numpy as np
import pandas as pd
import os
from did.monte_carlo_analysis.synthetic_data_gen import _generate_homogenous_ar1_data
from did.monte_carlo_analysis.subfunctions import _random_placebo_treatment, _ols_regression_function, _crse_regression_function, _res_agg_regression_function, _boot_regression_function, _calculate_statistics, _power_function, _fgls_regression_function

# MONTE CARLO OLS

def monte_carlo_homogenous_AR1_ols(data_info):
    """
    Perform Monte Carlo simulations for a homogenous AR(1) model with placebo treatment assignment.
    This function performs Monte Carlo simulations on the generated treated data to understand 
    the small-sample properties of the coefficient of Treatment using Ordinary Least Squares (OLS) 
    regression.
    The data is synthetically generated for each simulation, and placebo treatment is randomly
    assigned in each run. The small sample statistics of the Treatment coefficient are stored in 
    lists and compiled after the simulations are complete, to produce a DataFrame containing the 
    results.

    Parameters:
    -----------
    data_info : dict
        A dictionary containing the following keys:
        - 'N' : int
            The number of states or groups in the panel data. Must be a positive integer.
        - 'T' : int
            The number of time periods for each state or group. Must be a positive integer 
            greater than 1.
        - 'rho' : float
            The AR(1) autoregressive parameter, specifying the persistence of the process.
            Must be in the range [0, 1].
        - 'num_individuals' : int
            The number of individuals or entities within each state or group. Must be a positive
             integer.
        - 'num_simulations' : int, optional
            The number of Monte Carlo simulations to run.
        - 'alpha' : float, optional
            The significance level for hypothesis testing. Must be in the range (0, 1).
        - 'treatment_starting_period' : int, optional
            The starting period for assigning placebo treatment. Must be an integer not less than
            the minimum time period in the data. Default is None, which assigns it to the minimum 
            time period in the data.
        - 'treatment_ending_period' : int, optional
            The ending period for assigning placebo treatment. Must be an integer within the data's
            time range and not less than treat_start_period if specified. Default is None,
            which assigns it to the maximum time period in the data.

    Raises:
    -------
    ValueError:
        If any of the required parameters are missing or invalid. This function is created using 
        the subfunctions. So if any of the parameters required by the subfunctions 
        is missing or mis-provided, this function will raise a value error.

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the results of the Monte Carlo simulations. The DataFrame includes 
        the following columns:
        - 'Number of Simulations': The total number of simulations performed.
        - 'Type 1 Error': The percentage of simulations where the null hypothesis of no effect was
           rejected when there is actually no effect.
        - 'Power' : The percentage of times where the null of no effect is rejected after 
           adding a 25% effect to the dependent variable.
        - 'Bias': The average bias of the Treatment coefficient estimates.
        - 'MSE': The mean squared error of the Treatment coefficient estimates.
        - 'RMSE': The root mean squared error of the Treatment coefficient estimates.
        - 'Average Standard Error': The average standard error of the Treatment coefficient estimates.
        - 'Confidence Interval Lower Bound': The lower bound of the confidence interval for the
           Treatment coefficient estimates.
        - 'Confidence Interval Upper Bound': The upper bound of the confidence interval for the 
           Treatment coefficient estimates.

    Notes:
    ------
    - This function generates synthetic panel data using a homogenous AR(1) model, assigns
      placebo treatment to states, and performs Monte Carlo simulations to analyze the performance
      of OLS estimators under different scenarios, namely without a 25% effect and with a 25% effect.
    - The treatment_variable is hard coded to be 'TREATMENT' because the _random_placebo_treatment
      function generates a column called 'TREATMENT' which is subsequently used.
    Example:
    --------
    To perform Monte Carlo simulations with 5 states, 10 time periods, a rho of 0.5, and 100
    individuals per state, you can call the function as follows:
        data_dict = {
            'N': 5,
            'T': 10,
            'rho': 0.5,
            'num_individuals': 100
        }
    >>> results = monte_carlo_homogenous_AR1_ols(data_dict)

    The results DataFrame will contain simulation results and finite sample statistics on the
    Treatment coefficient estimates.
    """
    np.random.seed(42)

    N = data_info["N"]
    T = data_info["T"]
    rho = data_info["rho"]
    alpha = data_info["alpha"]
    num_individuals = data_info["num_individuals"]
    num_simulations = data_info["num_simulations"]
    treatment_starting_period=data_info["treatment_starting_period"]
    treatment_ending_period=data_info["treatment_ending_period"]
    
    reject_count_1 = 0
    reject_count_power = 0
    true_beta1_value = 0
    beta1_estimates = []
    bias_values = []
    squared_error_values = []
    standard_error_values = []

    for _ in range(num_simulations):

        data = _generate_homogenous_ar1_data(N, T, rho, num_individuals)

        treat_data = _random_placebo_treatment(data, 'state', 'time', treatment_starting_period, 
                                               treatment_ending_period)

        final_data = _power_function(treat_data, depvar= 'value', treatment_var= 'TREATMENT', 
                                     effect = 25)

        model_1 = _ols_regression_function(final_data, depvar='value', treat_var='TREATMENT',
                                            state_variable='state', time_variable='time')

        model_2 = _ols_regression_function(final_data, depvar='OUTCOME', treat_var='TREATMENT',
                                            state_variable='state', time_variable='time')

        bias = model_1.params['TREATMENT'] - true_beta1_value
        bias_values.append(bias)

        squared_error = (model_1.params['TREATMENT'] - true_beta1_value) ** 2
        squared_error_values.append(squared_error)

        standard_error = model_1.bse['TREATMENT']
        standard_error_values.append(standard_error)
        beta1_estimates.append(model_1.params['TREATMENT'])

        if model_1.pvalues['TREATMENT'] < alpha:
            reject_count_1 += 1

        if model_2.pvalues['TREATMENT'] < alpha:
            reject_count_power += 1    

    results_df = _calculate_statistics(num_simulations= num_simulations ,squared_error_values = 
                                       squared_error_values, beta1_estimates= beta1_estimates, 
                                       bias_values= bias_values, standard_error_values= 
                                       standard_error_values, reject_count_1= reject_count_1 , 
                                       reject_count_power= reject_count_power)

    return results_df



# MONTE CARLO CRSE t(g-1)

def monte_carlo_homogenous_AR1_crse(data_info):
    """
    Perform Monte Carlo simulations for a homogenous AR(1) model with placebo treatment assignment.
    This function performs Monte Carlo simulations on the generated treated data to understand the 
    small-sample properties of the coefficient of Treatment using Cluster Robust Standard Error 
    estimation of the treatment coefficient.
    The data is synthetically generated for each simulation, and placebo treatment is randomly 
    assigned in each run. The small sample statistics of the Treatment coefficient are stored in 
    lists and compiled after the simulations are complete, to produce a DataFrame containing the 
    results.

    Parameters:
    -----------
    data_info : dict
        A dictionary containing the following keys:
        - 'N' : int
            The number of states or groups in the panel data. Must be a positive integer.
        - 'T' : int
            The number of time periods for each state or group. Must be a positive integer
            greater than 1.
        - 'rho' : float
            The AR(1) autoregressive parameter, specifying the persistence of the process. 
            Must be in the range [0, 1].
        - 'num_individuals' : int
            The number of individuals or entities within each state or group. Must be a positive 
            integer.
        - 'cluster_variable' : str
            The variable required for clustering to obtain the cluster robust standard error 
            estimator.
        - 'num_simulations' : int, optional
            The number of Monte Carlo simulations to run.
        - 'alpha' : float, optional
            The significance level for hypothesis testing. Must be in the range (0, 1).
        - 'treatment_starting_period' : int, optional
            The starting period for assigning placebo treatment. Must be an integer not less than 
            the minimum time period in the data. Default is None, which assigns it to the minimum 
            time period in the data.
        - 'treatment_ending_period' : int, optional
            The ending period for assigning placebo treatment. Must be an integer within the data's 
            time range and not less than treat_start_period if specified. Default is None, which 
            assigns it to the maximum time period in the data.

    Raises:
    -------
    ValueError:
        If any of the required parameters are missing or invalid. This function is created using the 
        subfunctions. So if any of the parameters required by the subfunctions is missing or 
        mis-provided, this function will raise a value error.

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the results of the Monte Carlo simulations. The DataFrame includes the 
        following columns:
        - 'Number of Simulations': The total number of simulations performed.
        - 'Type 1 Error': The percentage of simulations where the null hypothesis of no effect was 
        rejected when there is actually no effect.
        - 'Power' : The percentage of times where the null of no effect is rejected after adding a 
           25% effect to the dependent variable.
        - 'Bias': The average bias of the Treatment coefficient estimates.
        - 'MSE': The mean squared error of the Treatment coefficient estimates.
        - 'RMSE': The root mean squared error of the Treatment coefficient estimates.
        - 'Average Standard Error': The average standard error of the Treatment coefficient estimates.
        - 'Confidence Interval Lower Bound': The lower bound of the confidence interval for the 
           Treatment coefficient estimates.
        - 'Confidence Interval Upper Bound': The upper bound of the confidence interval for the 
           Treatment coefficient estimates.

    Notes:
    ------
    - This function generates synthetic panel data using a homogenous AR(1) model, assigns placebo 
      treatment to states, and performs Monte Carlo simulations to analyze the performance of CRSE 
      estimator under different scenarios, namely without a 25% effect and with a 25% effect.
    - The treatment_variable is hard coded to be 'TREATMENT' because the _random_placebo_treatment 
      function generates a column called 'TREATMENT' which is subsequently used.

    Example:
    --------
    To perform Monte Carlo simulations with 5 states, 10 time periods, a rho of 0.5, and 100 
    individuals per state, you can call the function as follows:
        data_dict = {
            'N': 5,
            'T': 10,
            'rho': 0.5,
            'num_individuals': 100
            'cluster_variable': 'state'
        }
    >>> results = monte_carlo_homogenous_AR1_crse(data_dict)

    The results DataFrame will contain simulation results and finite sample statistics on the 
    Treatment coefficient estimates.
    """
    np.random.seed(42)
    
    N = data_info["N"]
    T = data_info["T"]
    rho = data_info["rho"]
    alpha = data_info["alpha"]
    num_individuals = data_info["num_individuals"]
    num_simulations = data_info["num_simulations"]
    treatment_starting_period=data_info["treatment_starting_period"]
    treatment_ending_period=data_info["treatment_ending_period"]
    cluster_variable = data_info['cluster_variable']
    
    reject_count_1 = 0
    reject_count_power = 0
    true_beta1_value = 0
    beta1_estimates = []
    bias_values = []
    squared_error_values = []
    standard_error_values = []

    for _ in range(num_simulations):

        data = _generate_homogenous_ar1_data(N, T, rho, num_individuals)

        treat_data = _random_placebo_treatment(data, 'state', 'time', treatment_starting_period, 
                                                treatment_ending_period)

        final_data = _power_function(treat_data, depvar= 'value', treatment_var= 'TREATMENT', 
                                      effect = 25)

        model_1 = _crse_regression_function(final_data, depvar='value', treat_var='TREATMENT', 
                                             state_variable='state', time_variable='time', cluster_var= cluster_variable)

        model_2 = _crse_regression_function(final_data, depvar='OUTCOME', treat_var='TREATMENT', 
                                             state_variable='state', time_variable='time', cluster_var= cluster_variable)

        bias = model_1.params['TREATMENT'] - true_beta1_value
        bias_values.append(bias)

        squared_error = (model_1.params['TREATMENT'] - true_beta1_value) ** 2
        squared_error_values.append(squared_error)

        standard_error = model_1.bse['TREATMENT']
        standard_error_values.append(standard_error)
        beta1_estimates.append(model_1.params['TREATMENT'])

        if model_1.pvalues['TREATMENT'] < alpha:
            reject_count_1 += 1

        if model_2.pvalues['TREATMENT'] < alpha:
            reject_count_power += 1    

    results_df = _calculate_statistics(num_simulations= num_simulations ,squared_error_values = 
                                        squared_error_values, beta1_estimates= beta1_estimates, 
                                         bias_values= bias_values, standard_error_values= 
                                          standard_error_values, reject_count_1= reject_count_1 , 
                                           reject_count_power= reject_count_power)

    return results_df


# MONTE CARLO RESIDUAL AGGREGATION

def monte_carlo_homogenous_AR1_res_agg(data_info):
    """
    Perform Monte Carlo simulations for a homogenous AR(1) model with placebo treatment assignment.
    This function performs Monte Carlo simulations on the generated treated data to understand the 
    small-sample properties of the coefficient of Treatment using the Residual Aggregation method of 
    regression regression.
    The data is synthetically generated for each simulation, and placebo treatment is randomly 
    assigned in each run. The small sample statistics of the Treatment coefficient are stored in 
    lists and compiled after the simulations are complete, to produce a DataFrame containing the 
    results.

    Parameters:
    -----------
    data_info : dict
        A dictionary containing the following keys:
        - 'N' : int
            The number of states or groups in the panel data. Must be a positive integer.
        - 'T' : int
            The number of time periods for each state or group. Must be a positive integer greater 
            than 1.
        - 'rho' : float
            The AR(1) autoregressive parameter, specifying the persistence of the process. Must be 
            in the range [0, 1].
        - 'num_individuals' : int
            The number of individuals or entities within each state or group. Must be a positive 
            integer.
        - 'num_simulations' : int, optional
            The number of Monte Carlo simulations to run.
        - 'alpha' : float, optional
            The significance level for hypothesis testing. Must be in the range (0, 1).
        - 'treatment_starting_period' : int, optional
            The starting period for assigning placebo treatment. Must be an integer not less than the 
            minimum time period in the data. Default is None, which assigns it to the minimum time 
            period in the data.
        - 'treatment_ending_period' : int, optional
            The ending period for assigning placebo treatment. Must be an integer within the data's 
            time range and not less than treat_start_period if specified. Default is None, which 
            assigns it to the maximum time period in the data.

    Raises:
    -------
    ValueError:
        If any of the required parameters are missing or invalid. This function is created using the 
        subfunctions. So if any of the parameters required by the subfunctions is missing or 
        mis-provided, this function will raise a value error.

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the results of the Monte Carlo simulations. The DataFrame includes the 
        following columns:
        - 'Number of Simulations': The total number of simulations performed.
        - 'Type 1 Error': The percentage of simulations where the null hypothesis of no effect was 
           rejected when there is actually no effect.
        - 'Power' : The percentage of times where the null of no effect is rejected after adding a 
           25% effect to the dependent variable.
        - 'Bias': The average bias of the Treatment coefficient estimates.
        - 'MSE': The mean squared error of the Treatment coefficient estimates.
        - 'RMSE': The root mean squared error of the Treatment coefficient estimates.
        - 'Average Standard Error': The average standard error of the Treatment coefficient estimates.
        - 'Confidence Interval Lower Bound': The lower bound of the confidence interval for the 
           Treatment coefficient estimates.
        - 'Confidence Interval Upper Bound': The upper bound of the confidence interval for the 
           Treatment coefficient estimates.

    Notes:
    ------
    - This function generates synthetic panel data using a homogenous AR(1) model, assigns placebo 
      treatment to states, and performs Monte Carlo simulations to analyze the performance of the
      estimator under different scenarios, namely without a 25% effect and with a 25% effect.
    - The treatment_variable is hard coded to be 'TREATMENT' because the _random_placebo_treatment 
      function generates a column called 'TREATMENT' which is subsequently used.
    Example:
    --------
    To perform Monte Carlo simulations with 5 states, 10 time periods, a rho of 0.5, and 100 
    individuals per state, you can call the function as follows:
        data_dict = {
            'N': 5,
            'T': 10,
            'rho': 0.5,
            'num_individuals': 100
        }
    >>> results = monte_carlo_homogenous_AR1_res_agg(data_dict)

    The results DataFrame will contain simulation results and finite sample statistics on the 
    Treatment coefficient estimates.

    """
    np.random.seed(42)
    
    N = data_info["N"]
    T = data_info["T"]
    rho = data_info["rho"]
    alpha = data_info["alpha"]
    num_individuals = data_info["num_individuals"]
    num_simulations = data_info["num_simulations"]
    treatment_starting_period=data_info["treatment_starting_period"]
    treatment_ending_period=data_info["treatment_ending_period"]

    reject_count_1 = 0
    reject_count_power = 0
    true_beta1_value = 0
    beta1_estimates = []
    bias_values = []
    squared_error_values = []
    standard_error_values = []

    for _ in range(num_simulations):

        data = _generate_homogenous_ar1_data(N, T, rho, num_individuals)

        treat_data = _random_placebo_treatment(data, 'state', 'time', treatment_starting_period, 
                                                treatment_ending_period)

        final_data = _power_function(treat_data, depvar= 'value', treatment_var= 'TREATMENT', 
                                      effect = 25)

        model_1 = _res_agg_regression_function(final_data, depvar='value', treatment_var= 'TREATMENT', 
                                                state_variable='state', time_variable='time')

        model_2 = _res_agg_regression_function(final_data, depvar='OUTCOME', treatment_var= 'TREATMENT', 
                                                state_variable='state', time_variable='time')

        bias = model_1.params['TREATMENT'] - true_beta1_value
        bias_values.append(bias)

        squared_error = (model_1.params['TREATMENT'] - true_beta1_value) ** 2
        squared_error_values.append(squared_error)

        standard_error = model_1.bse['TREATMENT']
        standard_error_values.append(standard_error)
        beta1_estimates.append(model_1.params['TREATMENT'])

        if model_1.pvalues['TREATMENT'] < alpha:
            reject_count_1 += 1

        if model_2.pvalues['TREATMENT'] < alpha:
            reject_count_power += 1    

    results_df = _calculate_statistics(num_simulations= num_simulations ,squared_error_values = 
                                        squared_error_values, beta1_estimates= beta1_estimates, 
                                         bias_values= bias_values, standard_error_values= 
                                          standard_error_values, reject_count_1= reject_count_1 , 
                                           reject_count_power= reject_count_power)

    return results_df


# MONTE CARLO WILD CLUSTER BOOTSTRAPPING

def monte_carlo_homogenous_AR1_wbtest(data_info):
    """
    Perform Monte Carlo simulations for a homogenous AR(1) model with placebo treatment assignment.
    This function performs Monte Carlo simulations on the generated treated data to understand the 
    small-sample properties of the coefficient of Treatment using Wild Cluster Bootstrapping 
    estimation of the treatment coefficient.
    The data is synthetically generated for each simulation, and placebo treatment is randomly 
    assigned in each run. The small sample statistics of the Treatment coefficient are stored in 
    lists and compiled after the simulations are complete, to produce a DataFrame containing the 
    results.

    Parameters:
    -----------
    data_info : dict
        A dictionary containing the following keys:
        - 'N' : int
            The number of states or groups in the panel data. Must be a positive integer.
        - 'T' : int
            The number of time periods for each state or group. Must be a positive integer greater 
            than 1.
        - 'rho' : float
            The AR(1) autoregressive parameter, specifying the persistence of the process. Must be 
            in the range [0, 1].
        - 'num_individuals' : int
            The number of individuals or entities within each state or group. Must be a positive 
            integer.
        - 'cluster_variable' : str
            The variable required for clustering to obtain the cluster robust standard error estimator.
        - 'num_simulations' : int, optional
            The number of Monte Carlo simulations to run.
        - 'alpha' : float, optional
            The significance level for hypothesis testing. Must be in the range (0, 1).
        - 'treatment_starting_period' : int, optional
            The starting period for assigning placebo treatment. Must be an integer not less than the 
            minimum time period in the data. Default is None, which assigns it to the minimum time 
            period in the data.
        - 'treatment_ending_period' : int, optional
            The ending period for assigning placebo treatment. Must be an integer within the data's 
            time range and not less than treat_start_period if specified. Default is None, which
            assigns it to the maximum time period in the data.

    Raises:
    -------
    ValueError:
        If any of the required parameters are missing or invalid. This function is created using 
        the subfunctions. So if any of the parameters required by the subfunctions is missing or 
        mis-provided, this function will raise a value error.

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the results of the Monte Carlo simulations. The DataFrame includes 
        the following columns:
        - 'Number of Simulations': The total number of simulations performed.
        - 'Type 1 Error': The percentage of simulations where the null hypothesis of no effect was 
           rejected when there is actually no effect.
        - 'Power' : The percentage of times where the null of no effect is rejected after adding a 
           25% effect to the dependent variable.
        - 'Bias': The average bias of the Treatment coefficient estimates.
        - 'MSE': The mean squared error of the Treatment coefficient estimates.
        - 'RMSE': The root mean squared error of the Treatment coefficient estimates.
        - 'Average Standard Error': The average standard error of the Treatment coefficient estimates.
        - 'Confidence Interval Lower Bound': The lower bound of the confidence interval for the 
           Treatment coefficient estimates.
        - 'Confidence Interval Upper Bound': The upper bound of the confidence interval for the 
           Treatment coefficient estimates.

    Notes:
    ------
    - This function generates synthetic panel data using a homogenous AR(1) model, assigns placebo 
      treatment to states, and performs Monte Carlo simulations to analyze the performance of wild 
      cluster bootstrapping estimators under different scenarios, namely without a 25% effect and 
      with a 25% effect.
    - The treatment_variable is hard coded to be 'TREATMENT' because the _random_placebo_treatment 
      function generates a column called 'TREATMENT' which is subsequently used.

    Example:
    --------
    To perform Monte Carlo simulations with 5 states, 10 time periods, a rho of 0.5, and 100 
    individuals per state, you can call the function as follows:
        data_dict = {
            'N': 5,
            'T': 10,
            'rho': 0.5,
            'num_individuals': 100
            'cluster_variable': 'state'
        }
    >>> results = monte_carlo_homogenous_AR1_wbtest(data_dict)

    The results DataFrame will contain simulation results and finite sample statistics on the 
    Treatment coefficient estimates.
    """
    np.random.seed(42)
    
    N = data_info["N"]
    T = data_info["T"]
    rho = data_info["rho"]
    alpha = data_info["alpha"]
    num_individuals = data_info["num_individuals"]
    num_simulations = data_info["num_simulations"]
    treatment_starting_period=data_info["treatment_starting_period"]
    treatment_ending_period=data_info["treatment_ending_period"]
    cluster_variable = data_info["cluster_variable"]
    
    reject_count_1 = 0
    reject_count_power = 0
    beta1_estimates = []
    bias_values = []
    squared_error_values = []
    standard_error_values = []
    

    for _ in range(num_simulations):
        data = _generate_homogenous_ar1_data(N, T, rho, num_individuals)

            
        if cluster_variable not in data.columns:
            raise ValueError(f"'{cluster_variable}' is not a column in the data.")

        treat_data = _random_placebo_treatment(data, 'state', 'time', treatment_starting_period, 
                                                treatment_ending_period)
        
        final_data = _power_function(data = treat_data, depvar = 'value', treatment_var= 'TREATMENT', 
                                      effect = 25)

        model_1 = _boot_regression_function(final_data, depvar='value', treat_var= 'TREATMENT', 
                                             cluster_var='state', state_variable= 'state', 
                                              time_variable= 'time')
        
        model_2 = _boot_regression_function(final_data, depvar='OUTCOME', treat_var= 'TREATMENT', 
                                             cluster_var='state', state_variable= 'state', 
                                              time_variable= 'time')

        bias = model_1['Bias'] 
        bias_values.append(bias)

        squared_error = model_1['MSE']
        squared_error_values.append(squared_error)

        standard_error = model_1['Standard Error']
        standard_error_values.append(standard_error)

        beta1_estimates.append(model_1['Observed Coefficient'])

        if model_1['P-Value'] < alpha:
            reject_count_1 += 1
        
        if model_2['P-Value'] < alpha:
            reject_count_power += 1

    results_df = _calculate_statistics(num_simulations= num_simulations ,squared_error_values = 
                                        squared_error_values, beta1_estimates= beta1_estimates, 
                                         bias_values= bias_values, reject_count_1= reject_count_1,
                                          reject_count_power= reject_count_power, 
                                           standard_error_values= standard_error_values)

    return results_df

# MONTE CARLO FGLS

def monte_carlo_homogenous_AR1_fgls(data_info):
    """
    Perform Monte Carlo simulations for a homogenous AR(1) model with placebo treatment assignment.
    This function performs Monte Carlo simulations on the generated treated data to understand 
    the small-sample properties of the coefficient of Treatment using the Feasible Generalised 
    Least Squares method of regression regression.
    The data is synthetically generated for each simulation, and placebo treatment is randomly 
    assigned in each run. The small sample statistics of the Treatment coefficient are stored in 
    lists and compiled after the simulations are complete, to produce a DataFrame containing the 
    results.

    Parameters:
    -----------
    data_info : dict
        A dictionary containing the following keys:
        - 'N' : int
            The number of states or groups in the panel data. Must be a positive integer.
        - 'T' : int
            The number of time periods for each state or group. Must be a positive integer greater 
            than 1.
        - 'rho' : float
            The AR(1) autoregressive parameter, specifying the persistence of the process. 
            Must be in the range [0, 1].
        - 'num_individuals' : int
            The number of individuals or entities within each state or group. 
            Must be a positive integer.
        - 'num_simulations' : int, optional
            The number of Monte Carlo simulations to run.
        - 'alpha' : float, optional
            The significance level for hypothesis testing. Must be in the range (0, 1).
        - 'treatment_starting_period' : int, optional
            The starting period for assigning placebo treatment. Must be an integer not less than 
            the minimum time period in the data. Default is None, which assigns it to the minimum 
            time period in the data.
        - 'treatment_ending_period' : int, optional
            The ending period for assigning placebo treatment. Must be an integer within the data's 
            time range and not less than treat_start_period if specified. Default is None, which 
            assigns it to the maximum time period in the data.

    Raises:
    -------
    ValueError:
        If any of the required parameters are missing or invalid. This function is created using 
        the subfunctions. So if any of the parameters required by the subfunctions is missing or 
        mis-provided, this function will raise a value error.

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the results of the Monte Carlo simulations. The DataFrame includes 
        the following columns:
        - 'Number of Simulations': The total number of simulations performed.
        - 'Type 1 Error': The percentage of simulations where the null hypothesis of no effect was 
           rejected when there is actually no effect.
        - 'Power' : The percentage of times where the null of no effect is rejected after adding a 
           25% effect to the dependent variable.
        - 'Bias': The average bias of the Treatment coefficient estimates.
        - 'MSE': The mean squared error of the Treatment coefficient estimates.
        - 'RMSE': The root mean squared error of the Treatment coefficient estimates.
        - 'Average Standard Error': The average standard error of the Treatment coefficient estimates.
        - 'Confidence Interval Lower Bound': The lower bound of the confidence interval for the 
           Treatment coefficient estimates.
        - 'Confidence Interval Upper Bound': The upper bound of the confidence interval for the 
           Treatment coefficient estimates.

    Notes:
    ------
    - This function generates synthetic panel data using a homogenous AR(1) model, assigns placebo 
      treatment to states, and performs Monte Carlo simulations to analyze the performance of estimator 
      under different scenarios, namely without a 25% effect and with a 25% effect.
    - The treatment_variable is hard coded to be 'TREATMENT' because the _random_placebo_treatment 
      function generates a column called 'TREATMENT' which is subsequently used.

    Example:
    --------
    To perform Monte Carlo simulations with 5 states, 10 time periods, a rho of 0.5, and 100 
    individuals per state, you can call the function as follows:
        data_dict = {
            'N': 5,
            'T': 10,
            'rho': 0.5,
            'num_individuals': 100
        }
    >>> results = monte_carlo_homogenous_AR1_fgls(data_dict)

    The results DataFrame will contain simulation results and finite sample statistics on the 
    Treatment coefficient estimates.

    """
    np.random.seed(42)

    N = data_info["N"]
    T = data_info["T"]
    rho = data_info["rho"]
    alpha = data_info["alpha"]
    num_individuals = data_info["num_individuals"]
    num_simulations = data_info["num_simulations"]
    treatment_starting_period=data_info["treatment_starting_period"]
    treatment_ending_period=data_info["treatment_ending_period"]

    reject_count_1 = 0
    reject_count_power = 0
    true_beta1_value = 0
    beta1_estimates = []
    bias_values = []
    squared_error_values = []
    standard_error_values = []

    for _ in range(num_simulations):

        data = _generate_homogenous_ar1_data(N, T, rho, num_individuals)

        treat_data = _random_placebo_treatment(data, 'state', 'time', treatment_starting_period, 
                                                treatment_ending_period)

        final_data = _power_function(treat_data, depvar= 'value', treatment_var= 'TREATMENT', 
                                      effect = 25)

        model_1 = _fgls_regression_function(final_data, depvar='value', treat_var='TREATMENT', 
                                             state_variable='state', time_variable='time')

        model_2 = _fgls_regression_function(final_data, depvar='OUTCOME', treat_var='TREATMENT', 
                                             state_variable='state', time_variable='time')

        bias = model_1.params['TREATMENT'] - true_beta1_value
        bias_values.append(bias)

        squared_error = (model_1.params['TREATMENT'] - true_beta1_value) ** 2
        squared_error_values.append(squared_error)

        standard_error = model_1.bse['TREATMENT']
        standard_error_values.append(standard_error)
        beta1_estimates.append(model_1.params['TREATMENT'])

        if model_1.pvalues['TREATMENT'] < alpha:
            reject_count_1 += 1

        if model_2.pvalues['TREATMENT'] < alpha:
            reject_count_power += 1    

    results_df = _calculate_statistics(num_simulations= num_simulations ,squared_error_values = 
                                        squared_error_values, beta1_estimates= beta1_estimates, 
                                         bias_values= bias_values, standard_error_values= 
                                          standard_error_values, reject_count_1= reject_count_1 , 
                                           reject_count_power= reject_count_power)

    return results_df
