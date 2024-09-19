import numpy as np
from did.monte_carlo_analysis.subfunctions import _random_placebo_treatment, _ols_regression_function, _calculate_statistics, _crse_regression_function, _res_agg_regression_function, _boot_regression_function, _power_function, _fgls_regression_function


# MONTE CARLO OLS

def cps_ols_monte_carlo(data, data_info) :
    """
    Perform Monte Carlo simulations for Ordinary Least Squares (OLS) regression using
    CPS (Current Population Survey) data.

    This function conducts Monte Carlo simulations to analyze the properties of OLS estimators 
    applied to CPS data with placebo treatment assignment.It generates treated data by assigning 
    placebo treatment to approximately half of the observations in a staggered manner within a 
    specified time range.The simulations are performed multiple times to evaluate the small-sample
    properties of the coefficient of Treatment using OLS regression.

    Parameters:
    -----------
    data : pandas.DataFrame
        The CPS data containing the following columns:
        - 'STATEFIP' : int
            State Federal Information Processing Standard (FIPS) code.
        - 'YEAR' : int
            Year of the observation.
        - 'Residual_wage' : float
            Residual wage data.
        - 'TREATMENT' : int
            Binary treatment indicator variable (0 for control, 1 for treatment).

    data_info : dict
        A dictionary containing the following keys:
        - 'num_simulations' : int
            The number of Monte Carlo simulations to run.
        - 'alpha' : float
            The significance level for hypothesis testing.

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the results of the Monte Carlo simulations. The DataFrame includes 
        the following columns:
        - 'Number of Simulations': The total number of simulations performed.
        - 'Type 1 Error': The percentage of simulations where the null hypothesis of no effect
           was rejected when there is actually no effect.
        - 'Power' : The percentage of times where the null of no effect is rejected after
           adding a 25% effect to the dependent variable.
        - 'Bias': The average bias of the Treatment coefficient estimates.
        - 'MSE': The mean squared error of the Treatment coefficient estimates.
        - 'RMSE': The root mean squared error of the Treatment coefficient estimates.
        - 'Average Standard Error': The average standard error of the Treatment coefficient
           estimates.
        - 'Confidence Interval Lower Bound': The lower bound of the confidence interval
           for the Treatment coefficient estimates.
        - 'Confidence Interval Upper Bound': The upper bound of the confidence interval 
           for the Treatment coefficient estimates.

    Notes:
    ------
    - This function assumes that the input data contains columns 'STATEFIP', 'YEAR', 'Residual_wage',
      and 'TREATMENT'.
    - Placebo treatment is assigned to observations within the specified time range 
     ('treatment_starting_period' to 'treatment_ending_period'). To suit our analysis the
     treatment starting date is hardcoded to be 1985 and the treatment ending date is set to be 1995.
    - The results provide insights into the performance of OLS estimators under different
      scenarios and treatment assignments.

    Example:
    --------
    To perform Monte Carlo simulations with 100 iterations and a significance level of 0.05,
    you can call the function as follows:

    >>> data_info = {'num_simulations': 100, 'alpha': 0.05}
    >>> results = cps_ols_monte_carlo(data, data_info)

    The results DataFrame will contain simulation results and finite sample statistics
    on the Treatment coefficient estimates.
    """
    np.random.seed(42)
    df = data.copy()

    num_simulations = data_info["num_simulations"]
    alpha = data_info["alpha"]
    bias_values = []
    squared_error_values = []
    standard_error_values = []
    beta1_estimates = []
    reject_count_1 = 0
    reject_count_power = 0
    true_beta1_value = 0

    for _ in range(num_simulations):

        treated_data = _random_placebo_treatment(df, state_variable= 'STATEFIP',
                                          time_variable= 'YEAR', treatment_starting_period= 1985,
                                          treatment_ending_period= 1995)

        final_data = _power_function(data =treated_data, depvar = 'Residual_wage',
                                      treatment_var= 'TREATMENT', effect = 25)

        model_1 = _ols_regression_function(final_data, depvar ='Residual_wage',
                                            treat_var= "TREATMENT", state_variable= 'STATEFIP',
                                              time_variable= 'YEAR' )

        model_2 = _ols_regression_function(final_data, depvar ='OUTCOME', treat_var= 'TREATMENT' ,
                                            state_variable= 'STATEFIP', time_variable= 'YEAR' )

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
    
    results_df = _calculate_statistics(num_simulations = num_simulations ,squared_error_values =
                                        squared_error_values, beta1_estimates= beta1_estimates,
                                         bias_values= bias_values, reject_count_1= reject_count_1,
                                          reject_count_power = reject_count_power,
                                           standard_error_values= standard_error_values)

    return results_df


# MONTE CARLO CRSE

def cps_crse_monte_carlo(data,data_info) :
    """
    Perform Monte Carlo simulations for Cluster-Robust Standard Errors (CRSE) regression using CPS 
    (Current Population Survey) data.

    This function conducts Monte Carlo simulations to analyze the properties of regression models with
    Cluster-Robust Standard Errors applied to CPS data with placebo treatment assignment.
    It generates treated data by assigning placebo treatment to approximately half of the observations
    in a staggered manner within a specified time range. The simulations are performed multiple times
    to evaluate the small-sample properties of the coefficient of Treatment using CRSE regression.

    Parameters:
    -----------
    data : pandas.DataFrame
        The CPS data containing the following columns:
        - 'STATEFIP' : int
            State Federal Information Processing Standard (FIPS) code.
        - 'YEAR' : int
            Year of the observation.
        - 'Residual_wage' : float
            Residual wage data.
        - 'TREATMENT' : int
            Binary treatment indicator variable (0 for control, 1 for treatment).

    data_info : dict
        A dictionary containing the following keys:
        - 'num_simulations' : int
            The number of Monte Carlo simulations to run.
        - 'alpha' : float
            The significance level for hypothesis testing.

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the results of the Monte Carlo simulations.
        The DataFrame includes the following columns:
        - 'Number of Simulations': The total number of simulations performed.
        - 'Type 1 Error': The percentage of simulations where the null hypothesis of no effect
           was rejected when there is actually no effect.
        - 'Power' : The percentage of times where the null of no effect is rejected after adding 
           a 25% effect to the dependent variable.
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
    - This function assumes that the input data contains columns 'STATEFIP', 'YEAR', 'Residual_wage',
      and 'TREATMENT'.
    - Placebo treatment is assigned to observations within the specified time range
      ('treatment_starting_period' to 'treatment_ending_period'). To suit our analysis the treatment
      starting date is hardcoded to be 1985 and the treatment ending date is set to be 1995.
    - The results provide insights into the performance of CRSE regression models under different
      scenarios and treatment assignments.

    Example:
    --------
    To perform Monte Carlo simulations with 100 iterations and a significance level of 0.05,
    you can call the function as follows:

    >>> data_info = {'num_simulations': 100, 'alpha': 0.05}
    >>> results = cps_crse_monte_carlo(data, data_info)

    The results DataFrame will contain simulation results and finite sample statistics on
    the Treatment coefficient estimates.
    """
    np.random.seed(42)
    df = data.copy()

    num_simulations = data_info["num_simulations"]
    alpha = data_info["alpha"]

    bias_values = []
    squared_error_values = []
    standard_error_values = []
    beta1_estimates = []
    reject_count_1 = 0
    reject_count_power = 0
    true_beta1_value = 0

    for _ in range(num_simulations):

        treated_data = _random_placebo_treatment(df, state_variable= 'STATEFIP', time_variable= 'YEAR',
                                                  treatment_starting_period= 1985,
                                                   treatment_ending_period= 1995)

        final_data = _power_function(data =treated_data, depvar = 'Residual_wage',
                                      treatment_var= 'TREATMENT', effect = 25)

        model_1 = _crse_regression_function(final_data, depvar ='Residual_wage',
                                             treat_var= "TREATMENT" , state_variable= 'STATEFIP',
                                               time_variable= 'YEAR' , cluster_var= 'STATEFIP')

        model_2 = _crse_regression_function(final_data, depvar ='OUTCOME', treat_var= 'TREATMENT',
                                             state_variable= 'STATEFIP', time_variable= 'YEAR' ,
                                               cluster_var='STATEFIP' )

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
                                         bias_values= bias_values, reject_count_1= reject_count_1,
                                          reject_count_power = reject_count_power, 
                                           standard_error_values= standard_error_values)

    return results_df


# MONTE CARLO RESIDUAL AGGREGATION

def cps_res_agg_monte_carlo(data,data_info) :
    """
    Perform Monte Carlo simulations for regression with residual aggregation using CPS 
    (Current Population Survey) data.

    This function conducts Monte Carlo simulations to analyze the properties of regression models
    with residual aggregation applied to CPS data with placebo treatment assignment. It generates 
    treated data by assigning placebo treatment to approximately half of the observations in 
    a staggered manner within a specified time range. The simulations are performed multiple 
    times to evaluate the small-sample properties of the coefficient of Treatment using
    regression with residual aggregation.

    Parameters:
    -----------
    data : pandas.DataFrame
        The CPS data containing the following columns:
        - 'STATEFIP' : int
            State Federal Information Processing Standard (FIPS) code.
        - 'YEAR' : int
            Year of the observation.
        - 'Residual_wage' : float
            Residual wage data.
        - 'TREATMENT' : int
            Binary treatment indicator variable (0 for control, 1 for treatment).

    data_info : dict
        A dictionary containing the following keys:
        - 'num_simulations' : int
            The number of Monte Carlo simulations to run.
        - 'alpha' : float
            The significance level for hypothesis testing.

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the results of the Monte Carlo simulations. 
        The DataFrame includes the following columns:
        - 'Number of Simulations': The total number of simulations performed.
        - 'Type 1 Error': The percentage of simulations where the null hypothesis of no effect 
           was rejected when there is actually no effect.
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
    - This function assumes that the input data contains columns 'STATEFIP', 'YEAR', 'Residual_wage',
      and 'TREATMENT'.
    - Placebo treatment is assigned to observations within the specified time range 
      ('treatment_starting_period' to 'treatment_ending_period'). To suit our analysis the
      treatment starting date is hardcoded to be 1985 and the treatment ending date is set to be 1995.
    - The results provide insights into the performance of regression models with residual aggregation
      under different scenarios and treatment assignments.

    Example:
    --------
    To perform Monte Carlo simulations with 100 iterations and a significance level of 0.05,
    you can call the function as follows:

    >>> data_info = {'num_simulations': 100, 'alpha': 0.05}
    >>> results = cps_res_agg_monte_carlo(data, data_info)

    The results DataFrame will contain simulation results and finite sample statistics 
    on the Treatment coefficient estimates.

    """
    np.random.seed(42)
    df = data.copy()

    num_simulations = data_info["num_simulations"]
    alpha = data_info["alpha"]
    bias_values = []
    squared_error_values = []
    standard_error_values = []
    beta1_estimates = []
    reject_count_1 = 0
    reject_count_power = 0
    true_beta1_value = 0

    for _ in range(num_simulations):

        treated_data = _random_placebo_treatment(df, state_variable= 'STATEFIP', time_variable='YEAR',
                                                  treatment_starting_period= 1985,
                                                   treatment_ending_period= 1995)

        final_data = _power_function(data =treated_data, depvar = 'Residual_wage',
                                      treatment_var= 'TREATMENT', effect = 25)

        model_1 = _res_agg_regression_function(final_data, depvar ='Residual_wage', 
                                                treatment_var= 'TREATMENT', state_variable='STATEFIP',
                                                 time_variable= 'YEAR' )

        model_2 = _res_agg_regression_function(final_data, depvar ='OUTCOME', 
                                                treatment_var = 'TREATMENT',state_variable= 'STATEFIP', 
                                                 time_variable= 'YEAR' )

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
    
    results_df = _calculate_statistics(num_simulations= num_simulations ,
                                       squared_error_values = squared_error_values,
                                         beta1_estimates= beta1_estimates, bias_values= bias_values,
                                           reject_count_1= reject_count_1,reject_count_power = 
                                            reject_count_power, standard_error_values= 
                                             standard_error_values)

    return results_df


# MONTE CARLO WILD CLUSTER BOOTSTRAPPING

def cps_wbtest_monte_carlo(data,data_info) :
    """
    Perform Monte Carlo simulations for Wild Cluster Bootstrapping (WB) regression using CPS 
    (Current Population Survey) data.

    This function conducts Monte Carlo simulations to analyze the properties of regression models 
    with Wild Cluster Bootstrapping applied to CPS data with placebo treatment assignment.
    It generates treated data by assigning placebo treatment to approximately half of the observations 
    in a staggered manner within a specified time range.
    The simulations are performed multiple times to evaluate the small-sample properties of the 
    coefficient of Treatment using Wild Cluster Bootstrapping regression.

    Parameters:
    -----------
    data : pandas.DataFrame
        The CPS data containing the following columns:
        - 'STATEFIP' : int
            State Federal Information Processing Standard (FIPS) code.
        - 'YEAR' : int
            Year of the observation.
        - 'Residual_wage' : float
            Residual wage data.
        - 'TREATMENT' : int
            Binary treatment indicator variable (0 for control, 1 for treatment).

    data_info : dict
        A dictionary containing the following keys:
        - 'num_simulations' : int
            The number of Monte Carlo simulations to run.
        - 'alpha' : float
            The significance level for hypothesis testing.

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
    - This function assumes that the input data contains columns 'STATEFIP', 'YEAR', 'Residual_wage', 
      and 'TREATMENT'.
    - Placebo treatment is assigned to observations within the specified time range 
      ('treatment_starting_period' to 'treatment_ending_period'). To suit our analysis the treatment 
      starting date is hardcoded to be 1985 and the treatment ending date is set to be 1995.
    - The results provide insights into the performance of Wild Cluster Bootstrapping regression 
      models under different scenarios and treatment assignments.

    Example:
    --------
    To perform Monte Carlo simulations with 100 iterations and a significance level of 0.05, you can 
    call the function as follows:

    >>> data_info = {'num_simulations': 100, 'alpha': 0.05}
    >>> results = cps_wbtest_monte_carlo(data, data_info)

    The results DataFrame will contain simulation results and finite sample statistics on the 
    Treatment coefficient estimates.
    """
    np.random.seed(42)

    df = data.copy()

    num_simulations = data_info["num_simulations"]
    alpha = data_info["alpha"]
    bias_values = []
    squared_error_values = []
    standard_error_values = []
    beta1_estimates = []
    reject_count_1 = 0
    reject_count_power = 0

    for _ in range(num_simulations):

        treated_data = _random_placebo_treatment(df, state_variable= 'STATEFIP', time_variable= 'YEAR',
                                                  treatment_starting_period= 1985,
                                                   treatment_ending_period= 1995)

        final_data = _power_function(data = treated_data, depvar = 'Residual_wage', 
                                      treatment_var= 'TREATMENT', effect= 25)

        model_1 = _boot_regression_function(final_data, depvar ='Residual_wage', 
                                             treat_var= 'TREATMENT' , cluster_var= 'STATEFIP', 
                                              state_variable= 'STATEFIP', time_variable= 'YEAR')

        model_2 = _boot_regression_function(final_data, depvar ='OUTCOME', treat_var= 'TREATMENT', 
                                             cluster_var= 'STATEFIP', state_variable= 'STATEFIP', 
                                              time_variable= 'YEAR' )

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
                                          reject_count_power = reject_count_power, standard_error_values= 
                                           standard_error_values)

    return results_df


# MONTE CARLO FGLS

def cps_fgls_monte_carlo(data,data_info) :
    """
    Perform Monte Carlo simulations for Feasible Generalized Least Squares (FGLS) regression using 
    CPS (Current Population Survey) data.

    This function conducts Monte Carlo simulations to analyze the properties of regression models with 
    Feasible Generalized Least Squares applied to CPS data with placebo treatment assignment.
    It generates treated data by assigning placebo treatment to approximately half of the observations 
    in a staggered manner within a specified time range.
    The simulations are performed multiple times to evaluate the small-sample properties of the 
    coefficient of Treatment using FGLS regression.

    Parameters:
    -----------
    data : pandas.DataFrame
        The CPS data containing the following columns:
        - 'STATEFIP' : int
            State Federal Information Processing Standard (FIPS) code.
        - 'YEAR' : int
            Year of the observation.
        - 'Residual_wage' : float
            Residual wage data.
        - 'TREATMENT' : int
            Binary treatment indicator variable (0 for control, 1 for treatment).

    data_info : dict
        A dictionary containing the following keys:
        - 'num_simulations' : int
            The number of Monte Carlo simulations to run.
        - 'alpha' : float
            The significance level for hypothesis testing.

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the results of the Monte Carlo simulations. The DataFrame includes the 
        following columns:
        - 'Number of Simulations': The total number of simulations performed.
        - 'Type 1 Error': The percentage of simulations where the null hypothesis of no effect was 
           rejected when there is actually no effect.
        - 'Power' : The percentage of times where the null of no effect is rejected after adding a 25% 
           effect to the dependent variable.
        - 'Bias': The average bias of the Treatment coefficient estimates.
        - 'MSE': The mean squared error of the Treatment coefficient estimates.
        - 'RMSE': The root mean squared error of the Treatment coefficient estimates.
        - 'Average Standard Error': The average standard error of the Treatment coefficient estimates.
        - 'Confidence Interval Lower Bound': The lower bound of the confidence interval for the Treatment 
           coefficient estimates.
        - 'Confidence Interval Upper Bound': The upper bound of the confidence interval for the Treatment 
           coefficient estimates.

    Notes:
    ------
    - This function assumes that the input data contains columns 'STATEFIP', 'YEAR', 'Residual_wage', 
      and 'TREATMENT'.
    - Placebo treatment is assigned to observations within the specified time range 
      ('treatment_starting_period' to 'treatment_ending_period'). To suit our analysis the treatment 
      starting date is hardcoded to be 1985 and the treatment ending date is set to be 1995.
    - The results provide insights into the performance of FGLS regression models under different 
      scenarios and treatment assignments.

    Example:
    --------
    To perform Monte Carlo simulations with 100 iterations and a significance level of 0.05, you can 
    call the function as follows:

    >>> data_info = {'num_simulations': 100, 'alpha': 0.05}
    >>> results = cps_fgls_monte_carlo(data, data_info)

    The results DataFrame will contain simulation results and finite sample statistics on the 
    Treatment coefficient estimates.
    """
    np.random.seed(42)

    df = data.copy()
    
    num_simulations = data_info["num_simulations"]
    alpha = data_info["alpha"]

    bias_values = []
    squared_error_values = []
    standard_error_values = []
    beta1_estimates = []
    reject_count_1 = 0
    reject_count_power = 0
    true_beta1_value = 0

    for _ in range(num_simulations):

        treated_data = _random_placebo_treatment(df, state_variable= 'STATEFIP', time_variable= 'YEAR', 
                                                  treatment_starting_period= 1985,
                                                   treatment_ending_period= 1995)

        final_data = _power_function(data=treated_data, depvar = 'Residual_wage', 
                                      treatment_var= 'TREATMENT', effect = 25)

        model_1 = _fgls_regression_function(final_data, depvar ='Residual_wage', treat_var= "TREATMENT", 
                                             state_variable= 'STATEFIP', time_variable= 'YEAR' )

        model_2 = _fgls_regression_function(final_data, depvar ='OUTCOME', treat_var= 'TREATMENT' , 
                                             state_variable= 'STATEFIP', time_variable= 'YEAR' )

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
                                         bias_values= bias_values, reject_count_1= reject_count_1,
                                          reject_count_power = reject_count_power, 
                                           standard_error_values= standard_error_values)

    return results_df