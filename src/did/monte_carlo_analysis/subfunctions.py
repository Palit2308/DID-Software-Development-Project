import numpy as np
import pandas as pd
import statsmodels.api as sm

# PLACEBO INTERVENTION

def _random_placebo_treatment(data, state_variable, time_variable, treatment_starting_period=None, 
                              treatment_ending_period=None):
    """
    Assign random placebo treatment indicators to a panel data based on specified criteria.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input panel data containing state, time, and other relevant variables.
    state_variable : str
        The name of the column in the DataFrame that represents the state or group identifier.
    time_variable : str
        The name of the column in the DataFrame that represents the time period identifier.
    treatment_starting_period : int, optional
        The starting period for assigning placebo treatment. Default is None which takes in the 
        minimum of the provided time variable to be the treatment starting date.
    treatment_ending_period : int, optional
        The ending period for assigning placebo treatment. Default is None which takes in the 
        maximum of the provided time variable to be the treatment ending date.

    Returns:
    --------
    data : pandas.DataFrame
        The input DataFrame with an additional 'TREATMENT' column indicating placebo treatment 
        assignment:
        - 'TREATMENT' column: 1 if the state and time period meet the placebo treatment criteria, 
           else 0.

    Notes:
    ------
    - This function assigns placebo treatment indicators to states and time periods based on a 
      staggered approach. 
    - Half of the unique states are randomly selected for placebo treatment, and treatment periods 
      for each treated state, are randomized within the specified range from a uniform distribution 
      of U ~ [treatment_starting_period, treatment_ending_period].
    - The treatment indicator (the 'TREATMENT' column) gets a value 1 if the for the treated states 
      from the treatment starting date of the state till the end of the time variable.
    - The function assumes the input data is in long format with at least 2 unique states and 2 
      unique time periods.

    Raises:
    -------
    ValueError:
        - If 'state_variable' or 'time_variable' is not provided as strings representing columns of 
          the data.
        - If 'state_variable' or 'time_variable' is not found in the DataFrame.
        - If data is not in long format with at least 2 unique states and 2 unique time periods.
        - If 'treatment_starting_period' or 'treatment_ending_period' is not an integer.
        - If 'treatment_starting_period' is less than the minimum of the provided time variable.
        - If 'treatment_ending_period' is greater than the maximum of the provided time variable or 
          less than 'treatment_starting_period'.    

    Examples:
    ---------
    # Example 1: Assign placebo treatment indicators with default treatment periods
    placebo_data = _random_placebo_treatment(data=my_panel_data, state_variable='State', 
    time_variable='Year')

    # Example 2: Assign placebo treatment indicators with custom treatment periods
    placebo_data = _random_placebo_treatment(data=my_panel_data, state_variable='State', 
    time_variable='Year', treatment_starting_period=2000, treatment_ending_period=2010)

    """

    if not isinstance(state_variable, str) or not isinstance(time_variable, str):
        raise ValueError("state_variable and time_variable must be provided as strings representing columns of the data (inside quotes).")

    if state_variable not in data.columns or time_variable not in data.columns:
        raise ValueError("state_variable and time_variable must be columns in the data.")

    if data[state_variable].nunique() <= 1 or data[time_variable].nunique() <= 1:
        raise ValueError("Data must be in long format with at least 2 unique states and 2 unique time periods.")

    if treatment_starting_period is not None:
        if not isinstance(treatment_starting_period, int):
            raise ValueError("treatment_starting_period must be an integer.")

        if treatment_starting_period < data[time_variable].min():
            raise ValueError("treatment_starting_period must be not less than data[time_variable].min().")

    if treatment_ending_period is not None:
        if not isinstance(treatment_ending_period, int) :
            raise ValueError("treatment_ending_period must be an integer.")
        
        if treatment_ending_period > data[time_variable].max():
            raise ValueError("treatment_ending_period must be within the data's time range.")

        if treatment_starting_period is not None and treatment_ending_period < treatment_starting_period:
            raise ValueError("treatment_ending_period must not be less than treatment_starting_period if specified.")

    if treatment_starting_period is None:
        treatment_starting_period = data[time_variable].min()

    if treatment_ending_period is None:
        treatment_ending_period = data[time_variable].max()

    states = data[state_variable].unique()
    treatment_states = np.random.choice(states, size=len(states)//2, replace=False)
    treatment_years = np.random.choice(range(treatment_starting_period, treatment_ending_period), 
                                       size=len(treatment_states), replace=True)
    state_to_treatment_year = dict(zip(treatment_states, treatment_years))

    data['TREATMENT'] = data.apply(lambda x: int(x[state_variable] in treatment_states and 
                                                 x[time_variable] >= state_to_treatment_year[x[state_variable]]), 
                                                 axis=1)

    return data

# POWER FUNCTION

def _power_function(data, depvar, treatment_var, effect):
    """
    Introduces an effect on the dependent variable to capture the treatment effect.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing the data.
    depvar : str
        The name of the dependent variable column.
    treatment_var : str
        The name of the treatment variable column.
    effect : float, int
        The effect size, representing the percentage change in the dependent variable if the 
        treatment variable is 1.

    Returns:
    --------
    data : pandas.DataFrame
        The input DataFrame with an additional column called 'OUTCOME' containing the power 
        transformed dependent variable.

    Raises:
    -------
    ValueError:
        - If 'depvar', 'treatment_var', or 'effect' is not provided as expected.
        - If either 'depvar' or 'treatment_var' is not found in the DataFrame.
        - If 'treatment_var' column contains values other than 0 or 1.
        - If 'effect' is not an integer or float.

    Example:
    ---------
    If 'treatment_var' is 1, the dependent variable ('depvar') is multiplied by (1 + effect/100).

    Examples:
    ---------
    # Example 1: Apply a 10% increase to 'depvar' when 'treatment_var' is 1
    transformed_data = _power_function(data=my_dataframe, depvar='Revenue', 
    treatment_var='Treatment', effect=10.0)

    # Example 2: Apply a 5% decrease to 'depvar' when 'treatment_var' is 1
    transformed_data = _power_function(data=my_dataframe, depvar='Profit', 
    treatment_var='Treatment', effect=-5.0)

    """
    
    if not isinstance(depvar, str):
        raise ValueError("'depvar' must be provided as a string.")
    
    if not isinstance(treatment_var, str):
        raise ValueError("'treatment_var' must be provided as a string.")

    if depvar not in data.columns:
        raise ValueError(f"'{depvar}' column not found in the DataFrame.")
    
    if treatment_var not in data.columns:
        raise ValueError(f"'{treatment_var}' column not found in the DataFrame.")

    if not set(data[treatment_var].unique()).issubset({0, 1}):
        raise ValueError(f"'{treatment_var}' column must contain only 0 and 1 values.")

    if not isinstance(effect, (int, float)):
        raise ValueError("'effect' must be a numeric type (int or float).")

    data['OUTCOME'] = data.apply(lambda x: x[depvar] * (1 + effect/100) if x[treatment_var] == 1 
                                 else x[depvar], axis=1)

    return data

#Small Sample Statistics

def _calculate_statistics(num_simulations, bias_values, squared_error_values, standard_error_values, 
                          beta1_estimates, reject_count_1, reject_count_power):
    """
    Calculate various statistics based on input data and parameters.

    Parameters:
    -----------
    num_simulations : int
        Number of simulations (a positive integer greater than 0).
    bias_values : list of float
        List of bias values (each value should be a floating-point number).
    squared_error_values : list of float
        List of squared error values (each value should be a floating-point number).
    standard_error_values : list of float
        List of standard error values (each value should be a floating-point number).
    beta1_estimates : list of float
        List of beta1 estimates (each value should be a floating-point number).
    reject_count_1 : int
        Count of rejected null hypotheses (a non-negative integer) for 0 effect size, to 
        calculate the Type 1 Error.
    reject_count_power : int
        Count of rejected null hypotheses (a non-negative integer) for the desired effect size, 
        to calculate the Power of the test.

    Returns:
    --------
    results_df : DataFrame
        Pandas DataFrame containing calculated statistics.

    Notes:
    ------
    This function calculates various statistical measures such as bias, mean squared error (MSE), 
    root mean squared error (RMSE),
    average standard error, and confidence intervals based on the provided data and parameters.

    It also reports the percentage of rejected null hypotheses, the number of simulations, and the
    data generation process and method used.

    Raises:
    -------
    ValueError:
        - If any of the input parameters do not meet the specified criteria (e.g., data type, 
          value range).

    Example usage:
    --------------
    num_simulations = 1000
    bias_values = [0.1, 0.2, 0.3]
    squared_error_values = [0.01, 0.04, 0.09]
    standard_error_values = [0.05, 0.06, 0.07]
    beta1_estimates = [1.2, 1.3, 1.4]
    reject_count_1 = 150
    reject_count_power = 200

    results = _calculate_statistics(num_simulations, bias_values, squared_error_values, 
                                    standard_error_values, beta1_estimates, reject_count_1, 
                                    reject_count_power)

    """
    if not bias_values or not squared_error_values or not standard_error_values or not beta1_estimates:
        raise ValueError("None of the input lists can be empty")

    if not isinstance(bias_values, list) or not all(isinstance(val, float) for val in bias_values):
        raise ValueError("bias_values must be a list of floats")
    
    if not isinstance(squared_error_values, list) or not all(isinstance(val, float) for val in squared_error_values):
        raise ValueError("squared_error_values must be a list of floats")
    
    if not isinstance(standard_error_values, list) or not all(isinstance(val, float) for val in standard_error_values):
        raise ValueError("standard_error_values must be a list of floats")
    
    if not isinstance(beta1_estimates, list) or not all(isinstance(val, float) for val in beta1_estimates):
        raise ValueError("beta1_estimates must be a list of floats")

    if not isinstance(num_simulations, int) or num_simulations <= 0:
        raise ValueError("num_simulations must be a positive integer greater than 0")
    
    if not isinstance(reject_count_1, int) or reject_count_1 < 0:
        raise ValueError("reject_count must be a non-negative integer")
    
    if not isinstance(reject_count_power, int) or reject_count_power < 0:
        raise ValueError("reject_count_power must be a non-negative integer")

    average_bias = np.mean(bias_values)
    average_mse = np.mean(squared_error_values)
    average_rmse = np.sqrt(average_mse)
    average_standard_error = np.mean(standard_error_values)
    std_error_beta_distribution = np.std(beta1_estimates)
    confidence_interval = (
        np.mean(beta1_estimates) - 1.96 * std_error_beta_distribution,
        np.mean(beta1_estimates) + 1.96 * std_error_beta_distribution
    )
    reject_rate_1 = reject_count_1 / num_simulations
    reject_rate_2 = reject_count_power / num_simulations
    lower_bound_conf_int = confidence_interval[0]
    upper_bound_conf_int = confidence_interval[1]

    results_dict = {
        "Number of Simulations": num_simulations,
        "Type 1 Error in percentage": f'{reject_rate_1 * 100}',
        "Power in percentage" : f'{reject_rate_2 * 100}',
        "Bias": average_bias,
        "MSE": average_mse,
        "RMSE": average_rmse,
        "Average Standard Error": average_standard_error,
        "Confidence Interval Lower Bound": lower_bound_conf_int,
        "Confidence Interval Upper Bound": upper_bound_conf_int
    }

    data = list(results_dict.items())
    df = pd.DataFrame(data, columns=['Variable', 'Value'])

    return df

# OLS REGRESSION FUNCTION

def _ols_regression_function(data, depvar, treat_var, indvar = None, state_variable=None, 
                             time_variable=None):
    """
    Perform ordinary least squares (OLS) regression on panel data.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input panel data containing dependent, treatment, independent, state, and time variables.
    depvar : str
        The name of the column in the DataFrame that represents the dependent variable.
    treat_var : str
        The name of the column in the DataFrame that represents the treatment variable.
    indvar : list of str, optional
        A list of names of columns in the DataFrame that represent the independent variables.
        Default is None (no independent variables).
    state_variable : str, optional
        The name of the column in the DataFrame that represents the state or group identifier 
        (fixed effect).
        Default is None (no state fixed effect).
    time_variable : str, optional
        The name of the column in the DataFrame that represents the time period identifier 
        (fixed effect).
        Default is None (no time fixed effect).

    Returns:
    --------
    model : statsmodels.regression.linear_model.OLSResults
        Model of the OLS regression.

    Raises:
    -------
    ValueError:
        If any of the input parameters do not meet the specified criteria:
        - If 'depvar' is not a column of the data or is not provided within quotes.
        - If 'indvar' is provided but it is not provided in a list of strings(within quotes), 
          or any element in 'indvar' is not a column of the data.
        - If 'treat_var' is not a column of the data or is not provided within quotes.
        - If 'state_variable' is provided and is not a column of the data or is not provided 
          within quotes.
        - If 'time_variable' is provided and is not a column of the data or is not provided 
          within quotes.
        - If 'treat_var' column contains values other than 0 or 1.

    Examples:
    ---------
    # Example 1: OLS regression with only a dependent variable and a treatment variable
    model = _ols_regression_function(data=my_panel_data, depvar='Y', treat_var='TREATMENT')

    # Example 2: OLS regression with additional independent variables and fixed effects
    model = _ols_regression_function(data=my_panel_data, depvar='Y', treat_var='TREATMENT', 
                                     indvar=['X1', 'X2'], state_variable='State', 
                                     time_variable='Year')

    """
    
    if not isinstance(depvar, str):
        raise ValueError("depvar must be provided as a string as a column of the data (inside quotes).")

    if indvar is not None:
        if not isinstance(indvar,list):
            raise ValueError("indvar must be provided as a list of strings representing columns of the data (inside quotes)")    
        if not all(isinstance(var, str) for var in indvar):
            raise ValueError("All elements in indvar must be provided as strings as columns of the data (inside quotes).")
        
    if indvar is not None:
        for var in indvar:
            if var not in data.columns:
                    raise ValueError(f"'{var}' is not a column in the data.")
        
    if depvar not in data.columns:
        raise ValueError(f"'{depvar}' is not a column in the data.")
    
    if treat_var is None:
        raise ValueError("treatment_var must be provided as a string representing a column of the data (inside quotes).")
    
    if not isinstance(treat_var, str):
        raise ValueError("treat_var must be provided as a string as a column of the data (inside quotes).")

    if treat_var not in data.columns:
            raise ValueError(f"'{treat_var}' column not found in the DataFrame.")

    if not set(data[treat_var].unique()).issubset({0, 1}):
        raise ValueError(f"'{treat_var}' column must contain only 0 and 1 values.")

    if state_variable is not None and state_variable not in data.columns:
        raise ValueError(f"'{state_variable}' is not a column in the data")
    
    if time_variable is not None and time_variable not in data.columns:
        raise ValueError(f"'{time_variable}' is not a column in the data")
     
    if indvar is not None:
        X = data[indvar + [treat_var]]  

    if indvar is None:
        X= data[treat_var]    
    
    X = sm.add_constant(X)
    y = data[depvar]

    if state_variable is not None:
        state_dummies = pd.get_dummies(data[state_variable], prefix='state', 
                                       drop_first=True).astype(int)
        X = pd.concat([X, state_dummies], axis=1)

    if time_variable is not None:
        time_dummies = pd.get_dummies(data[time_variable], prefix='time', 
                                      drop_first=True).astype(int)
        X = pd.concat([X, time_dummies], axis=1)

    model = sm.OLS(y, X).fit()
    

    return model


# CRSE T(G-1) REGRESSION FUNCTION

def _crse_regression_function(data, depvar, treat_var,cluster_var, indvar = None, state_variable=None, 
                              time_variable=None):
    """
    Perform ordinary least squares (OLS) regression with cluster-robust standard errors (CRSE) on 
    panel data and the critical values are taken from a t distribution with G-1 degrees of freedom 
    where G represents the number of clusters.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input panel data containing dependent, independent, state, and time variables.
    depvar : str
        The name of the column in the DataFrame that represents the dependent variable.
    treat_var : str
        The name of the column in the DataFrame that represents the treatment variable.
    cluster_var : str
        The name of the column in the DataFrame that represents the cluster variable for CRSE.
    indvar : list of str, optional
        A list of names of columns in the DataFrame that represent the independent variables.
        Default is None (no independent variables).
    state_variable : str, optional
        The name of the column in the DataFrame that represents the state or group identifier 
        (fixed effect).
        Default is None (no state fixed effect).
    time_variable : str, optional
        The name of the column in the DataFrame that represents the time period identifier 
        (fixed effect).
        Default is None (no time fixed effect).

    Returns:
    --------
    model : statsmodels.regression.linear_model.RegressionResults
        Model of the OLS regression with cluster-robust standard errors.

    Raises:
    -------
    ValueError:
        If any of the input parameters do not meet the specified criteria:
        - If 'depvar' is not a column of the data or is not provided within quotes.
        - If 'indvar' is provided but it is not provided in a list of strings(within quotes), or 
          any element in 'indvar' is not a column of the data.
        - If 'treat_var' is not a column of the data or is not provided within quotes.
        - If 'cluster_var' is not a column of the data or is not provided within quotes.
        - If 'treat_var' column contains values other than 0 or 1.
        - If 'state_variable' is provided and is not a column of the data or is not provided within 
          quotes.
        - If 'time_variable' is provided and is not a column of the data or is not provided within 
          quotes.

    Examples:
    ---------
    # Example 1: OLS regression with only a dependent variable and a treatment variable with 
                 cluster-robust standard errors
    model = _crse_regression_function(data=my_panel_data, depvar='Y', treat_var='TREATMENT', 
                 cluster_var='Cluster')

    # Example 2: OLS regression with additional independent variables, state fixed effect, 
                 time fixed effect, and cluster-robust standard errors
    model = _crse_regression_function(data=my_panel_data, depvar='Y', treat_var='TREATMENT',
                                      cluster_var='Cluster', indvar=['X1', 'X2'], 
                                      state_variable='State', time_variable='Year')
    """
    
    if not isinstance(depvar, str):
        raise ValueError("depvar must be provided as a string as a column of the data (inside quotes).")

    if indvar is not None:
        if not isinstance(indvar,list):
            raise ValueError("indvar must be provided as a list of strings representing columns of the data (inside quotes)")    
        if not all(isinstance(var, str) for var in indvar):
            raise ValueError("All elements in indvar must be provided as strings as columns of the data (inside quotes).")
        
    if indvar is not None:
        for var in indvar:
            if var not in data.columns:
                    raise ValueError(f"'{var}' is not a column in the data.")
        
    if depvar not in data.columns:
        raise ValueError(f"'{depvar}' is not a column in the data.")
    
    if not isinstance(treat_var, str):
        raise ValueError("treat_var must be provided as a string as a column of the data (inside quotes).")
    
    if treat_var is None:
        raise ValueError("treat_var must be provided as a string representing a column of the data (inside quotes).")

    if treat_var not in data.columns:
        raise ValueError(f"'{treat_var}' is not a column in the data.")

    if not set(data[treat_var].unique()).issubset({0, 1}):
        raise ValueError(f"'{treat_var}' column must contain only 0 and 1 values.")
        
    if cluster_var not in data.columns:
        raise ValueError(f"'{cluster_var}' is not a column in the data.")
    
    if not isinstance(cluster_var, str):
        raise ValueError("cluster_var must be provided as a string as a column of the data (inside quotes).")

    if state_variable is not None and state_variable not in data.columns:
        raise ValueError(f"'{state_variable}' is not a column in the data")
    
    if time_variable is not None and time_variable not in data.columns:
        raise ValueError(f"'{time_variable}' is not a column in the data")
      
    if indvar is not None:
        X = data[indvar + [treat_var]]  

    if indvar is None:
        X= data[treat_var]    
    
    X = sm.add_constant(X)
    y = data[depvar]

    if state_variable is not None:
        state_dummies = pd.get_dummies(data[state_variable], prefix='state', 
                                       drop_first=True).astype(int)
        X = pd.concat([X, state_dummies], axis=1)

    if time_variable is not None:
        time_dummies = pd.get_dummies(data[time_variable], prefix='time', 
                                      drop_first=True).astype(int)
        X = pd.concat([X, time_dummies], axis=1)

    model = sm.OLS(y, X).fit(cov_type='cluster', use_t = True, 
                             cov_kwds={'groups': data[cluster_var].astype(str)})
    

    return model



# RESIDUAL AGGREGATION REGRESSION FUNCTION

def _res_agg_regression_function(data, depvar, treatment_var, state_variable, time_variable, 
                                 indvar=None):
    """
    Perform a residual aggregation regression.

    This function performs a residual aggregation regression where the time series information is 
    compressed into a two period panel and the treatment effect is calculated.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input data containing all relevant variables.
    depvar : str
        The dependent variable to be used in the regression. This variable should represent the 
        outcome or response being studied.
    treatment_var : str
        The variable indicating treatment/control status. It should contain binary values 
        indicating treatment (1) or control (0).
    state_variable : str
        The variable representing the states/groups in the data. This variable is used to aggregate 
        residuals within each state/group.
    time_variable : str
        The variable representing time periods. It can be represented as integers or strings and is 
        used to distinguish different time points or periods.
    indvar : list of str, optional
        A list of names of additional independent variables to include in the regression. These 
        variables can represent covariates or other factors that may influence the outcome.

    Returns:
    --------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        The results of the residual aggregation regression, which include coefficients, standard 
        errors, t-values, p-values, and other statistics.

    Raises:
    -------
    ValueError
        If any of the input arguments are invalid or if required columns are not present in the data.

    Notes:
    ------
    - The residual aggregation regression works as follows:
      1. Estimation of residuals from an initial regression of the dependent variable on all 
         explanatory variables, including state and time effects, using ordinary least squares (OLS).
      2. Filtering of residuals only for states undergoing treatment.
      3. Separation of filtered residuals into two panels: pre-treatment and post-treatment.
      4. Creation of a treatment indicator 'TREATMENT' where it is 0 in the pre-treatment panel and 
         1 in the post-treatment panel.
      5. Computation of the average of each state in both panels, resulting in one observation for 
         each state in each panel.
      6. Regression of these residuals against the dummy variable 'TREATMENT' to estimate the 
         average treatment effect.

    - The function requires the dependent variable, treatment variable, state variable, and time 
      variable to be present in the data. The state and time dummies are generated for the initial 
      regression.
    - Optionally, additional independent variables can be provided to control for additional factors.
    - This function internally utilizes the statsmodels library for regression analysis.

    Example:
    --------
    # Assuming 'data' is a pandas DataFrame containing relevant columns
    result = _res_agg_regression_function(data, 'Outcome', 'Treatment', 'State', 'Time', 
                                          indvar=['Covariate1', 'Covariate2'])
    print(result.summary())

    """
 
    if not isinstance(depvar, str):
        raise ValueError("depvar must be provided as a string as a column of the data (inside quotes).")

    if indvar is not None:
        if not all(isinstance(var, str) for var in indvar):
            raise ValueError("All elements in covar must be provided as strings as columns of the data (inside quotes).")

    if depvar not in data.columns:
        raise ValueError(f"'{depvar}' is not a column in the data.")

    if indvar is not None:
        for var in indvar:
            if var not in data.columns:
                raise ValueError(f"'{var}' is not a column in the data.")

    if indvar is not None:
        if not isinstance(indvar, list):
            raise ValueError("covar must be provided as a list of strings representing columns of the data (inside quotes)")

    if indvar is not None:    
        if not all(isinstance(var, str) for var in indvar):
            raise ValueError("All elements in covar must be provided as strings as columns of the data (inside quotes).")
            
    if treatment_var is None:
        raise ValueError("treatment_var must be provided as a string representing a column of the data (inside quotes).")    

    if treatment_var not in data.columns:
        raise ValueError(f"'{treatment_var}' is not a column in the data.")

    if not all(isinstance(var, str) for var in treatment_var):
        raise ValueError("All elements in treatment_var must be provided as strings as columns of the data (inside quotes).")
    
    if not set(data[treatment_var].unique()).issubset({0, 1}):
        raise ValueError(f"'{treatment_var}' column must contain only 0 and 1 values.")

    if not isinstance(state_variable, str):
        raise ValueError("state_variable must be provided as a string as a column of the data (inside quotes).")

    if not isinstance(time_variable, str):
        raise ValueError("time_variable must be provided as a string as a column of the data (inside quotes).")

    if state_variable not in data.columns:
        raise ValueError(f"'{state_variable}' is not a column in the data.")
    
    if time_variable not in data.columns:
        raise ValueError(f"'{time_variable}' is not a column in the data.")
    
    X = pd.DataFrame()
    
    time_dummies = pd.get_dummies(data[time_variable], prefix='time', drop_first=True).astype(int)
    X = pd.concat([X, time_dummies], axis=1)

    state_dummies = pd.get_dummies(data[state_variable], prefix='state', drop_first=True).astype(int)
    X = pd.concat([X, state_dummies], axis=1)

    if indvar is not None:
        X = pd.concat(X,  data[indvar], axis=1)
     
    X = sm.add_constant(X)
    y = data[depvar]

    model = sm.OLS(y, X).fit()

    data['Residuals'] = model.resid

    states_with_treatment = data[data[treatment_var] == 1][state_variable].unique()

    filtered_data = data[data[state_variable].isin(states_with_treatment)]

    pre_treatment_df = filtered_data[filtered_data[treatment_var] == 0]
    post_treatment_df = filtered_data[filtered_data[treatment_var] == 1]

    avg_residuals_pre_treatment = pre_treatment_df.groupby(state_variable)['Residuals'].mean().reset_index()
    avg_residuals_post_treatment = post_treatment_df.groupby(state_variable)['Residuals'].mean().reset_index()

    avg_residuals_pre_treatment['TREATMENT'] = 0
    avg_residuals_post_treatment['TREATMENT'] = 1
    
    two_period_panel_df = pd.concat([avg_residuals_pre_treatment, avg_residuals_post_treatment],
                                     ignore_index=True)

    y = two_period_panel_df['Residuals']
    X = two_period_panel_df['TREATMENT']

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    return model


# WILD CLUSTER BOOTSTRAPPING FUNCTION

def _boot_regression_function(data, depvar, treat_var, cluster_var, indvar = None, 
                              state_variable=None, time_variable=None, num_boot = 100):
    """
    This function performs Wild Cluster Bootstrapping method on the treatment variable for each 
    cluster using the Rademacher Distribution.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input panel data containing dependent, independent, state, and time variables.
    depvar : str
        The name of the column in the DataFrame that represents the dependent variable.
    treat_var : str
        The name of the column in the DataFrame that represents the treatment/control indicator 
        (binary: 0 or 1).
    cluster_var : str
        The name of the column in the DataFrame that represents the cluster variable identifier. 
        Used for cluster-based bootstrapping.
    indvar : list of str, optional
        A list of names of columns in the DataFrame that represent the independent variables.
    state_variable : str, optional
        The name of the column in the DataFrame that represents the state or group identifier 
        (fixed effect). Default is None (no fixed effect).
    time_variable : str, optional
        The name of the column in the DataFrame that represents the time period identifier 
        (fixed effect). Default is None (no fixed effect).
    num_boot : int, optional
        The number of bootstrap iterations to perform. Default is 100.

    Returns:
    --------
    dict
        A dictionary containing results of the bootstrap analysis, including observed coefficient, 
        mean bootstrapped coefficient, bias, MSE, RMSE, standard error, and p-value.

    Raises:
    -------
    ValueError:
        If 'depvar' or 'treat_var' is not a column of the data or is not provided within quotes.
        If 'cluster_var' is not provided within quotes.
        If 'cluster_var' is not a column in the data.

    Notes:
    ------
    - This function first performs OLS regression on panel data using the provided dependent and 
      independent variables.
    - It then conducts cluster-based bootstrapping to estimate the treatment effect while accounting 
      for potential cluster-level heteroscedasticity.
    - The number of bootstrap iterations can be adjusted using the 'num_boot' parameter.
    - Additional control for state and time effects can be provided using 'state_variable' and 
      'time_variable' parameters, respectively.
    - The function internally utilizes the statsmodels library for regression analysis and numpy 
      for bootstrap sampling.

    Example:
    --------
    # Assuming 'data' is a pandas DataFrame containing relevant columns
    results = _boot_regression_function(data, 'Outcome', 'Treatment', 'Cluster', 
                                        indvar=['Covariate1', 'Covariate2'], state_variable='State', 
                                        time_variable='Time', num_boot=1000)
    print(results)

    """

    if not isinstance(depvar, str):
        raise ValueError("depvar must be provided as a string as a column of the data (inside quotes).")
    
    if treat_var is None:
        raise ValueError("treatment_var must be provided as a string representing a column of the data (inside quotes).")
    
    if not isinstance(treat_var, str):
        raise ValueError("treat_var must be provided as a string as a column of the data (inside quotes).")
    
    if indvar is not None:
        if not isinstance(indvar, list):
            raise ValueError("indvar must be provided as a list of strings representing column names.") 

    if indvar is not None:
        if not all(isinstance(var, str) for var in indvar):
            raise ValueError("All elements in indvar must be provided as strings as columns of the data (inside quotes).")
    
    if depvar not in data.columns:
        raise ValueError(f"'{depvar}' is not a column in the data.")
    
    if treat_var not in data.columns:
        raise ValueError(f"'{treat_var}' is not a column in the data.")
    
    if not set(data[treat_var].unique()).issubset({0, 1}):
        raise ValueError(f"'{treat_var}' column must contain only 0 and 1 values.")
    
    if not isinstance(cluster_var, str):
        raise ValueError("cluster_var must be provided as a string (inside quotes).")
    
    if cluster_var not in data.columns:
        raise ValueError(f"'{cluster_var}' is not a column in the data.")

    if indvar is not None:
        for var in indvar:
            if var not in data.columns:
                    raise ValueError(f"'{var}' is not a column in the data.")

    if indvar is None:
        indvar = []
    elif not all(isinstance(var, str) for var in indvar):
        raise ValueError("All elements in indvar must be provided as strings as columns of the data (inside quotes).")

    if len(set(indvar + [treat_var])) != len(indvar) + 1:
        raise ValueError("Duplicate variables detected in independent variables.") 
             
    X = data[indvar + [treat_var]]
    X = sm.add_constant(X)
    y = data[depvar]

    if state_variable is not None:
        state_dummies = pd.get_dummies(data[state_variable], prefix='state', 
                                       drop_first=True).astype(int)
        X = pd.concat([X, state_dummies], axis=1)

    if time_variable is not None:
        time_dummies = pd.get_dummies(data[time_variable], prefix='time', 
                                      drop_first=True).astype(int)
        X = pd.concat([X, time_dummies], axis=1)

    model = sm.OLS(y, X).fit()
    initial_residuals = model.resid
    pred = model.predict(X)

    data['residuals'] = initial_residuals

    observed_coef = model.params[treat_var]

    bootstrap_coefs = []

    for _ in range(num_boot):
    
        rademacher_for_clusters = data[cluster_var].unique().tolist()
        rademacher_dict = {cluster: np.random.choice([-1, 1]) for cluster in rademacher_for_clusters}
        boot_residuals = data.apply(lambda row: rademacher_dict[row[cluster_var]] * 
                                    row['residuals'], axis=1)

        y_boot = pred + boot_residuals  
        model_boot = sm.OLS(y_boot, X ).fit()

        bootstrap_coefs.append(model_boot.params[treat_var])


    mean_coef = np.mean(bootstrap_coefs)
    bias = mean_coef - observed_coef
    mse = np.mean((bootstrap_coefs - observed_coef) ** 2)
    rmse = np.sqrt(mse)
    standard_error = np.std(bootstrap_coefs)
    p_value = (np.sum(bootstrap_coefs >= observed_coef) if observed_coef > 0
               else np.sum(bootstrap_coefs <= observed_coef)) / num_boot

    analysis_results = {
        'Observed Coefficient': observed_coef,
        'Mean Bootstrapped Coefficient': mean_coef,
        'Bias': bias,
        'MSE': mse,
        'RMSE': rmse,
        'Standard Error': standard_error,
        'P-Value': p_value
    }

    return analysis_results


# FGLS REGRESSION FUNCTION

def _fgls_regression_function(data, depvar , treat_var, indvar = None, dummy_indvar= None, 
                              state_variable=None, time_variable=None):
    """
    Perform Feasible Generalised Least Squares (FGLS) regression on panel data.

    This function performs Feasible Generalised Least Squares (FGLS) regression on panel data, 
    allowing for autoregressive errors and controlling for state and time effects.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input panel data containing dependent, independent, state, and time variables.
    depvar : str
        The name of the column in the DataFrame that represents the dependent variable.
    treat_var : str
        The name of the column in the DataFrame that represents the treatment/control indicator 
        (binary: 0 or 1).
    indvar : list of str, optional
        A list of names of columns in the DataFrame that represent the independent variables.
    dummy_indvar : list of str, optional
        A list of names of columns in the DataFrame that represent dummy variables. These variables 
        are treated separately from regular independent variables.
    state_variable : str, optional
        The name of the column in the DataFrame that represents the state or group identifier 
        (fixed effect). Default is None (no state fixed effect).
    time_variable : str, optional
        The name of the column in the DataFrame that represents the time period identifier 
        (fixed effect). Default is None (no time fixed effect).

    Returns:
    --------
    model : statsmodels.regression.linear_model.OLSResults
        Model of the FGLS regression.

    Raises:
    -------
    ValueError:
        If 'depvar' or 'treat_var' is not a column of the data or is not provided within quotes.
        If 'indvar' or 'dummy_indvar' is provided but any element is not a column of the data or 
        is not provided within quotes.

    Notes:
    ------
    - The first step of FGLS is performing an OLS regression of the dependent variable on all the 
      explanatory variables to obtain the residuals.
    - These residuals are regressed based on up to 2 lags(for each state if state_variable is 
      provided) to obtain the autocorrelation coefficients of the residuals. These auto correlation 
      coefficients are denoted by rho_1 and rho_2.
    - The next step involves a linear GLS transformation on the dependent variable and the 
      continuous independent variables(if provided).
    - Finally, the transformed dependent variables are regressed on the treatment variable and the 
      transformed independent variables(if provided). 
    - Dummy variables are treated separately from regular independent variables and are included in 
      the regression if provided.
    - Control for state and time effects can be provided using 'state_variable' and 'time_variable' 
      parameters, respectively.
    - The function internally utilizes the statsmodels library for regression analysis.

    Example:
    --------
    # Assuming 'data' is a pandas DataFrame containing relevant columns
    model_fgls = _fgls_regression_function(data, 'Outcome', 'Treatment', indvar=['Independent1', 
                                           'Independent2'], dummy_indvar=['Dummy1', 'Dummy2'], 
                                            state_variable='State', time_variable='Time')
    print(model_fgls.summary())

    """
    
    if not isinstance(depvar, str):
        raise ValueError("depvar must be provided as a string as a column of the data (inside quotes).")

    if depvar not in data.columns:
        raise ValueError(f"'{depvar}' is not a column in the data.")
    
    if treat_var is None:
        raise ValueError("treatment_var must be provided as a string representing a column of the data (inside quotes).")

    if not isinstance(treat_var, str):
        raise ValueError("treat_var must be provided as a string as a column of the data (inside quotes).")

    if treat_var not in data.columns:
        raise ValueError(f"'{treat_var}' is not a column in the data.")
    
    if not set(data[treat_var].unique()).issubset({0, 1}):
        raise ValueError(f"'{treat_var}' column must contain only 0 and 1 values.")
    
    if indvar is not None:
        if not isinstance(indvar, list):
            raise ValueError(f"'{indvar}' must be provided as a list.")
        for var in indvar:
            if not isinstance(var, str):
                raise ValueError("All elements in indvar must be strings (inside quotes).")
            if var not in data.columns:
                raise ValueError(f"'{var}' is not a column in the data.")
                
    if dummy_indvar is not None:
        if not isinstance(dummy_indvar, list):
            raise ValueError(f"'{dummy_indvar}' must be provided as a list.")
        for var in dummy_indvar:
            if not isinstance(var, str):
                raise ValueError("All elements in dummy_indvar must be strings (inside quotes).")
            if var not in data.columns:
                raise ValueError(f"'{var}' is not a column in the data.")
                                
    if state_variable is not None and not isinstance(state_variable, str):
        raise ValueError(f"'{state_variable}' must be provided as a string (inside quotes).")
    
    if time_variable is not None and not isinstance(time_variable, str):
        raise ValueError(f"'{time_variable}' must be provided as a string (inside quotes).")

    if dummy_indvar is not None and  indvar is not None:
        X = data[indvar + dummy_indvar + [treat_var]]

    if indvar is None and dummy_indvar is not None :
        X = data[dummy_indvar + [treat_var]]

    if dummy_indvar is None and indvar is not None :
        X = data[indvar + [treat_var]]

    if dummy_indvar is None and indvar is None:
        X = data[treat_var] 

    X = sm.add_constant(X)
    y = data[depvar]

    if state_variable is not None:
        state_dummies = pd.get_dummies(data[state_variable], prefix='state', 
                                       drop_first=True).astype(int)
        X = pd.concat([X, state_dummies], axis=1)

    if time_variable is not None:
        time_dummies = pd.get_dummies(data[time_variable], prefix='time', 
                                      drop_first=True).astype(int)
        X = pd.concat([X, time_dummies], axis=1)

    model = sm.OLS(y, X).fit()

    df = data.copy()
    df2 = data.copy()

    df['resid'] = model.resid

    if state_variable is not None :

        df['lag_1'] = df.groupby(state_variable)['resid'].shift(1)
        df['lag_2'] = df.groupby(state_variable)['resid'].shift(2)
        df = df.dropna()

        model_ar = sm.OLS(df['resid'], sm.add_constant(df[['lag_1', 'lag_2']])).fit()
        rho_1 = model_ar.params['lag_1']
        rho_2 = model_ar.params['lag_2']

        df2['Y_lag_1'] = df2.groupby(state_variable)[depvar].shift(1)
        df2['Y_lag_2'] = df2.groupby(state_variable)[depvar].shift(2)

        df2['y_transformed'] = df2[depvar] - rho_1 * df2['Y_lag_1'] - rho_2 * df2['Y_lag_2']

        if indvar is not None:
            for col in indvar:
                df2[f'{col}_lag_1'] = df2.groupby(state_variable)[col].shift(1)
                df2[f'{col}_lag_2'] = df2.groupby(state_variable)[col].shift(2)

                df2[col] = df2[col] - model_ar.params['lag_1'] * df2[f'{col}_lag_1'] - model_ar.params['lag_2'] * df2[f'{col}_lag_2']
        df2 = df2.dropna()

        if dummy_indvar is not None and indvar is not None:
            X_transformed = df2[indvar + dummy_indvar + [treat_var]]

        if indvar is None and dummy_indvar is not None :
            X_transformed = df2[dummy_indvar + [treat_var]]

        if dummy_indvar is None and indvar is not None :
            X_transformed = df2[indvar + [treat_var]]

        if dummy_indvar is None and indvar is None:
            X_transformed = df2[treat_var] 

        X_transformed = sm.add_constant(X_transformed)
        y_transformed = df2['y_transformed']

        model_fgls = sm.OLS(y_transformed, X_transformed).fit()

    if state_variable is None :

        df['lag_1'] = df['resid'].shift(1)
        df['lag_2'] = df['resid'].shift(2)
        df = df.dropna()

        model_ar = sm.OLS(df['resid'], sm.add_constant(df[['lag_1', 'lag_2']])).fit()
        rho_1 = model_ar.params['lag_1']
        rho_2 = model_ar.params['lag_2']

        df2['Y_lag_1'] = df2[depvar].shift(1)
        df2['Y_lag_2'] = df2[depvar].shift(2)

        df2['y_transformed'] = df2[depvar] - rho_1 * df2['Y_lag_1'] - rho_2 * df2['Y_lag_2']

        for col in indvar:
            df2[f'{col}_lag_1'] = df2[col].shift(1)
            df2[f'{col}_lag_2'] = df2[col].shift(2)

        for col in indvar:
            df2[col] = df2[col] - rho_1 * df2[f'{col}_lag_1'] - rho_2 * df2[f'{col}_lag_2']

        df2 = df2.dropna()

        if dummy_indvar is not None and indvar is not None:
            X_transformed = df2[indvar + dummy_indvar + [treat_var]]

        if indvar is None and dummy_indvar is not None :
            X_transformed = df2[dummy_indvar + [treat_var]]

        if dummy_indvar is None and indvar is not None :
            X_transformed = df2[indvar + [treat_var]]

        if dummy_indvar is None and indvar is None:
            X_transformed = df2[treat_var] 

        X_transformed = sm.add_constant(X_transformed)
        y_transformed = df2['y_transformed']

        model_fgls = sm.OLS(y_transformed, X_transformed).fit()

    return model_fgls








    
