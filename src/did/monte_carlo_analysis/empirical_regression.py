from did.monte_carlo_analysis.subfunctions import _ols_regression_function,  _crse_regression_function
import pandas as pd

def fit_empirical_ols_model(data, data_info) :

    """
    Fit an Ordinary Least Squares (OLS) regression model to empirical data.

    This function fits an OLS regression model to empirical data using the provided information.
    The regression model is specified with the dependent variable, treatment variable, state variable, 
    and time variable.
    It returns the fitted OLS regression model.

    Parameters:
    -----------
    data : pandas.DataFrame
        The empirical data containing the relevant variables for regression analysis.

    data_info : dict
        A dictionary containing the following keys:
        - 'empirical_depvar' : str
            Name of the dependent variable in the empirical data.
        - 'treatment_variable' : str
            Name of the treatment variable in the empirical data.
        - 'empirical_state_variable' : str
            Name of the state variable in the empirical data.
        - 'empirical_time_variable' : str
            Name of the time variable in the empirical data.

    Returns:
    --------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted OLS regression model.

    Notes:
    ------
    - This function uses the '_ols_regression_function' to fit the OLS model to the empirical data.

    Example:
    --------
    To fit an OLS regression model to empirical data with the dependent variable 'Y', treatment 
    variable 'TREATMENT', state variable 'STATE', and time variable 'YEAR', you can call the function 
    as follows:

    >>> data_info = {'empirical_depvar': 'Y', 'treatment_variable': 'TREATMENT', 
                      'empirical_state_variable': 'STATE', 'empirical_time_variable': 'YEAR'}

    >>> model = fit_empirical_ols_model(data, data_info)

    The 'model' object will contain the fitted OLS regression model.

    """

    depvar = data_info['empirical_depvar']
    treat_var = data_info['treatment_variable']
    state_variable = data_info["empirical_state_variable"]
    time_variable = data_info["empirical_time_variable"]

    model = _ols_regression_function(data, depvar = depvar, treat_var= treat_var , 
                                      state_variable= state_variable , time_variable= time_variable)

    return model


def fit_empirical_crse_model(data, data_info) :

    """
    Fit a Cluster Robust Standard Error (CRSE) regression model to empirical data.

    This function fits a CRSE regression model to empirical data using the provided information.
    The regression model is specified with the dependent variable, treatment variable, 
    cluster variable, state variable, and time variable.
    It returns the fitted CRSE regression model.

    Parameters:
    -----------
    data : pandas.DataFrame
        The empirical data containing the relevant variables for regression analysis.

    data_info : dict
        A dictionary containing the following keys:
        - 'empirical_depvar' : str
            Name of the dependent variable in the empirical data.
        - 'treatment_variable' : str
            Name of the treatment variable in the empirical data.
        - 'empirical_clustervar' : str
            Name of the cluster variable in the empirical data.
        - 'empirical_state_variable' : str
            Name of the state variable in the empirical data.
        - 'empirical_time_variable' : str
            Name of the time variable in the empirical data.

    Returns:
    --------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted OLS regression model.

    Notes:
    ------
    - This function uses the '_crse_regression_function' to fit the OLS model to the empirical data.

    Example:
    --------
    To fit a CRSE regression model to empirical data with the dependent variable 'Y', treatment 
    variable 'TREATMENT', cluster variable 'STATE', state variable 'STATE', and time variable 'YEAR', 
    you can call the function as follows:

    >>> data_info = {'empirical_depvar': 'Y', 'treatment_variable': 'TREATMENT', 
                     empirical_clustervar: 'STATE', 'empirical_state_variable': 'STATE', 
                     'empirical_time_variable': 'YEAR'}

    >>> model = fit_empirical_ols_model(data, data_info)

    The 'model' object will contain the fitted CRSE regression model.

    """

    depvar = data_info['empirical_depvar']
    treat_var = data_info['treatment_variable']
    cluster_var = data_info['empirical_clustervar']
    state_variable = data_info["empirical_state_variable"]
    time_variable = data_info["empirical_time_variable"]
    
    model = _crse_regression_function(data, depvar = depvar, treat_var= treat_var , 
                                       cluster_var= cluster_var, state_variable= state_variable, 
                                        time_variable= time_variable)
    
    summary_table = model.summary()

    column_names = summary_table.tables[1].data[0]

    filtered_rows = []

    for row in summary_table.tables[1].data:
        if row[0] in ['const', 'TREATMENT']:
            filtered_rows.append(row)

    df = pd.DataFrame(filtered_rows, columns=column_names)
    
    return df
    
