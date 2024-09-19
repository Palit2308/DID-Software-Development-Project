"""Functions plotting results."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from did.monte_carlo_analysis.subfunctions import _random_placebo_treatment, _power_function, _ols_regression_function, _crse_regression_function, _res_agg_regression_function, _boot_regression_function, _fgls_regression_function, _calculate_statistics
from did.monte_carlo_analysis.synthetic_data_gen import _generate_homogenous_ar1_data, _generate_homogenous_ma1_data, _generate_heterogenous_ar1_data


# PLOTS FOR TYPE 1 ERROR AND POWER FOR HOMOGENOUS AR 1

def plot_homogenous_AR1(data_info):
    """
    Plot the convergence of Type 1 Errors and Power for different regression methods.

    This function generates a series of simulations to assess the convergence of Type 1 Errors and 
    Power for different
    regression methods under homogenous AR(1) processes. It performs simulations based on the 
    provided data_info dictionary,
    which contains parameters such as the number of individuals, number of time periods, 
    autocorrelation coefficient (rho),
    significance level (alpha), number of simulations, and variables related to treatment and 
    clustering.

    Parameters:
    -----------
    data_info : dict
        A dictionary containing information necessary for the simulations, including:
            - 'N': Number of individuals.
            - 'T': Number of time periods.
            - 'rho': Autocorrelation coefficient.
            - 'alpha': Significance level.
            - 'num_individuals': Number of individuals for treatment assignment.
            - 'num_simulations': Number of simulations to run.
            - 'cluster_variable': Name of the variable to cluster on.
            - 'treatment_variable': Name of the treatment variable.

    Returns:
    --------
    plotly.graph_objs._figure.Figure, plotly.graph_objs._figure.Figure
        Two Plotly figures representing the convergence of Type 1 Errors and Power for each 
        regression method.

    Raises:
    -------
    None. Since this function is created using the subfunctions, if the dictionary inputs do not 
    comply wiht the input requirements 
    of the subfunctions, this function raises the Value Errors.

    Example:
    --------
    To visualize the convergence of Type 1 Errors and Power for different regression methods 
    based on a set of
    simulation parameters provided in the 'data_info' dictionary, you can call the function as 
    follows:

    >>> type1_errors_figure, power_figure = plot_homogenous_AR1(data_info)

    The 'type1_errors_figure' and 'power_figure' variables will contain Plotly figures showing 
    the convergence of Type 1 Errors and Power, respectively, for each regression method.

    Notes:
    ------
    - This function relies on internal functions for generating data, performing treatment assignment, 
      and running regression models. Ensure that these functions are defined and accessible within the 
      same environment.
    - The 'data_info' dictionary should contain all necessary parameters for running the simulations.
    """    
    np.random.seed(42)

    N = data_info["N"]
    T = data_info["T"]
    rho = data_info["rho"]
    alpha = data_info["alpha"]
    num_individuals = data_info["num_individuals"]
    num_simulations = data_info["num_simulations"]
    cluster_variable = data_info["cluster_variable"]
    treatment_variable = data_info["treatment_variable"]

    reject_count_1_ols = 0
    reject_count_1_crse = 0
    reject_count_1_res_agg = 0
    reject_count_1_wbtest = 0
    reject_count_1_fgls = 0
    reject_count_2_ols = 0
    reject_count_2_crse = 0
    reject_count_2_res_agg = 0
    reject_count_2_wbtest = 0
    reject_count_2_fgls = 0
    type1_ols_list = []
    type1_crse_list = []
    type1_res_agg_list = []
    type1_wbtest_list = []
    type1_fgls_list = []
    power_ols_list = []
    power_crse_list = []
    power_res_agg_list = []
    power_wbtest_list = []
    power_fgls_list = []

    for i in range(num_simulations):

        data = _generate_homogenous_ar1_data(N, T, rho, num_individuals)

        treat_data = _random_placebo_treatment(data, 'state', 'time')

        final_data = _power_function(treat_data, depvar= 'value', treatment_var= treatment_variable ,
                                      effect = 25)

        model_1_ols = _ols_regression_function(final_data, depvar='value', treat_var= 
                                                treatment_variable , state_variable='state', 
                                                 time_variable='time')
        model_2_ols = _ols_regression_function(final_data, depvar='OUTCOME', treat_var= 
                                                treatment_variable , state_variable='state', 
                                                 time_variable='time')

        model_1_crse = _crse_regression_function(final_data , depvar= 'value', treat_var=
                                                  treatment_variable , cluster_var= cluster_variable,
                                                   state_variable='state', time_variable='time')
        model_2_crse = _crse_regression_function(final_data , depvar= 'OUTCOME', treat_var= 
                                                  treatment_variable , cluster_var= cluster_variable,
                                                   state_variable='state', time_variable='time')

        model_1_res_agg = _res_agg_regression_function(final_data, depvar='value', treatment_var= 
                                                        treatment_variable , state_variable='state',
                                                         time_variable='time')
        model_2_res_agg = _res_agg_regression_function(final_data, depvar='OUTCOME', treatment_var=
                                                        treatment_variable , state_variable='state',
                                                         time_variable='time')

        model_1_wbtest = _boot_regression_function(final_data , depvar= 'value', treat_var=
                                                    treatment_variable, cluster_var= cluster_variable,
                                                     state_variable='state', time_variable='time')
        model_2_wbtest = _boot_regression_function(final_data , depvar= 'OUTCOME', treat_var= 
                                                    treatment_variable , cluster_var= cluster_variable,
                                                      state_variable='state', time_variable='time')

        model_1_fgls = _fgls_regression_function(final_data, depvar='value', treat_var= 
                                                  treatment_variable , state_variable='state', 
                                                   time_variable='time')
        model_2_fgls = _fgls_regression_function(final_data, depvar='OUTCOME', treat_var= 
                                                  treatment_variable , state_variable='state', 
                                                   time_variable='time')

        if model_1_ols.pvalues['TREATMENT'] < alpha:
            reject_count_1_ols += 1
        type1_ols = reject_count_1_ols/ (i + 1)
        type1_ols_list.append(type1_ols)

        if model_2_ols.pvalues['TREATMENT'] < alpha:
            reject_count_2_ols += 1
        power_ols = reject_count_2_ols/ (i + 1)
        power_ols_list.append(power_ols)

        if model_1_crse.pvalues['TREATMENT'] < alpha:
            reject_count_1_crse += 1
        type1_crse = reject_count_1_crse/ (i + 1)
        type1_crse_list.append(type1_crse)

        if model_2_crse.pvalues['TREATMENT'] < alpha:
            reject_count_2_crse += 1
        power_crse = reject_count_2_crse/ (i + 1)
        power_crse_list.append(power_crse)

        if model_1_res_agg.pvalues['TREATMENT'] < alpha:
            reject_count_1_res_agg += 1
        type1_res_agg = reject_count_1_res_agg/ (i + 1)
        type1_res_agg_list.append(type1_res_agg)

        if model_2_res_agg.pvalues['TREATMENT'] < alpha:
            reject_count_2_res_agg += 1
        power_res_agg = reject_count_1_res_agg/ (i + 1)
        power_res_agg_list.append(power_res_agg)

        if model_1_wbtest['P-Value'] < alpha:
            reject_count_1_wbtest += 1
        type1_wbtest = reject_count_1_wbtest/ (i + 1)
        type1_wbtest_list.append(type1_wbtest)

        if model_2_wbtest['P-Value'] < alpha:
            reject_count_2_wbtest += 1
        power_wbtest = reject_count_2_wbtest/ (i + 1)
        power_wbtest_list.append(power_wbtest)

        if model_1_fgls.pvalues['TREATMENT'] < alpha:
            reject_count_1_fgls += 1
        type1_fgls = reject_count_1_fgls/ (i + 1)
        type1_fgls_list.append(type1_fgls)

        if model_2_fgls.pvalues['TREATMENT'] < alpha:
            reject_count_2_fgls += 1
        power_fgls = reject_count_2_fgls/ (i + 1)
        power_fgls_list.append(power_fgls)

    df1 = pd.DataFrame({'Simulation': np.arange(1, num_simulations + 1),
                            'Type_1_OLS': type1_ols_list,
                            'Type_1_CRSE': type1_crse_list,
                            'Type_1_Res_Agg': type1_res_agg_list,
                            'Type_1_WBTest': type1_wbtest,
                            'Type_1_Fgls': type1_fgls_list}) 
    fig1_px = px.line(df1, x='Simulation', y=['Type_1_OLS', 'Type_1_CRSE','Type_1_Res_Agg', 
                                              'Type_1_WBTest', 'Type_1_Fgls' ], 
                                               labels={'x': 'Simulation', 'y': 'Type 1 Errors'})
    fig1_px.update_layout(title='Type 1 Error Convergence for Each Method',
                         title_x=0.5,
                        shapes=[dict(type="line", x0=0, y0=alpha, x1=num_simulations, y1=alpha, 
                                      line=dict(color="black", width=1, dash="dot"))])

    df2 = pd.DataFrame({'Simulation': np.arange(1, num_simulations + 1),
                            'Power_OLS': power_ols_list,
                            'Power_CRSE': power_crse_list,
                            'Power_Res_Agg': power_res_agg_list,
                            'Power_WBTest': power_wbtest,
                            'Power_Fgls': power_fgls_list}) 
    fig2_px = px.line(df2, x='Simulation', y=['Power_OLS', 'Power_CRSE','Power_Res_Agg', 
                                              'Power_WBTest', 'Power_Fgls' ], 
                                               labels={'x': 'Simulation', 'y': 'Power'})
    fig2_px.update_layout(title='Power Convergence for Each Method',
                         title_x=0.5,
                        shapes=[dict(type="line", x0=0, y0=alpha, x1=num_simulations, 
                                      y1=alpha, line=dict(color="black", width=1, dash="dot"))])
     
    return fig1_px, fig2_px



# PLOTS FOR TYPE 1 ERROR AND POWER FOR HETEROGENOUS AR 1

def plot_heterogenous_AR1(data_info):
    """
    Plot the convergence of Type 1 Errors and Power for different regression methods under 
    heterogenous AR(1) processes.

    This function generates a series of simulations to assess the convergence of Type 1 Errors and 
    Power for different
    regression methods under heterogenous AR(1) processes. It performs simulations based on the 
    provided data_info dictionary,
    which contains parameters such as the number of individuals, number of time periods, 
    significance level (alpha),
    number of simulations, and variables related to treatment and clustering.

    Parameters:
    -----------
    data_info : dict
        A dictionary containing information necessary for the simulations, including:
            - 'N': Number of individuals.
            - 'T': Number of time periods.
            - 'alpha': Significance level.
            - 'num_individuals': Number of individuals for treatment assignment.
            - 'num_simulations': Number of simulations to run.
            - 'cluster_variable': Name of the variable to cluster on.
            - 'treatment_variable': Name of the treatment variable.

    Returns:
    --------
    plotly.graph_objs._figure.Figure, plotly.graph_objs._figure.Figure
        Two Plotly figures representing the convergence of Type 1 Errors and Power for each 
        regression method.

    Raises:
    -------
    None. Since this function is created using the subfunctions, if the dictionary inputs do not 
    comply wiht the input requirements of the subfunctions, this function raises the Value Errors.

    Example:
    --------
    To visualize the convergence of Type 1 Errors and Power for different regression methods 
    based on a set of simulation parameters provided in the 'data_info' dictionary, 
    you can call the function as follows:

    >>> type1_errors_figure, power_figure = plot_heterogenous_AR1(data_info)

    The 'type1_errors_figure' and 'power_figure' variables will contain Plotly figures showing 
    the convergence of Type 1 Errors and Power, respectively, for each regression method under 
    heterogenous AR(1) processes.

    Notes:
    ------
    - This function relies on internal functions for generating data, performing treatment 
      assignment, and running regression models. Ensure that these functions are defined and 
      accessible within the same environment.
    - The 'data_info' dictionary should contain all necessary parameters for running the simulations.
    """
    np.random.seed(42)

    N = data_info["N"]
    T = data_info["T"]
    alpha = data_info["alpha"]
    num_individuals = data_info["num_individuals"]
    num_simulations = data_info["num_simulations"]
    cluster_variable = data_info["cluster_variable"]
    treatment_variable = data_info["treatment_variable"]

    reject_count_1_ols = 0
    reject_count_1_crse = 0
    reject_count_1_res_agg = 0
    reject_count_1_wbtest = 0
    reject_count_1_fgls = 0
    reject_count_2_ols = 0
    reject_count_2_crse = 0
    reject_count_2_res_agg = 0
    reject_count_2_wbtest = 0
    reject_count_2_fgls = 0
    type1_ols_list = []
    type1_crse_list = []
    type1_res_agg_list = []
    type1_wbtest_list = []
    type1_fgls_list = []
    power_ols_list = []
    power_crse_list = []
    power_res_agg_list = []
    power_wbtest_list = []
    power_fgls_list = []    

    for i in range(num_simulations):

        data = _generate_heterogenous_ar1_data(N, T, num_individuals)

        treat_data = _random_placebo_treatment(data, 'state', 'time')

        final_data = _power_function(treat_data, depvar= 'value', treatment_var= treatment_variable, 
                                      effect = 25)

        model_1_ols = _ols_regression_function(final_data, depvar='value', treat_var= treatment_variable, 
                                                state_variable='state', time_variable='time')
        model_2_ols = _ols_regression_function(final_data, depvar='OUTCOME', treat_var= treatment_variable, 
                                                state_variable='state', time_variable='time')

        model_1_crse = _crse_regression_function(final_data , depvar= 'value', treat_var= treatment_variable, 
                                                  cluster_var= cluster_variable,  state_variable='state', 
                                                   time_variable='time')
        model_2_crse = _crse_regression_function(final_data , depvar= 'OUTCOME', treat_var= treatment_variable , 
                                                  cluster_var= cluster_variable,  state_variable='state', 
                                                   time_variable='time')

        model_1_res_agg = _res_agg_regression_function(final_data, depvar='value', treatment_var= 
                                                        treatment_variable , state_variable='state', 
                                                         time_variable='time')
        model_2_res_agg = _res_agg_regression_function(final_data, depvar='OUTCOME', treatment_var= 
                                                        treatment_variable , state_variable='state', 
                                                         time_variable='time')

        model_1_wbtest = _boot_regression_function(final_data , depvar= 'value', treat_var=  
                                                    treatment_variable , cluster_var= cluster_variable,  
                                                     state_variable='state', time_variable='time')
        model_2_wbtest = _boot_regression_function(final_data , depvar= 'OUTCOME', treat_var=  
                                                    treatment_variable , cluster_var= cluster_variable,  
                                                     state_variable='state', time_variable='time')

        model_1_fgls = _fgls_regression_function(final_data, depvar='value', treat_var= 
                                                  treatment_variable , state_variable='state', 
                                                   time_variable='time')
        model_2_fgls = _fgls_regression_function(final_data, depvar='OUTCOME', treat_var= 
                                                  treatment_variable , state_variable='state', 
                                                   time_variable='time')

        if model_1_ols.pvalues['TREATMENT'] < alpha:
            reject_count_1_ols += 1
        type1_ols = reject_count_1_ols/ (i + 1)
        type1_ols_list.append(type1_ols)

        if model_2_ols.pvalues['TREATMENT'] < alpha:
            reject_count_2_ols += 1
        power_ols = reject_count_2_ols/ (i + 1)
        power_ols_list.append(power_ols)

        if model_1_crse.pvalues['TREATMENT'] < alpha:
            reject_count_1_crse += 1
        type1_crse = reject_count_1_crse/ (i + 1)
        type1_crse_list.append(type1_crse)

        if model_2_crse.pvalues['TREATMENT'] < alpha:
            reject_count_2_crse += 1
        power_crse = reject_count_2_crse/ (i + 1)
        power_crse_list.append(power_crse)

        if model_1_res_agg.pvalues['TREATMENT'] < alpha:
            reject_count_1_res_agg += 1
        type1_res_agg = reject_count_1_res_agg/ (i + 1)
        type1_res_agg_list.append(type1_res_agg)

        if model_2_res_agg.pvalues['TREATMENT'] < alpha:
            reject_count_2_res_agg += 1
        power_res_agg = reject_count_1_res_agg/ (i + 1)
        power_res_agg_list.append(power_res_agg)

        if model_1_wbtest['P-Value'] < alpha:
            reject_count_1_wbtest += 1
        type1_wbtest = reject_count_1_wbtest/ (i + 1)
        type1_wbtest_list.append(type1_wbtest)

        if model_2_wbtest['P-Value'] < alpha:
            reject_count_2_wbtest += 1
        power_wbtest = reject_count_2_wbtest/ (i + 1)
        power_wbtest_list.append(power_wbtest)

        if model_1_fgls.pvalues['TREATMENT'] < alpha:
            reject_count_1_fgls += 1
        type1_fgls = reject_count_1_fgls/ (i + 1)
        type1_fgls_list.append(type1_fgls)

        if model_2_fgls.pvalues['TREATMENT'] < alpha:
            reject_count_2_fgls += 1
        power_fgls = reject_count_2_fgls/ (i + 1)
        power_fgls_list.append(power_fgls)

    df1 = pd.DataFrame({'Simulation': np.arange(1, num_simulations + 1),
                            'Type_1_OLS': type1_ols_list,
                            'Type_1_CRSE': type1_crse_list,
                            'Type_1_Res_Agg': type1_res_agg_list,
                            'Type_1_WBTest': type1_wbtest,
                            'Type_1_Fgls': type1_fgls_list}) 
    fig1_px = px.line(df1, x='Simulation', y=['Type_1_OLS', 'Type_1_CRSE','Type_1_Res_Agg', 
                                              'Type_1_WBTest', 'Type_1_Fgls' ], 
                                               labels={'x': 'Simulation', 'y': 'Type 1 Errors'})
    fig1_px.update_layout(title='Type 1 Error Convergence for Each Method',
                         title_x=0.5,
                        shapes=[dict(type="line", x0=0, y0=alpha, x1=num_simulations, y1=alpha, 
                                     line=dict(color="black", width=1, dash="dot"))])

    df2 = pd.DataFrame({'Simulation': np.arange(1, num_simulations + 1),
                            'Power_OLS': power_ols_list,
                            'Power_CRSE': power_crse_list,
                            'Power_Res_Agg': power_res_agg_list,
                            'Power_WBTest': power_wbtest,
                            'Power_Fgls': power_fgls_list}) 
    fig2_px = px.line(df2, x='Simulation', y=['Power_OLS', 'Power_CRSE','Power_Res_Agg', 
                                              'Power_WBTest', 'Power_Fgls' ], 
                                               labels={'x': 'Simulation', 'y': 'Power'})
    fig2_px.update_layout(title='Power Convergence for Each Method',
                         title_x=0.5,
                        shapes=[dict(type="line", x0=0, y0=alpha, x1=num_simulations, 
                                      y1=alpha, line=dict(color="black", width=1, dash="dot"))])
     
    return fig1_px, fig2_px


# PLOTS FOR TYPE 1 ERROR AND POWER FOR HOMOGENOUS MA 1

def plot_homogenous_MA1(data_info):
    """
    Plot the convergence of Type 1 Errors and Power for different regression methods under 
    homogenous MA(1) processes.

    This function generates a series of simulations to assess the convergence of Type 1 Errors and 
    Power for different regression methods under homogenous MA(1) processes. It performs simulations 
    based on the provided data_info dictionary, which contains parameters such as the number of 
    individuals, number of time periods, MA(1) parameter (theta), significance level (alpha), 
    number of simulations, and variables related to treatment and clustering.

    Parameters:
    -----------
    data_info : dict
        A dictionary containing information necessary for the simulations, including:
            - 'N': Number of individuals.
            - 'T': Number of time periods.
            - 'theta': MA(1) parameter.
            - 'alpha': Significance level.
            - 'num_individuals': Number of individuals for treatment assignment.
            - 'num_simulations': Number of simulations to run.
            - 'cluster_variable': Name of the variable to cluster on.
            - 'treatment_variable': Name of the treatment variable.

    Returns:
    --------
    plotly.graph_objs._figure.Figure, plotly.graph_objs._figure.Figure
        Two Plotly figures representing the convergence of Type 1 Errors and Power for each 
        regression method.

    Raises:
    -------
    None. Since this function is created using the subfunctions, if the dictionary inputs do not 
    comply wiht the input requirements of the subfunctions, this function raises the Value Errors.

    Example:
    --------
    To visualize the convergence of Type 1 Errors and Power for different regression methods based 
    on a set of simulation parameters provided in the 'data_info' dictionary, you can call the 
    function as follows:

    >>> type1_errors_figure, power_figure = plot_homogenous_MA1(data_info)

    The 'type1_errors_figure' and 'power_figure' variables will contain Plotly figures showing the 
    convergence of Type 1 Errors and Power, respectively, for each regression method under homogenous
    MA(1) processes.

    Notes:
    ------
    - This function relies on internal functions for generating data, performing treatment assignment, 
      and running regression models. Ensure that these functions are defined and accessible within 
      the same environment.
    - The 'data_info' dictionary should contain all necessary parameters for running the simulations.
    """
    np.random.seed(42)

    N = data_info["N"]
    T = data_info["T"]
    theta = data_info["theta"]
    alpha = data_info["alpha"]
    num_individuals = data_info["num_individuals"]
    num_simulations = data_info["num_simulations"]
    cluster_variable = data_info["cluster_variable"]
    treatment_variable = data_info["treatment_variable"]

    reject_count_1_ols = 0
    reject_count_1_crse = 0
    reject_count_1_res_agg = 0
    reject_count_1_wbtest = 0
    reject_count_1_fgls = 0
    reject_count_2_ols = 0
    reject_count_2_crse = 0
    reject_count_2_res_agg = 0
    reject_count_2_wbtest = 0
    reject_count_2_fgls = 0
    type1_ols_list = []
    type1_crse_list = []
    type1_res_agg_list = []
    type1_wbtest_list = []
    type1_fgls_list = []
    power_ols_list = []
    power_crse_list = []
    power_res_agg_list = []
    power_wbtest_list = []
    power_fgls_list = []

    for i in range(num_simulations):

        data = _generate_homogenous_ma1_data(N, T, theta, num_individuals)

        treat_data = _random_placebo_treatment(data, 'state', 'time')

        final_data = _power_function(treat_data, depvar= 'value', treatment_var= treatment_variable , 
                                      effect = 25)

        model_1_ols = _ols_regression_function(final_data, depvar='value', treat_var= 
                                                treatment_variable , state_variable='state', 
                                                 time_variable='time')
        model_2_ols = _ols_regression_function(final_data, depvar='OUTCOME', treat_var= 
                                                treatment_variable , state_variable='state', 
                                                 time_variable='time')

        model_1_crse = _crse_regression_function(final_data , depvar= 'value', treat_var= 
                                                  treatment_variable , cluster_var= cluster_variable,  
                                                   state_variable='state', time_variable='time')
        model_2_crse = _crse_regression_function(final_data , depvar= 'OUTCOME', treat_var= 
                                                  treatment_variable , cluster_var= cluster_variable,  
                                                   state_variable='state', time_variable='time')

        model_1_res_agg = _res_agg_regression_function(final_data, depvar='value', treatment_var= 
                                                        treatment_variable , state_variable='state', 
                                                         time_variable='time')
        model_2_res_agg = _res_agg_regression_function(final_data, depvar='OUTCOME', treatment_var= 
                                                        treatment_variable , state_variable='state', 
                                                         time_variable='time')

        model_1_wbtest = _boot_regression_function(final_data , depvar= 'value', treat_var=  
                                                    treatment_variable , cluster_var= cluster_variable,  
                                                     state_variable='state', time_variable='time')
        model_2_wbtest = _boot_regression_function(final_data , depvar= 'OUTCOME', treat_var=  
                                                    treatment_variable , cluster_var= cluster_variable,  
                                                     state_variable='state', time_variable='time')

        model_1_fgls = _fgls_regression_function(final_data, depvar='value', treat_var= 
                                                  treatment_variable , state_variable='state', 
                                                   time_variable='time')
        model_2_fgls = _fgls_regression_function(final_data, depvar='OUTCOME', treat_var= 
                                                  treatment_variable , state_variable='state', 
                                                   time_variable='time')

        if model_1_ols.pvalues['TREATMENT'] < alpha:
            reject_count_1_ols += 1
        type1_ols = reject_count_1_ols/ (i + 1)
        type1_ols_list.append(type1_ols)

        if model_2_ols.pvalues['TREATMENT'] < alpha:
            reject_count_2_ols += 1
        power_ols = reject_count_2_ols/ (i + 1)
        power_ols_list.append(power_ols)

        if model_1_crse.pvalues['TREATMENT'] < alpha:
            reject_count_1_crse += 1
        type1_crse = reject_count_1_crse/ (i + 1)
        type1_crse_list.append(type1_crse)

        if model_2_crse.pvalues['TREATMENT'] < alpha:
            reject_count_2_crse += 1
        power_crse = reject_count_2_crse/ (i + 1)
        power_crse_list.append(power_crse)

        if model_1_res_agg.pvalues['TREATMENT'] < alpha:
            reject_count_1_res_agg += 1
        type1_res_agg = reject_count_1_res_agg/ (i + 1)
        type1_res_agg_list.append(type1_res_agg)

        if model_2_res_agg.pvalues['TREATMENT'] < alpha:
            reject_count_2_res_agg += 1
        power_res_agg = reject_count_1_res_agg/ (i + 1)
        power_res_agg_list.append(power_res_agg)

        if model_1_wbtest['P-Value'] < alpha:
            reject_count_1_wbtest += 1
        type1_wbtest = reject_count_1_wbtest/ (i + 1)
        type1_wbtest_list.append(type1_wbtest)

        if model_2_wbtest['P-Value'] < alpha:
            reject_count_2_wbtest += 1
        power_wbtest = reject_count_2_wbtest/ (i + 1)
        power_wbtest_list.append(power_wbtest)

        if model_1_fgls.pvalues['TREATMENT'] < alpha:
            reject_count_1_fgls += 1
        type1_fgls = reject_count_1_fgls/ (i + 1)
        type1_fgls_list.append(type1_fgls)

        if model_2_fgls.pvalues['TREATMENT'] < alpha:
            reject_count_2_fgls += 1
        power_fgls = reject_count_2_fgls/ (i + 1)
        power_fgls_list.append(power_fgls)

    df1 = pd.DataFrame({'Simulation': np.arange(1, num_simulations + 1),
                            'Type_1_OLS': type1_ols_list,
                            'Type_1_CRSE': type1_crse_list,
                            'Type_1_Res_Agg': type1_res_agg_list,
                            'Type_1_WBTest': type1_wbtest,
                            'Type_1_Fgls': type1_fgls_list}) 
    fig1_px = px.line(df1, x='Simulation', y=['Type_1_OLS', 'Type_1_CRSE','Type_1_Res_Agg', 
                                              'Type_1_WBTest', 'Type_1_Fgls' ], 
                                               labels={'x': 'Simulation', 'y': 'Type 1 Errors'})
    fig1_px.update_layout(title='Type 1 Error Convergence for Each Method',
                         title_x=0.5,
                        shapes=[dict(type="line", x0=0, y0=alpha, x1=num_simulations, y1=alpha, 
                                     line=dict(color="black", width=1, dash="dot"))])

    df2 = pd.DataFrame({'Simulation': np.arange(1, num_simulations + 1),
                            'Power_OLS': power_ols_list,
                            'Power_CRSE': power_crse_list,
                            'Power_Res_Agg': power_res_agg_list,
                            'Power_WBTest': power_wbtest,
                            'Power_Fgls': power_fgls_list}) 
    fig2_px = px.line(df2, x='Simulation', y=['Power_OLS', 'Power_CRSE','Power_Res_Agg', 
                                              'Power_WBTest', 'Power_Fgls' ], 
                                               labels={'x': 'Simulation', 'y': 'Power'})
    fig2_px.update_layout(title='Power Convergence for Each Method',
                         title_x=0.5,
                        shapes=[dict(type="line", x0=0, y0=alpha, x1=num_simulations, y1=alpha, 
                                      line=dict(color="black", width=1, dash="dot"))])
     
    return fig1_px, fig2_px