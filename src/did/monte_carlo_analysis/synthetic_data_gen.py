import pandas as pd
import numpy as np


# HOMOGENOUS AR1 DATA GENERATION

def _generate_homogenous_ar1_data(N, T, rho=0, num_individuals=1, mean=0, std_dev=1):
    """
    Generate synthetic panel data following a homogenous AR(1) process.

    Parameters:
    -----------
    N : int
        The number of states or groups in the panel data. Must be a positive integer.
    T : int
        The number of time periods for each state or group. Must be a positive integer.
    rho : float, optional
        The AR(1) autoregressive parameter, specifying the persistence of the process.
        Must be in the range [0, 1]. Default is 0.
    num_individuals : int, optional
        The number of individuals or entities within each state or group. Must be a positive integer. 
        Default is 1.
    mean : float, optional
        The mean of the white noise process added to the data. Default is 0.
    std_dev : float, optional
        The standard deviation of the white noise process added to the data. Must be greater than 0. 
        Default is 1.
    
    Returns:
    --------
    data : pandas.DataFrame
        A pandas DataFrame containing the generated panel data with the following columns:
        - 'state': State or group identifier.
        - 'individual': Individual identifier.
        - 'time': Time period identifier.
        - 'value': Observed value.

    Raises:
    -------
    ValueError:
        If any of the input parameters violate the specified conditions.

    Notes:
    ------
    This function generates synthetic panel data by simulating a homogenous AR(1) process for all the 
    states or groups.
    It uses a given autoregressive parameters (rho), random state-specific intercepts (alpha), and 
    random time-specific coefficients (beta) to generate the data.
    The generated data is returned as a pandas DataFrame with each row representing an observation 
    for a specific state, individual, and time period.

    Example:
    --------
    To generate a synthetic panel dataset with 5 states, 10 time periods, and 100 individuals per 
    state, you can call the function as follows:
    
    >>> data = generate_homogenous_ar1_data(N=5, T=10, num_individuals=100)
    >>> print(data.head())
    """

    if not isinstance(N, int) or N < 2:
        raise ValueError("N must be an integer greater than or equal to 2.")
    
    if not isinstance(T, int) or T < 2:
        raise ValueError("T must be an integer greater than or equal to 2.")
    
    if not isinstance(num_individuals, int) or num_individuals <= 0:
        raise ValueError("num_individuals must be a positive integer.")
    
    if not 0 <= rho <= 1:
        raise ValueError("Invalid values for rho, It should be in the range of [0,1]")
    
    if std_dev <= 0:
        raise ValueError("Invalid values for std_dev, It should be > 0 ")
    
    white_noise = np.random.normal(mean, std_dev, size=(N, num_individuals, T))

    alphas = np.random.normal(0, 1, size=N)
    betas = np.random.normal(0, 1, size=T)

    data = np.zeros((N, num_individuals, T))

    for i in range(N):
        alpha = alphas[i]
        for j in range(num_individuals):
            for t in range(T):
                beta = betas[t]
                if t == 0:
                    data[i, j, t] = alpha + beta + white_noise[i, j, t]
                else:
                    data[i, j, t] = alpha + beta + rho * data[i, j, t - 1] + white_noise[i, j, t]

    reshaped_data = data.reshape((N * num_individuals, T))
    columns = [str(t) for t in range(T)]

    df = pd.DataFrame(reshaped_data, columns=columns)
    df['state'] = np.repeat(np.arange(1, N + 1), num_individuals)
    df['individual'] = np.tile(np.arange(1, num_individuals + 1), N)

    melted_df = pd.melt(df, id_vars=['state', 'individual'], var_name='time', value_name='value')
    melted_df['time'] = melted_df['time'].astype(int)

    return melted_df


# HETEROGENOUS AR1 DATA  GENERATION

def _generate_heterogenous_ar1_data(N, T, num_individuals, mean=0, std_dev=1):
    """
    Generate synthetic panel data following a heterogenous AR(1) process.

    Parameters:
    -----------
    N : int
        The number of states or groups in the panel data. Must be a positive integer.
    T : int
        The number of time periods for each state or group. Must be a positive integer.
    num_individuals : int
        The number of individuals or entities within each state or group. Must be a positive integer.
    mean : float, optional
        The mean of the white noise process added to the data. Default is 0.
    std_dev : float, optional
        The standard deviation of the white noise process added to the data. Must be greater than 0. 
        Default is 1.
    
    Returns:
    --------
    data : pandas.DataFrame
        A pandas DataFrame containing the generated panel data with the following columns:
        - 'state': State or group identifier.
        - 'individual': Individual identifier.
        - 'time': Time period identifier.
        - 'value': Observed value.

    Raises:
    -------
    ValueError:
        If any of the input parameters violate the specified conditions.

    Notes:
    ------
    This function generates synthetic panel data by simulating a heterogenous AR(1) process for all 
    the states or groups.
    It uses a randomly chosen autoregressive parameters (rho) from the uniform distribution 
    U ~ (0.2, 0.9) for each state. It also uses random state-specific intercepts (alpha), 
    and random time-specific coefficients (beta) from the normal distribution to generate 
    the data.
    The generated data is returned as a pandas DataFrame with each row representing an observation 
    for a specific state, individual, and time period.

    Example:
    --------
    To generate a synthetic panel dataset with 5 states, 10 time periods, and 100 individuals per 
    state, you can call the function as follows:
    
    >>> data = generate_heterogenous_ar1_data(N=5, T=10, num_individuals=100)
    >>> print(data.head())
    """
    if not isinstance(N, int) or N < 2:
        raise ValueError("N must be an integer greater than or equal to 2.")
    
    if not isinstance(T, int) or T < 2:
        raise ValueError("T must be an integer greater than or equal to 2.")
    
    if not isinstance(num_individuals, int) or num_individuals < 0:
        raise ValueError("num_individuals must be a positive integer.")
    
    if std_dev <= 0:
        raise ValueError("std_dev must be greater than 0.")
    
   
    white_noise = np.random.normal(mean, std_dev, size=(N, num_individuals, T))

    data = np.zeros((N, num_individuals, T))

    rhos = np.random.uniform(0.2, 0.9, size=N)
    alphas = np.random.normal(0, 1, size=N)
    betas = np.random.normal(0, 1, size=T)

   
    for i in range(N):
        alpha = alphas[i]
        rho = rhos[i]
        for j in range(num_individuals):
            for t in range(T):
                beta = betas[t]
                if t == 0:
                    data[i, j, t] = alpha + beta + white_noise[i, j, t]
                else:
                    data[i, j, t] = alpha + beta + rho * data[i, j, t - 1] + white_noise[i, j, t]

    reshaped_data = data.reshape((N * num_individuals, T))

    
    df = pd.DataFrame(reshaped_data, columns=[f'{t}' for t in range(T)])

    
    df['state'] = np.repeat(np.arange(1, N + 1), num_individuals)


    df['individual'] = np.tile(np.arange(1, num_individuals + 1), N)

    
    melted_df = pd.melt(df, id_vars=['state', 'individual'], var_name='time', value_name='value')

    
    melted_df['time'] = melted_df['time'].astype(int)

    return melted_df


# HOMOGENOUS MA1 DATA GENERATION

def _generate_homogenous_ma1_data(N, T, theta=0.5, num_individuals=1, mean=0, std_dev=1):
    """
    Generate synthetic panel data following a homogenous MA(1) process.

    Parameters:
    -----------
    N : int
        The number of states or groups in the panel data. Must be a positive integer.
    T : int
        The number of time periods for each state or group. Must be a positive integer.
    theta : float, optional
        The MA(1) shock parameter, specifying the persistence of the error process. Must be in the 
        range [0, 1]. Default is 0.5 .
    num_individuals : int, optional
        The number of individuals or entities within each state or group. Must be a positive integer. 
        Default is 1.
    mean : float, optional
        The mean of the white noise process added to the data. Default is 0.
    std_dev : float, optional
        The standard deviation of the white noise process added to the data. Must be greater than 0. 
        Default is 1.
    
    Returns:
    --------
    data : pandas.DataFrame
        A pandas DataFrame containing the generated panel data with the following columns:
        - 'state': State or group identifier.
        - 'individual': Individual identifier.
        - 'time': Time period identifier.
        - 'value': Observed value.

    Raises:
    -------
    ValueError:
        If any of the input parameters violate the specified conditions.

    Notes:
    ------
    This function generates synthetic panel data by simulating a homogenous MA(1) process for all the 
    states or groups.
    It uses a given moving average parameters (theta), random state-specific intercepts (alpha), and 
    random time-specific coefficients (beta) to generate the data.
    The generated data is returned as a pandas DataFrame with each row representing an observation 
    for a specific state, individual, and time period.

    Example:
    --------
    To generate a synthetic panel dataset with 5 states, 10 time periods, and 100 individuals per 
    state, you can call the function as follows:
    
    >>> data = generate_homogenous_ma1_data(N=5, T=10, num_individuals=100)
    >>> print(data.head())
    """
    if not isinstance(N, int) or N < 2:
        raise ValueError("N must be an integer greater than or equal to 2.")
    
    if not isinstance(T, int) or T < 2:
        raise ValueError("T must be an integer greater than or equal to 2.")
    
    if not isinstance(num_individuals, int) or num_individuals <= 0:
        raise ValueError("num_individuals must be a positive integer.")
    
    if not 0 <= theta <= 1:
        raise ValueError("Invalid values for theta, It should be in the range of [0,1]")
    
    if std_dev <= 0:
        raise ValueError("Invalid values for std_dev, It should be > 0 ")
    
    
    white_noise = np.random.normal(mean, std_dev, size=(N, num_individuals, T))

    alphas = np.random.normal(0, 1, size=N)
    betas = np.random.normal(0, 1, size=T)

    data = np.zeros((N, num_individuals, T))

    for i in range(N):
        alpha = alphas[i]
        for j in range(num_individuals):
            for t in range(T):
                beta = betas[t]
                if t == 0:
                    data[i, j, t] = alpha + beta + white_noise[i, j, t]
                else:
                    data[i, j, t] = alpha + beta + theta * white_noise[i, j, t - 1] + white_noise[i, j, t]

    reshaped_data = data.reshape((N * num_individuals, T))
    columns = [str(t) for t in range(T)]

    df = pd.DataFrame(reshaped_data, columns=columns)
    df['state'] = np.repeat(np.arange(1, N + 1), num_individuals)
    df['individual'] = np.tile(np.arange(1, num_individuals + 1), N)

    melted_df = pd.melt(df, id_vars=['state', 'individual'], var_name='time', value_name='value')
    melted_df['time'] = melted_df['time'].astype(int)

    return melted_df