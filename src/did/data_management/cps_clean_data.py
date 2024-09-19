import pandas as pd
import numpy as np
import statsmodels.api as sm

def process_cps_data(file_path):
    """
    Process the CPS (Current Population Survey) data from a compressed CSV file.

    This function reads the compressed CSV file specified by 'file_path', processes the data,
    and returns a cleaned DataFrame containing the CPS data.

    Parameters:
    -----------
    file_path : str
        The path to the compressed CSV file containing CPS data.

    Returns:
    --------
    pd.DataFrame
        A cleaned DataFrame containing the CPS data.

    Raises:
    -------
    ValueError
        - If any of the following columns are missing in the DataFrame after reading and processing 
          the data:
            - 'INCWAGE'
            - 'EDUC'
            - 'YEAR'
            - 'STATEFIP'
            - 'AGE'
            - 'SEX'
        - If an error occurs while reading the CSV file.

    Notes:a
    ------
    This function performs several data cleaning and processing steps:
        1. Reads the compressed CSV file into a DataFrame.
        2. Filters out invalid 'INCWAGE' values (99999999, 0, 999) and takes the natural logarithm 
           of 'INCWAGE'.
        3. Filters out invalid 'EDUC' values (0, 1).
        4. Filters the data for the specified year range (1980-2000).
        5. Categorizes education levels into four categories: 'Up to Grade 10', 'High School',
           "Master's Degree", and 'Doctorate Degree'.
        6. Creates dummy variables for the 'Education_Category' column using one-hot encoding.
        7. Converts boolean columns ('High School', "Master's Degree", 'Up to Grade 10') to integer
           values (0 or 1).
        8. Filters out rows where 'STATEFIP' is greater than 56 or equal to 11.
        9. Filters the data by age (25-50) and sex (2 for female).
    """

    df = pd.read_csv(file_path, compression='gzip', header=0)

    if 'INCWAGE' not in df.columns:
        raise ValueError("Column 'INCWAGE' does not exist in the DataFrame.")
    
    if 'EDUC' not in df.columns:
        raise ValueError("Column 'EDUC' does not exist in the DataFrame.")
    
    if 'YEAR' not in df.columns:
        raise ValueError("Column 'YEAR' does not exist in the DataFrame.")

    if 'STATEFIP' not in df.columns:
        raise ValueError("Column 'STATEFIP' does not exist in the DataFrame.")
    
    if 'AGE' not in df.columns:
        raise ValueError("Column 'AGE' does not exist in the DataFrame.")
    
    if 'SEX' not in df.columns:
        raise ValueError("Column 'SEX' does not exist in the DataFrame.")
    
    df = df[(df['INCWAGE'] != 99999999) & (df['INCWAGE'] != 0) & (df['INCWAGE'] != 999)]
    df['INCWAGE'] = np.log(df['INCWAGE'])
    
    df = df[(df['EDUC'] != 0) & (df['EDUC'] != 1)]
    
    df = df[(df['YEAR'] >= 1980) & (df['YEAR'] <= 2000)]

    df = df[(df['AGE'] >= 25) & (df['AGE'] <= 50)]
    df = df[df['SEX'] == 2]

    def categorize_education(educ_code):
        if educ_code <= 10:
            return 'Up to Grade 10'
        elif 10 < educ_code <= 70:
            return 'High School'
        elif 70 < educ_code <= 123:
            return "Master's Degree"
        else:
            return 'Doctorate Degree'
    
    df['Education_Category'] = df['EDUC'].apply(categorize_education)
    df = pd.get_dummies(df, columns=['Education_Category'], prefix='', prefix_sep='', drop_first=True)

    boolean_columns = ['High School', "Master's Degree",
                       'Up to Grade 10']

    df[boolean_columns] = df[boolean_columns].astype(int)
    
    df = df[~((df['STATEFIP'] > 56) | (df['STATEFIP'] == 11))]
 
    return df


def aggregate_cps_data(df):

    """
    Aggregate CPS (Current Population Survey) data by grouping and calculating mean residuals.

    This function aggregates CPS data by grouping observations by 'STATEFIP' and 'YEAR' and
    calculating the mean residuals.
    The function assumes that the input DataFrame has been cleaned and contains the necessary
    columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The cleaned CPS data DataFrame obtained from the 'process_cps_data' function.
        It should contain the following columns:
        - 'High School' : int
            Number of individuals with a high school education.
        - "Master's Degree" : int
            Number of individuals with a master's degree.
        - 'Up to Grade 10' : int
            Number of individuals with education up to grade 10.
        - 'AGE' : int
            Age of individuals.
        - 'INCWAGE' : float
            Income/wage of individuals.
        - 'STATEFIP' : int
            State Federal Information Processing Standard (FIPS) code.
        - 'YEAR' : int
            Year of the observation.

    Returns:
    --------
    residuals_mean_by_state_year : pandas.DataFrame
        A DataFrame containing the aggregated CPS data with mean residuals.

    Raises:
    -------
    ValueError:
        If any of the required columns ('INCWAGE', 'High School', "Master's Degree", 'Up to Grade 10', 'AGE') 
        is missing in the DataFrame.

    Notes:
    ------
    - This function assumes that the input DataFrame has been cleaned and contains the required columns.
    - It fits an Ordinary Least Squares (OLS) regression model to the cleaned data to obtain residuals.
    - Residuals are then aggregated by grouping observations by 'STATEFIP' and 'YEAR'.

    Example:
    --------
    To aggregate CPS data and calculate mean residuals, you can call the function as follows:

    >>> aggregated_data = aggregate_cps_data(cleaned_data)

    The 'aggregated_data' DataFrame will contain the aggregated CPS data with mean residuals. 
    """

    if 'INCWAGE' not in df.columns:
        raise ValueError("Column 'INCWAGE' does not exist in the DataFrame.")
    
    if 'High School' not in df.columns:
        raise ValueError("Column 'High School' does not exist in the DataFrame.") 
    
    if "Master's Degree" not in df.columns:
        raise ValueError("Column Master's Degree does not exist in the DataFrame.")

    if 'Up to Grade 10' not in df.columns:
        raise ValueError("Column 'Up to Grade 10' does not exist in the DataFrame.")

    if 'AGE' not in df.columns:
        raise ValueError("Column 'AGE' does not exist in the DataFrame.")

    X = df[['High School', "Master's Degree", 'Up to Grade 10', 'AGE']]
    y = df['INCWAGE']

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    df['Residual_wage'] = model.resid

    residuals_mean_by_state_year = df.groupby(['STATEFIP', 'YEAR'])['Residual_wage'].mean().reset_index()

    return residuals_mean_by_state_year


