import pandas as pd


def clean_empirical_data(file_path):

    """
    Clean and preprocess empirical agricultural import data.

    This function reads a CSV file containing empirical agricultural import data,
    performs data cleaning and preprocessing, and returns a DataFrame with cleaned data.

    Parameters:
    -----------
    file_path : str
        The file path to the CSV file containing empirical agricultural import data.

    Returns:
    --------
    df_selected : pandas.DataFrame
        A DataFrame containing cleaned and preprocessed empirical agricultural import data.
        It includes columns for the area, year, and total agricultural imports.

    Raises:
    -------
    ValueError:
        If the specified file path is incorrect or the file cannot be decoded with the specified 
        encoding.
        If one or more columns specified for dropping do not exist in the DataFrame.
        If one or more specified years for melting do not exist in the DataFrame.

    Notes:
    ------
    - The function reads the CSV file with compression "gzip" and encoding 'latin1'.
    - It filters the DataFrame to include only rows where the 'Element' column equals 
      'Import Value Base Period Quantity'.
    - Missing values are filled with 0.
    - Columns corresponding to years prior to 1992 and after 2015 are dropped.
    - The DataFrame is then grouped by 'Area' and summed over the years 1992 to 2015.
    - The resulting DataFrame is melted to have separate rows for each year.
    - The 'Year' column is converted to integers after removing the 'Y' prefix.
    - Only selected areas are retained in the final DataFrame.
    
    Example:
    --------
    To clean and preprocess empirical agricultural import data, you can call the function as follows:

    >>> cleaned_data = clean_empirical_data('path_to_file.csv.gz')

    The 'cleaned_data' DataFrame will contain the cleaned and preprocessed empirical agricultural
    import data.

    """

    try:
        df = pd.read_csv(file_path, compression = "gzip", encoding='latin1')
    except FileNotFoundError:
        raise ValueError("File not found at the specified path.")
    except UnicodeDecodeError:
        raise ValueError("Unable to decode the file with the specified encoding.")

    df = df[df['Element'] == 'Import Value Base Period Quantity']
    df = df.fillna(0)

    columns_to_drop = ['Y1961', 'Y1962', 'Y1963', 'Y1964', 'Y1965', 'Y1966',
                    'Y1967', 'Y1968', 'Y1969', 'Y1970', 'Y1971', 'Y1972',
                    'Y1973', 'Y1974', 'Y1975', 'Y1976', 'Y1977', 'Y1978', 
                    'Y1979', 'Y1980', 'Y1981', 'Y1982', 'Y1983', 'Y1984',  
                    'Y1985', 'Y1986', 'Y1987', 'Y1988', 'Y1989', 'Y1990',                        
                    'Y1991', 'Y2016', 'Y2017', 'Y2018', 'Y2019', 'Y2020',
                    'Y2021', 'Y2022'         
        ]

    try:
        df = df.drop(columns_to_drop, axis = 1)
    except KeyError:
        raise ValueError("One or more columns specified for dropping do not exist in the DataFrame.")

    subset_df = df
        
    sum_by_area = subset_df.groupby('Area')[ [ 'Y1992', 'Y1993', 'Y1994','Y1995', 'Y1996',
                                            'Y1997', 'Y1998', 'Y1999', 'Y2000', 'Y2001', 'Y2002',
                                            'Y2003', 'Y2004', 'Y2005', 'Y2006', 'Y2007', 'Y2008', 'Y2009', 
                                            'Y2010', 'Y2011', 'Y2012', 'Y2013', 'Y2014', 'Y2015']].sum()
        
    sum_by_area_df = pd.DataFrame(sum_by_area)

    sum_by_area_df = sum_by_area_df.reset_index()
    id_vars = ['Area']
    value_vars = sum_by_area_df.columns[1:]

    try:
        df_long = pd.melt(sum_by_area_df, id_vars=id_vars, value_vars=value_vars, var_name='Year', 
                          value_name='Total_Agri_Imports')
    except KeyError:
        raise ValueError("One or more specified years for melting do not exist in the DataFrame.")

    df_long['Year'] = df_long['Year'].str.replace('Y', '').astype(int)

    selected_areas = ['Albania', 'Angola', 'Armenia', 'Bangladesh', 'Barbados', 'Belize',
        'Benin', 'Bolivia (Plurinational State of)', 'Botswana', 'Brazil',
        'Bulgaria', 'Cape Verde', 'Cambodia', 'Cameroon',
        'Central African Republic', 'Chad', 'Chile', 'China, mainland',
        'Congo', 'Costa Rica', "Côte d'Ivoire", 'Croatia', 'Cuba',
        'Congo, Dem. Rep', 'Dominica', 'Dominican Republic', 'Ecuador',
        'El Salvador', 'Estonia', 'Fiji', 'Gambia', 'Georgia', 'Ghana',
        'Grenada', 'Guatemala', 'Guinea-Bissau', 'Guyana', 'Haiti',
        'Honduras', 'India', 'Indonesia', 'Jamaica', 'Jordan',
        'Kazakhstan', 'Kenya', "Lao People's Democratic Republic",
        'Latvia', 'Lesotho', 'Lithuania', 'Maldives', 'Mauritania',
        'Mauritius', 'Mongolia', 'Morocco', 'Myanmar', 'Namibia', 'Nepal',
        'Nicaragua', 'Nigeria', 'North Macedonia', 'Pakistan', 'Panama',
        'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Moldova',
        'Russian Federation', 'St Vincent', 'Samoa', 'Senegal',
        'Solomon Islands', 'South Africa', 'Sri Lanka', 'Suriname',
        'Tajikistan', 'Thailand', 'Togo', 'Tonga', 'Tunisia', 'Türkiye',
        'Ukraine', 'Uruguay', 'Vanuatu',
        'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'Yemen',
        'Zambia', 'Zimbabwe']

    df_selected = df_long[df_long['Area'].isin(selected_areas)]

    return df_selected


def create_dict():

    """
    Create a dictionary with Article XII countries and their joining dates in the World Trade
    Organisation (WTO).

    This function creates a dictionary where the keys are Article XII country names and the values 
    are their respective joining dates in the World Trade Organisation (WTO).

    Returns:
    --------
    countries_dict : dict
        A dictionary containing Article XII country names as keys and their joining dates in the WTO
        as values.

    Example:
    --------
    To obtain the dictionary of Article XII countries and their joining dates in the WTO, you can
    call the function as follows:

    >>> countries_info = create_dict()

    The 'countries_info' dictionary will contain Article XII country names as keys and their
    WTO joining dates as values.
    """

    countries_dict = {
        'Mongolia': 1997,
        'Ecuador': 1996,
        'Latvia': 1999,
        'Estonia': 1999,
        'North Macedonia': 2003,
        'Panama': 1997,
        'Nepal': 2004,
        'Bulgaria': 1996,
        "Lao People's Democratic Republic": 2013,
        'Cape Verde': 2008,
        'Croatia': 2000,
        'Moldova': 2001,
        'Cambodia': 2009,
        'Lithuania': 2001,
        'Tonga': 2007,
        'Yemen': 2014,
        'Georgia': 2000,
        'Albania': 2000,
        'Jordan': 2000,
        'Vanuatu': 2012,
        'Montenegro': 2012,
        'Samoa': 2012,
        'Armenia': 2003,
        'Tajikistan': 2013,
        'Ukraine': 2008,
        'Viet Nam': 2007,
        'China': 2001,
        'Kazakhstan': 2015,
        'Russian Federation': 2012
    }

    return countries_dict



def treatment_assignment(data, dict):
   
    """
    Assign treatment based on the joining dates of countries in a given dictionary.

    This function assigns a treatment variable to each row in the DataFrame based on the joining 
    dates of countries provided in a dictionary. If a country belonging to the DataFrame is a key
    in the dictionary, it receives the treatment assignment 1 for the year specified as the value 
    in the dictionary and all the years after that year.

    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame containing information about areas and years.
    country_joining_dates : dict
        A dictionary where keys are country names and values are their joining dates.

    Returns:
    --------
    data : pandas.DataFrame
        The DataFrame with the 'TREATMENT' column added based on the treatment assignment.

    Raises:
    -------
    ValueError:
        If one or more keys in the dictionary do not match with values in the DataFrame's 'Area' 
        column.
        If the 'Year' values or dictionary values are not compatible for comparison.

    Example:
    --------
    To assign treatment based on the joining dates of countries in a dictionary 'countries_joining_dates'
    to a DataFrame 'df', you can call the function as follows:

    >>> treated_df = treatment_assignment(df, countries_joining_dates)

    The 'treated_df' DataFrame will contain the original data along with the 'TREATMENT' column
    indicating the treatment assignment.

    Note:
    -----
    This function assigns treatment only to the countries present in the dictionary and for the year 
    and after the year given as the value for each country in the dictionary.

    """

    try:
        data['TREATMENT'] = data.apply(lambda x: 1 if x['Area'] in dict and x['Year'] >= dict[x['Area']] 
                                       else 0, axis=1)
    except KeyError:
        raise ValueError("One or more keys in the dictionary do not match with the 'Area' column.")
    except TypeError:
        raise ValueError("The 'Year' values or dictionary values are not compatible for comparison")
    return data