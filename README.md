# EPP Project : Difference in Differences Estimation Methods

Authors: Sneha Roy (Matriculation number : 50078578)
         Biswajit Palit (Matriculation Number : 50071214)

## Background

This project has been jointly conducted by us. The main idea of the project is to test the efficacy of different econometric methods in the context of difference in differences estimation when the treatment is in a staggered approach. We have tried to analyse the abilty of the different methods to provide us with a correctly sized test (Type I error of 5%, because the chosen significance level is 5%) and the power of the methods to detect real effects, in the context of a panel dataset. The different methods that we test are Ordinary Least Squares (OLS) , Cluster-Robust Standard Error (CRSE), Residual Aggregation, Wild-Cluster Bootstrapping and Feasible Generalised Least Squares (FGLS).

To this end, we have conducted extensive Monte Carlo Simulations with a variety of data generation processes such as Homogenous AR1, Heterogenous AR1 and Homogenous MA1 process across 50 states and 20 time periods. We also cross-validated the efficacy of these methods on a real world dataset of the Census Population Survey concerning the weekly earnings of women for the 50 states in USA across the years 1980 to 2000 inclusive.

Finally, we apply the best method that we identify ( which gives the consistent test size and highest power) to understand the treatment effect of WTO ascension on the agricultural imports of the Article XII members (the developing countries which ascended to WTO post 1995 in a staggered approach).

The detailed methodology of the procedure that we followed is given in the paper named "did.pdf" that generates automatically once the project is built. A presentation called "did_pres.pdf" contaning the highlights of our results will also be generated.

Our results are completely reproducible as we have accounted all avenues of randomness in the project. The seed 42 was used globally across all the functions. 

We have kept the number of simulations for all the analyses to be 200, to restrict the run time below an hour. To run the analysis for more simulations, the argument 'num_simulations' can be updated accordingly in the data_info.yaml file.

We have included both the functions for error handling and the testing in the respective test folders.

## Usage

To get started, create and activate the environment with

```console
$ conda/mamba env create -f environment.yml
$ conda activate did
$ pip install kaleido==0.1.0.post1
```

To build the project, type

```console
$ pytask
```

To test the functions, type

```console
$ pytest
```

## Project Structure

- SRC :
  The SRC folder containing all the codes, regarding building the project is structured into four subfolders.

     1. data : This folder contains the CPS micro data and the Empirical Project raw data namely cps_00006.csv.gz and Trade_Indices_E_All_Data_NOFLAG.csv.gz respectively.

     2. data_management :
           1. cps_clean_data.py : This file contains the functions to clean and aggregate the micro level CPS data.
           2. empirical_clean_data.py : This file contains the fucntions to clean and prepare the empirical dataset for the regression analysis.
           3. data_info.yaml : This file has all the necessary arguments defined to build the project in form of a dictionary.
           4. task_data_management.py : This file contains all the related tasks of this folder for execution.

     3. monte_carlo_analysis :
           1. synthetic_data_gen.py : This file contains the synthetic data generation fucntions.
           2. subfunctions.py : This file contains all the subfunctions which are required to run the Monte Carlo simulation functions.
           3. homogenous_AR1.py : This file contains the Monte Carlo simulations for all the methods, for the data generation process of homogenous AR1.
           4. heterogenous_AR1.py : This file contains the Monte Carlo simulations for all the methods, for the data generation process of heterogenous AR1.
           5. homogenous_MA1.py : This file contains the Monte Carlo simulations for all the methods, for the data generation process of homogenous MA1.
           6. cps_monte_carlo.py : This file contains the Monte Carlo simulations for all the methods for the CPS dataset.
           7. empirical_regression.py : This file contains the regression analysis of the empirical dataset using the OLS and the CRSE method.
           8. task_analysis.py : This file contains all the related tasks of this folder for execution.
           
     4. final :
           1. plot.py : This file contains the plot functions of both Type I Error and Power for all the data generation processes of all the methods.
           2. task_final.py : This file contains all the related tasks of this folder for execution. 

     5. config.py : This file contains all the necessary configurations and lists used for executing the tasks.

     6. utilities.py : This file contains the function to read the .yaml files.

- tests :
   The tests folder contains all the tests for the functions.
     1. analysis : This folder has 2 files namely test_data_gen.py and test_subfunctions.py
     2. data_management : This folder contains 2 files namely test_cps_clean_data.py and test_empirical_clean_data.py  

- paper :
   The paper folder contains the .tex files for generating the pdfs, namely did.tex , did_pres.tex and the task_paper.py which contains all the tasks for the generation of the Latex outputs.      

- Structure of the **BLD** folder which generates on running the project :  

  The BLD folders has 2 subfolders : latex and python.
     1. latex : It contains the paper and the presentation generated.
     2. python : It is structured as follows :
          1. cps : This folder contains the processed CPS data and all the results pertaining to the CPS dataset.
          2. empirical_study : This folder contains the processed emprical project data and all the results pertaining to the empirical project dataset.
          3. tables : It contains 3 subfolders for each data generation process and each of these folder contains the table outputs of the Monte Carlo Analysis for each method.
          4. figures : It contains 2 folders type_1 and power, each of which contains the Type I and the Power convergence plots for each data generation method. 
    


[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Palit2308/did/main.svg)](https://results.pre-commit.ci/latest/github/Palit2308/did/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
