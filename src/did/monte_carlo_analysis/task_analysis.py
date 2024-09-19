from pathlib import Path
import pandas as pd
import pytask
from did.monte_carlo_analysis.homogenous_AR1 import monte_carlo_homogenous_AR1_ols, monte_carlo_homogenous_AR1_crse, monte_carlo_homogenous_AR1_res_agg, monte_carlo_homogenous_AR1_wbtest,monte_carlo_homogenous_AR1_fgls
from did.monte_carlo_analysis.cps_monte_carlo import cps_ols_monte_carlo, cps_crse_monte_carlo, cps_res_agg_monte_carlo, cps_wbtest_monte_carlo, cps_fgls_monte_carlo
from did.monte_carlo_analysis.heterogenous_AR1 import monte_carlo_heterogenous_AR1_ols, monte_carlo_heterogenous_AR1_crse, monte_carlo_heterogenous_AR1_res_agg, monte_carlo_heterogenous_AR1_wbtest, monte_carlo_heterogenous_AR1_fgls
from did.monte_carlo_analysis.homogenous_MA1 import monte_carlo_homogenous_MA1_ols, monte_carlo_homogenous_MA1_crse, monte_carlo_homogenous_MA1_res_agg, monte_carlo_homogenous_MA1_wbtest, monte_carlo_homogenous_MA1_fgls
from did.monte_carlo_analysis.empirical_regression import fit_empirical_crse_model, fit_empirical_ols_model
from did.config import BLD, SRC, METHODS
from did.utilities import read_yaml
import warnings




for method in METHODS:
    cps_deps = {
        "scripts": [Path("cps_monte_carlo.py")],
        "data": BLD / "python" / "cps" / "data" / "agg_data.csv",
        "data_info": SRC / "data_management" / "data_info.yaml" 
    }
    cps_methods = {}

    @pytask.task(id=method)
    def task_cps_monte_carlo(
        method=method, 
        depends_on=cps_deps,
        produces=BLD / "python" / "cps" / "cps_results" / f"cps_{method}.tex",
    ):
        method_name = f"cps_{method}_monte_carlo"
        cps_methods[method] = globals()[method_name]
        data = pd.read_csv(depends_on["data"])
        data_info = read_yaml(depends_on["data_info"])
        model = cps_methods[method](data, data_info)
        latex_table = model.to_latex(index=False)
        with open(produces, 'w') as f:
            f.write(latex_table)


for method in METHODS:
    homogenous_ar1_monte_carlo_dep = {
        "scripts": [Path("homogenous_AR1.py")],
        "data_info": SRC / "data_management" / "data_info.yaml",
    }

    homogenous_ar1_methods = {}

    @pytask.task(id=method)
    def task_monte_carlo_hom_ar1(
        method=method,
        depends_on=homogenous_ar1_monte_carlo_dep,
        produces=BLD / "python" / "tables" / "Homogenous_AR1" / f"{method}.tex"
    ): 
        method_name = f"monte_carlo_homogenous_AR1_{method}"
        homogenous_ar1_methods[method] = globals()[method_name]
        data_info = read_yaml(depends_on["data_info"])
        model = homogenous_ar1_methods[method](data_info)
        latex_table = model.to_latex(index=False)
        with open(produces, 'w') as f:
            f.write(latex_table)




for method in METHODS:
    heterogenous_ar1_monte_carlo_dep = {
        "scripts": [Path("heterogenous_AR1.py")],
        "data_info": SRC / "data_management" / "data_info.yaml",
    }

    heterogenous_ar1_methods = {}

    @pytask.task(id=method)
    def task_monte_carlo_het_ar1(
        method=method,
        depends_on=heterogenous_ar1_monte_carlo_dep,
        produces=BLD / "python" / "tables" / "Heterogenous_AR1" / f"{method}.tex"
    ): 
        method_name = f"monte_carlo_heterogenous_AR1_{method}"
        heterogenous_ar1_methods[method] = globals()[method_name]
        data_info = read_yaml(depends_on["data_info"])
        model = heterogenous_ar1_methods[method](data_info)
        latex_table = model.to_latex(index=False)
        with open(produces, 'w') as f:
            f.write(latex_table)


for method in METHODS:
    homogenous_ma1_monte_carlo_dep = {
        "scripts": [Path("homogenous_MA1.py")],
        "data_info": SRC / "data_management" / "data_info.yaml",
    }

    homogenous_ma1_methods = {}

    @pytask.task(id=method)
    def task_monte_carlo_hom_ma1(
        method=method,
        depends_on=homogenous_ma1_monte_carlo_dep,
        produces=BLD / "python" / "tables" / "Homogenous_MA1" / f"{method}.tex"
    ): 
        method_name = f"monte_carlo_homogenous_MA1_{method}"
        homogenous_ma1_methods[method] = globals()[method_name]
        data_info = read_yaml(depends_on["data_info"])
        model = homogenous_ma1_methods[method](data_info)
        latex_table = model.to_latex(index=False)
        with open(produces, 'w') as f:
            f.write(latex_table)


empirical_deps = {
    "scripts": [Path("empirical_regression.py")],
    "data": BLD / "python" / "empirical_study" / "data" / "treated_data.csv",
    "data_info" : SRC / "data_management" / "data_info.yaml",
}


def task_crse_save_empirical_table(
        depends_on = empirical_deps,
        produces = BLD / "python" / "empirical_study" / "results"/ "empirical_results_crse.tex"

): 
    warnings.filterwarnings("ignore")
    data = pd.read_csv(depends_on["data"])
    data_info = read_yaml(depends_on["data_info"])
    df = fit_empirical_crse_model(data, data_info)
    table = df.to_latex(index = False)
    with open(produces, "w") as f:
        f.writelines(table)


def task_ols_save_empirical_table(
        depends_on = empirical_deps,
        produces = BLD / "python" / "empirical_study" / "results"/ "empirical_results_ols.tex"

):
    data = pd.read_csv(depends_on["data"])
    data_info = read_yaml(depends_on["data_info"])
    model = fit_empirical_ols_model(data, data_info)
    table = model.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table)

warnings.resetwarnings()        