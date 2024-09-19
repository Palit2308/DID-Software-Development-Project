"""Tasks for managing the data."""

from pathlib import Path
import pandas as pd
from did.config import BLD, SRC
from did.data_management.cps_clean_data import process_cps_data , aggregate_cps_data
from did.data_management.empirical_clean_data import clean_empirical_data, create_dict, treatment_assignment


cps_clean_dep = {
    "scripts": Path("cps_clean_data.py"),
    "data": SRC / "data" / "cps_00006.csv.gz",
}

def task_clean_cps_data(
        depends_on = cps_clean_dep,
        produces = BLD / "python" / "cps" / "data" / "clean_data.csv",
):
    data = process_cps_data(depends_on["data"])
    data.to_csv(produces, index = False)

cps_agg_dep = {
    "scripts": Path("cps_clean_data.py"),
    "data": BLD / "python" / "cps" / "data" / "clean_data.csv",
}

def task_agg_cps_data(
        depends_on = cps_agg_dep,
        produces = BLD / "python" / "cps" / "data" / "agg_data.csv",
):
    df = pd.read_csv(depends_on["data"])
    data = aggregate_cps_data(df)
    data.to_csv(produces, index = False)


empirical_clean_dep = {
    "scripts": Path("empirical_clean_data.py"),
    "data": SRC/ "data" / "Trade_Indices_E_All_Data_NOFLAG.csv.gz",
}

def task_clean_epirical_data(
        depends_on = empirical_clean_dep,
        produces = BLD / "python" / "empirical_study" / "data" / "long_data.csv"
):
    file_path = depends_on["data"]
    data = clean_empirical_data(file_path)
    data.to_csv(produces, index = False)


empirical_final_dep = {
    "scripts": Path("empirical_clean_data.py"),
    "data": BLD / "python" / "empirical_study" / "data" / "long_data.csv"

}

def task_treat_empirical_data(
        depends_on = empirical_final_dep,
        produces = BLD / "python" / "empirical_study" / "data" / "treated_data.csv"
):
    dict = create_dict()
    data = pd.read_csv(depends_on["data"])
    model = treatment_assignment(data, dict)
    model.to_csv(produces, index = False)