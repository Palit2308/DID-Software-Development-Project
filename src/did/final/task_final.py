import pytask
from pathlib import Path
from did.final.plot import plot_homogenous_AR1 , plot_heterogenous_AR1 , plot_homogenous_MA1
from did.config import BLD, SRC
from did.utilities import read_yaml

plot_dep = {
    "scripts": [Path("plot.py")],
    "data_info": SRC / "data_management" / "data_info.yaml"
}

def task_plot_homogenous_ar1(
        depends_on = plot_dep,
        produces = [BLD /  "python" / "figures" / "type_1" / "homogenous_ar1_type1_convergence.png",
                    BLD /  "python" / "figures" / "power" / "homogenous_ar1_power_convergence.png"]
):
    data_info = read_yaml(depends_on["data_info"])
    figure1 , figure2 = plot_homogenous_AR1(data_info= data_info)
    figure1.write_image(produces[0])
    figure2.write_image(produces[1])

def task_plot_heterogenous_ar1(
        depends_on = plot_dep,
        produces = [BLD /  "python" / "figures" / "type_1" / "heterogenous_ar1_type1_convergence.png",
                    BLD /  "python" / "figures" / "power" / "heterogenous_ar1_power_convergence.png"]
):
    data_info = read_yaml(depends_on["data_info"])
    figure1 , figure2 = plot_heterogenous_AR1(data_info= data_info)
    figure1.write_image(produces[0])
    figure2.write_image(produces[1])

def task_plot_homogenous_ma1(
        depends_on = plot_dep,
        produces = [BLD /  "python" / "figures" / "type_1" / "homogenous_ma1_type1_convergence.png",
                    BLD /  "python" / "figures" / "power" / "homogenous_ma1_power_convergence.png"]
):
    data_info = read_yaml(depends_on["data_info"])
    figure1 , figure2 = plot_homogenous_MA1(data_info= data_info)
    figure1.write_image(produces[0])
    figure2.write_image(produces[1])