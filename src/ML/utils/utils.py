import sys
import os
import math
import numpy as np
import pandas as pd
import csv
import copy
from IPython.display import display


def sanitize_path(path : str) -> str:
    """
    Sanitizes the given path.

    Parameters:
        path (str) : path to be sanitized
    
    Returns:
        str : sanitized path
    """
    if not isinstance(path, str):
        print("Error: path should be initialized as a string.")
        sys.exit(1)
    if "\\" in path:
        path = path.replace("\\", "/")
    if not path.endswith("/"):
        path += "/"
    if not os.path.exists(path):
        print("Error: path does not exist.")
        sys.exit(1)
    return path
    
def gaussian(dsts):
    """
    Computes Gaussian weights based on the given distances.

    Parameters:
        dsts (numpy.ndarray) : array of distances
    
    Returns:
        numpy.ndarray : array of Gaussian weights
    """
    kernel_width = .5
    weights = np.exp(-(dsts**2)/kernel_width)
    return weights

def isclose_pandas_apply(row, col_name, value, bool_index, rel_tol=1e-6):
    if math.isclose(row[col_name], value, rel_tol=rel_tol):
        bool_index.append(True)
    else:
        bool_index.append(False)

def isclose_pandas(df, col_name, value, rel_tol=1e-6):
    bool_index = []
    df.apply(isclose_pandas_apply, axis=1, args=(col_name, value, bool_index, rel_tol))
    return bool_index

def print_uniques(col_name, df):
    uniques = df[col_name].unique()
    uniques.sort()
    if len(uniques) < 50:
        print(col_name + " : " + str(uniques))
    else:
        print(f"{col_name} : {len(uniques)} unique values, range [{uniques[0]}, {uniques[-1]}]")

def print_uniques_count(col_name, df):
    uniques = df[col_name].unique()
    print(f"{col_name} : ")
    for unique in uniques:
        print(f"\t{unique} => {np.count_nonzero(df[col_name] == unique)}")

def compare_metrics(path : str, model_names : list[str], physical_models : list[str], data_filters : list[str], output_parameters : list[str]) -> pd.DataFrame:
    """
    Compares the statistics between one or more models in a single table and displays it. # TODO? aussi le return?
    The path to the metrics and images needs to follow this hierarchy : path/to/results/(training_type/)model_name/physical_model/data_filter/

    Parameters:
        path (str) : path to the results folder containing the models
        model_names list(str) : name of the models which need to be added to the comparison
        physical_models list(str) : which physical model ("MIST" and "PARSEC") needs to be used in the comparison
        data_filters list(str) : which filters need to be used in the comparison
        output_parameters list(str) : which output parameters of the model we want to add to the comparison
    
    the dataframe has the following form:

    | model | physical_models | filter | output_parameter | metrics    |
    |       |                 |                           |RVE|RMSE|...|
    --------------------------------------------------------------------
    | mlp   | MIST            | Base   | mass             |X.X|X.X|... |
    |       |                 |        | radius           |X.X|X.X|... |
    |       |                 | log_g  | mass             |X.X|X.X|... |
    |       |                 |        | radius           |X.X|X.X|... |
    ...
    |       | PARSEC          | Base   | mass             |X.X|X.X|... |
    |       |                 |        | radius           |X.X|X.X|... |
    |       |                 | log_g  | mass             |X.X|X.X|... |
    |       |                 |        | radius           |X.X|X.X|... |
    ...
    | KNN   | MIST            | Base   | mass             |X.X|X.X|... |
    |       |                 |        | radius           |X.X|X.X|... |
    ...
    ...
    """

    path = sanitize_path(path)

    results_dict = dict()
    for model_name in model_names:
        results_dict[model_name] = dict()
        for physical_model in physical_models:
            results_dict[model_name][physical_model] = dict()
            for data_filter in data_filters:
                results_dict[model_name][physical_model][data_filter] = dict()
                full_path = path + f"{model_name}/{physical_model}/{data_filter}/"

                with open(full_path + "metrics.csv", 'r') as metrics_file:
                    with open(full_path + "time_taken.txt", 'r') as time_file:
                        lines = time_file.readlines()
                        dict_reader = csv.DictReader(metrics_file)

                        for metrics_dict in list(dict_reader):
                            output_parameter = metrics_dict.pop("")

                            percentiles = eval(metrics_dict.pop("Percentiles"))
                            str_percentiles = ""
                            for thresh, value in percentiles.items():
                                str_percentiles += f" {thresh} : {round(value, 5)} /"
                            str_percentiles = str_percentiles[:-1]
                            metrics_dict["Percentiles"] = str_percentiles

                            metrics_dict["time"] = round(float(lines[1].split(',')[0]), 5)

                            if output_parameter in output_parameters:
                                results_dict[model_name][physical_model][data_filter][output_parameter] = copy.deepcopy(metrics_dict)
    
    rows = []
    for model_name, physical_model_dict in results_dict.items():
        for physical_model, data_filter_dict in physical_model_dict.items():
            for data_filter, output_parameter_dict in data_filter_dict.items():
                for output_parameter, metrics_dict in output_parameter_dict.items():
                    row = {"model": model_name, "physical_model": physical_model, "filter": data_filter, "output_parameter": output_parameter}
                    row.update(metrics_dict)
                    rows.append(row)
    
    df = pd.DataFrame(rows).set_index(["model", "physical_model", "filter", "output_parameter"])
    pd.set_option('display.max_colwidth', None)
    display(df)


if __name__ == "__main__":
    pass
    # compare_metrics(r"C:\Users\antoi\Code\unif\MA2\thesis\results\K_fold", ["linear_regression"], ["MIST"], ["Base", "phase_filtered"], ["mass", "radius"])

    # path = "C:/Users/antoi/Code/unif/MA2/Thèse/results/K_fold/"
    # print(sanitize_path(path))

