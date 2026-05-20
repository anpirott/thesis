import sys
import os
import math
import numpy as np
import pandas as pd
import csv
import copy
from IPython.display import display
from pathlib import Path
from sklearn.metrics import max_error


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

def print_uniques(col_name, df, force=False):
    uniques = df[col_name].unique()
    uniques.sort()
    if force or len(uniques) < 50:
        print(col_name + " : " + str(uniques))
    else:
        print(f"{col_name} : {len(uniques)} unique values, range [{uniques[0]}, {uniques[-1]}]")

def print_uniques_count(col_name, df):
    uniques = df[col_name].unique()
    print(f"{col_name} : ")
    for unique in uniques:
        print(f"\t{unique} => {np.count_nonzero(df[col_name] == unique)}")

def print_all_uniques(df):
    for col_name in df.columns:
        if col_name == "phase" or col_name == "label" or col_name == "metallicity":
            str_data = ""
            for value in df[col_name].unique():
                str_data = str_data + str(value) + ", "
            str_data = str_data[:-2]
            print(f"Values in {col_name} column : {str_data} ")
        else:
            print(f"{col_name}  : Range : {round(min(df[col_name]), 2)} - {round(max(df[col_name]), 4)}, Mean : {round(df[col_name].mean(), 4)}, Median : {round(df[col_name].median(), 4)}")
        print()

def smallest_encompassing_pair(n : int) -> tuple[int, int]:
    """
    Gives the smallest pair of numbers possible which, when multiplied, are bigger than the input value.

    Parameters:
        n (int) : The number we want to encompass with the smallest pair of numbers possible
    
    Returns:
        tuple (int, int) : The two numbers of the smallest pair of numbers possible
    """
    a = int(np.round(np.sqrt(n)))
    b = (n + a - 1) // a 
    return a, b
    
def compare_metrics(path : str, output_parameters : list[str], model_names : list[str]=None, physical_models : list[str]=None, data_filters : list[str]=None,
                    categories : list[str]=None, value_rounding : int=-1, relative_values : bool=False) -> pd.DataFrame:
    """
    Compares the statistics between one or more models in a single table and displays it. # TODO? aussi le return?
    The path to the metrics and images needs to follow this hierarchy : path/to/results/(training_type/)model_name/physical_model/data_filter/

    Parameters:
        path (str) : path to the results folder containing the models
        output_parameters (list[str]) : which output parameters of the model we want to add to the comparison
        model_names (list[str]) : name of the models which need to be added to the comparison; if set to None, uses all possible values in the directory
        physical_models (list[str]) : which physical model ("MIST" and "PARSEC") needs to be used in the comparison; if set to None, uses all possible values in the directory
        data_filters (list[str]) : which filters need to be used in the comparison; if set to None, uses all possible values in the directory
        categories (list[str]) : which categories need to be used in the comparison; if set to None, uses all possible values in the directory
        value_rounding (int) : what rounding of the values needs to be applied; if smaller or equal to 0, does not round values
        relative_values (bool) : wether or not to use relative errors when comparing the models
    
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

    if model_names is None: # TODO? je peux faire une fct de ça mais pas du tout obligatoire
        model_names = []
        for file_or_directory in os.listdir(path):
            if os.path.isdir(path + f"{file_or_directory}"):
                model_names.append(file_or_directory)
    if len(model_names) == 0:
        print("Error : no model with existing results.")
        sys.exit(1)

    if physical_models is None:
        physical_models = []
        new_path = path + f"{model_names[0]}/"
        for file_or_directory in os.listdir(new_path):
            if os.path.isdir(new_path + f"{file_or_directory}"):
                physical_models.append(file_or_directory)
    if len(physical_models) == 0:
        print("Error : no results for any physical model exists.")
        sys.exit(1)

    if data_filters is None:
        data_filters = []
        new_path = path + f"{model_names[0]}/{physical_models[0]}/" # TODO je pense que ça ne fonctionne pas si je n'ai pas les mêmes filtres entre PARSEC et MIST
        for file_or_directory in os.listdir(new_path):
            if os.path.isdir(new_path + f"{file_or_directory}"):
                data_filters.append(file_or_directory)
    if len(data_filters) == 0:
        print("Error : no results for any physical model exists.")
        sys.exit(1)

    check_results_added = False
    init_metrics_path = path + f"{model_names[0]}/{physical_models[0]}/{data_filters[0]}/metrics/" 
    # all the metrics file should have the same name in every metrics directory

    for filename in os.listdir(init_metrics_path):
        cat_name, cat_value = filename.split("_")[0], filename.split("_")[1]
        if categories is not None and cat_name not in categories:
            continue
        print(f"Results for the {cat_name} category with a value of {cat_value}")
        results_dict = dict()
        for model_name in model_names:
            results_dict[model_name] = dict()
            for physical_model in physical_models:
                results_dict[model_name][physical_model] = dict()
                for data_filter in data_filters:
                    results_dict[model_name][physical_model][data_filter] = dict()
                    time_path = path + f"{model_name}/{physical_model}/{data_filter}/"
                    metrics_path = path + f"{model_name}/{physical_model}/{data_filter}/metrics/"

                    with open(metrics_path + filename, 'r') as metrics_file:
                        with open(time_path + "time_taken.txt", 'r') as time_file:
                            lines = time_file.readlines() # retrieving the time taken
                            dict_reader = csv.DictReader(metrics_file) # retrieving the metrics

                            for metrics_dict in list(dict_reader):
                                metrics_dict_copy = copy.deepcopy(metrics_dict)
                                # creating a nicer way to display the percentiles
                                percentiles = eval(metrics_dict_copy.pop("Percentiles"))
                                str_percentiles = ""
                                for thresh, value in percentiles.items():
                                    str_percentiles += f" {thresh} : {round(value, 5)} /"
                                str_percentiles = str_percentiles[:-1]
                                metrics_dict_copy["Percentiles"] = str_percentiles

                                metrics_dict_copy["time"] = lines[1].split(',')[0] # adding the time to the metrics dictionnary

                                output_parameter = metrics_dict_copy.pop("")

                                min_value, max_value = metrics_dict_copy["value range"].split(" - ")
                                denominator = eval(max_value) - eval(min_value) # the denominator which will be used to divide the values

                                for key in metrics_dict_copy.keys():
                                    if key == "Percentiles" and relative_values: # TODO ne round pas avec value_rounding et pas relative_values
                                        str_percentiles = ""
                                        percentiles_dict = eval("{" + f"{metrics_dict_copy[key].replace("/", ",")}" + "}")
                                        for key_percentile in percentiles_dict.keys():
                                            if value_rounding >= 1:
                                                str_percentiles += f"{key_percentile} : {round((percentiles_dict[key_percentile]/denominator)*100, value_rounding)}% / "
                                            else:
                                                str_percentiles += f"{key_percentile} : {(percentiles_dict[key_percentile]/denominator)*100}% / "
                                        metrics_dict_copy[key] = str_percentiles.removesuffix(" / ")
                                    elif key == "value range" and value_rounding >= 1:
                                        min_value, max_value = metrics_dict_copy[key].split(" - ")
                                        min_value = round(eval(min_value), value_rounding)
                                        max_value = round(eval(max_value), value_rounding)
                                        metrics_dict_copy[key] = f"{min_value} - {max_value}"
                                    elif (key == "RVE" or key == "CORR" or key == "time") and value_rounding >= 1:
                                        metrics_dict_copy[key] = round(eval(metrics_dict_copy[key]), value_rounding)
                                    elif value_rounding >= 1 and not relative_values:
                                        if key != "Percentiles":
                                            metrics_dict_copy[key] = round(eval(metrics_dict_copy[key]), value_rounding)
                                        else:
                                            str_percentiles = ""
                                            percentiles_dict = eval("{" + f"{metrics_dict_copy[key].replace("/", ",")}" + "}")
                                            for key_percentile in percentiles_dict.keys():
                                                str_percentiles += f"{key_percentile} : {round((percentiles_dict[key_percentile]), value_rounding)} / "
                                            metrics_dict_copy[key] = str_percentiles.removesuffix(" / ")
                                    elif value_rounding < 1 and relative_values and key != "value range" and key != "RVE" and key != "CORR" and key != "time":
                                        metrics_dict_copy[key] = str((eval(metrics_dict_copy[key])/denominator)*100) + "%"
                                    elif value_rounding >= 1 and relative_values and key != "RVE" and key != "CORR" and key != "time":
                                        metrics_dict_copy[key] = str(round((eval(metrics_dict_copy[key])/denominator)*100, value_rounding)) + "%"

                                if output_parameter in output_parameters: # adding the metrics dictionnary
                                    check_results_added = True
                                    results_dict[model_name][physical_model][data_filter][output_parameter] = metrics_dict_copy

        if not check_results_added:
            print("No results to show, please make sure that the output parameters are correct.")
            sys.exit(1)
        
        # creating the dataframe as is shown in the docstring
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
    
def max_error_single(y_true, y_pred):
    """
    Uses the max_error metric from sklearn with only the first output
    """
    return max_error(y_true[:, 0], y_pred[:, 0])


if __name__ == "__main__":
    pass
    # compare_metrics(r"C:\Users\antoi\Code\unif\MA2\thesis\results\K_fold", ["linear_regression"], ["MIST"], ["Base", "phase_filtered"], ["mass", "radius"])

    # path = "C:/Users/antoi/Code/unif/MA2/Thèse/results/K_fold/"
    # print(sanitize_path(path))

