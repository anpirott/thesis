import sys
import os
import math
import numpy as np


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

def compare_metrics(path : list[str], ):
    pass # TODO!

if __name__ == "__main__":
    pass

    # path = "C:/Users/antoi/Code/unif/MA2/Thèse/results/K_fold/"
    # print(sanitize_path(path))

