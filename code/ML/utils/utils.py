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


if __name__ == "__main__":
    # test_class = Iso_data_handler("C:/Users/antoi/Code/unif/MA2/Thèse/data/MIST_v1.2_vvcrit0.0_basic_isos/",
    #                               ['log10_isochrone_age_yr', 'log_Teff', 'log_g', 'star_mass', 'phase'])
    # test_df = test_class.full_iso_data_to_panda(override=False)
    # print(test_df)
    # initialize data of lists.

    # test_evaluator = Model_evaluator("test_model", path="C:/Users/antoi/Code/unif/MA2/Thèse/results/K_fold/", residuals_truth_plot=False, residuals_boxplot=False,
    #                                  residuals_histogram=False,qq_plot=False)

    # y_true = np.array([3.0, -0.5, 2.0, 7.0])
    # y_pred = np.array([2.5, -0.5, 2.0, 8.0])

    # test_evaluator.calculate_model_evaluation("test_parameter", truth=y_true, preds=y_pred)
    # test_evaluator.show_model_evaluation("test_parameter")

    # test_evaluator.save_model_evaluation()

    # path = "C:/Users/antoi/Code/unif/MA2/Thèse/results/K_fold/"
    # print(sanitize_path(path))
    pass

