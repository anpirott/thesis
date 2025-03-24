import numpy as np
import astropy as ap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import sys


import read_mist_models



def iso_data_to_panda(file : "str", col_names : "list[str]") -> "pd.DataFrame":
    """
    Read in a MIST isochrone file and return a pandas dataframe with the requested columns.

    Parameters:
        file (str) : path to the MIST isochrone file
        col_names (list of str) : names of the columns to be extracted.
            Possible names are: log10_isochrone_age_yr, initial_mass, star_mass, star_mdot, he_core_mass, c_core_mass, log_L, log_LH, log_LHe, log_Teff, log_R, log_g, surface_h1,
                                surface_he3, surface_he4, surface_c12, surface_o16, log_center_T, log_center_Rho, center_gamma, center_h1, center_he4, center_c12, phase
    
    Returns:
        pandas.DataFrame : a pandas dataframe containing the data of the isochrones with the requested columns
    """
    iso = read_mist_models.ISO(file)

    col_dict = {key: [] for key in col_names}

    for iso_ind in range(len(iso.isos)):
        for keys in col_dict.keys():
            col_dict[keys].extend(iso.isos[iso_ind][keys])

    return pd.DataFrame.from_dict(col_dict)