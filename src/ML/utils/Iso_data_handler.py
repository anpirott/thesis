import pandas as pd
import sys
import os

from physics import read_mist_models
from ML.utils.utils import sanitize_path


# TODO? avoir des méthodes qui peuvent dire dans quelle isochrone un point de donnée est, 
# TODO? des plots(?), une meilleur façon de garder les noms des colonnes pour les choisir, de quoi regrouper les données en un set, etc
# TODO rajouter les exceptions pour les erreurs
class Iso_data_handler():
    def __init__(self, path : str, col_names : list[str]):
        """
        Initializes the Iso_data_handler class.

        Parameters:
            path (str) : complete path to the directory containing the MIST isochrone files
            col_names (list of str) : names of the columns to be extracted. If set to an empty list, uses all the columns.
                Possible names are: log10_isochrone_age_yr, initial_mass, star_mass, star_mdot, he_core_mass, c_core_mass, log_L, log_LH, log_LHe, log_Teff, log_R, log_g, surface_h1,
                                    surface_he3, surface_he4, surface_c12, surface_o16, log_center_T, log_center_Rho, center_gamma, center_h1, center_he4, center_c12, phase
        """
        self.path = sanitize_path(path)

        if not isinstance(col_names, list):
            print("Error: col_names should be initialized as a list of strings.")
            sys.exit(1)
        self.col_names = col_names

        self.all_col_names = ["log10_isochrone_age_yr", "initial_mass", "star_mass", "star_mdot", "he_core_mass", "c_core_mass", "log_L", "log_LH", 
                              "log_LHe", "log_Teff", "log_R", "log_g", "surface_h1", "surface_he3", "surface_he4", "surface_c12", "surface_o16", 
                              "log_center_T", "log_center_Rho", "center_gamma", "center_h1", "center_he4", "center_c12", "phase"] # + "metallicity" once the csv has been created
    
    def full_iso_data_to_panda(self, path : str=None, col_names : list[str]=None, override : bool=False) -> pd.DataFrame:
        """
        Reads all .iso files in the given directory and creates a pandas dataframe of all the data with the requested columns.
        The dataframe, with all columns, is saved in the directory under the name "MIST_iso_full_data.csv".

        Parameters:
            path (str) : path to the directory containing the MIST isochrone files
            col_names (list of str) : names of the columns to be extracted. If set to an empty list, uses all the columns.
                Possible names are: log10_isochrone_age_yr, initial_mass, star_mass, star_mdot, he_core_mass, c_core_mass, log_L, log_LH, log_LHe, log_Teff, log_R, log_g, surface_h1,
                                    surface_he3, surface_he4, surface_c12, surface_o16, log_center_T, log_center_Rho, center_gamma, center_h1, center_he4, center_c12, phase
            override (bool) : recomputes the dataframe and save it in a csv file if set to True. 
                              Otherwise, it only computes the dataframe if the file does not exist and returns the saved dataframe if it does.
                                    
        Returns:
            pandas.DataFrame : a pandas dataframe containing the data of the isochrones with the requested columns
        """
        if path is None:
            path = self.path
        if col_names is None:
            if len(self.col_names) == 0:
                col_names = self.all_col_names
            else:
                col_names = self.col_names
        
        if override | (not os.path.exists(path + "MIST_iso_full_data.csv")):
            # creates a dictionary containing an empty list for each given column 
            col_dict = {key: [] for key in self.all_col_names}
            col_dict["metallicity"] = []
            full_iso_df = pd.DataFrame.from_dict(col_dict)
            
            # reads all the .iso files in the directory and appends their data to the dataframe
            for filename in os.listdir(path):
                if filename.endswith(".iso"):
                    iso_df = Iso_data_handler.iso_data_to_panda(path + filename, self.all_col_names)
                    full_iso_df = pd.concat([full_iso_df, iso_df], ignore_index=True)
            
            # saves the dataframe in a csv file
            print("Writing dataframe to csv file...")
            full_iso_df.to_csv(path + "MIST_iso_full_data.csv", sep=',', encoding='utf-8', index=False, header=True)

        else:
            print("Reading dataframe from csv file...")
            full_iso_df = pd.read_csv(path + "MIST_iso_full_data.csv")
        
        return full_iso_df[col_names]

    @staticmethod
    def iso_data_to_panda(path : str, col_names : list[str]) -> pd.DataFrame:
        """
        Reads a MIST isochrone file and returns a pandas dataframe with the requested columns.

        Parameters:
            path (str) : path to the MIST isochrone file
            col_names (list of str) : names of the columns to be extracted.
                Possible names are: log10_isochrone_age_yr, initial_mass, star_mass, star_mdot, he_core_mass, c_core_mass, log_L, log_LH, log_LHe, log_Teff, log_R, log_g, surface_h1,
                                    surface_he3, surface_he4, surface_c12, surface_o16, log_center_T, log_center_Rho, center_gamma, center_h1, center_he4, center_c12, phase
        
        Returns:
            pandas.DataFrame : a pandas dataframe containing the data of the isochrones with the requested columns
        """
        iso = read_mist_models.ISO(path)

        # creates a dictionary containing an empty list for each given column 
        col_dict = {key: [] for key in col_names}
        metallicity = iso.abun['[Fe/H]']

        for iso_ind in range(len(iso.isos)):
            for keys in col_dict.keys():
                col_dict[keys].extend(iso.isos[iso_ind][keys])
        
        iso_df = pd.DataFrame.from_dict(col_dict)
        iso_df["metallicity"] = metallicity

        return iso_df


if __name__ == "__main__":
    pass
    # test_class = Iso_data_handler("C:/Users/antoi/Code/unif/MA2/Thèse/data/MIST_v1.2_vvcrit0.0_basic_isos/",
    #                               ['log10_isochrone_age_yr', 'log_Teff', 'log_g', 'star_mass', 'phase'])
    # test_df = test_class.full_iso_data_to_panda(override=False)
    # print(test_df)
    # initialize data of lists.