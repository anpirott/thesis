import pandas as pd
from astropy.table import Table
import sys
import os

from physics import read_mist_models
from ML.utils.utils import sanitize_path

# TODO plutôt mettre ça dans "physics"?

# TODO? avoir des méthodes qui peuvent dire dans quelle isochrone un point de donnée est, 
# TODO? des plots(?), une meilleur façon de garder les noms des colonnes pour les choisir, de quoi regrouper les données en un set, etc
# TODO rajouter les exceptions pour les erreurs
class Iso_data_handler():
    """
    Class used to handle the isochrone data from either MIST or PARSEC

    Methods:
        get_isochrone_dataframe (pd.DataFrame) : 
        _full_PARSEC_data_to_panda (pd.DataFrame) :
        _full_MIST_data_to_panda (pd.DataFrame) :
        _MIST_data_to_panda (pd.DataFrame) :
    """
    def __init__(self, path : str, col_names : list[str], physical_model : str):
        """
        Initializes the Iso_data_handler class.

        Parameters:
            path (str) : complete path to the directory containing the MIST isochrone files
            col_names (list of str) : names of the columns to be extracted. If set to an empty list, uses all the columns.
                Possible names for MIST are: log10_isochrone_age_yr, initial_mass, star_mass, star_mdot, he_core_mass, c_core_mass, log_L, log_LH, 
                                             log_LHe, log_Teff, log_R, log_g, surface_h1,surface_he3, surface_he4, surface_c12, surface_o16, 
                                             log_center_T, log_center_Rho, center_gamma, center_h1, center_he4, center_c12, phase, metallicity
                
                Possible names for PARSEC are: Zini, MH, logAge, Mini, int_IMF, Mass, logL, logTe, logg, label, McoreTP, C_O, period0, 
                                               period1, period2, period3, period4, pmode, Mloss, tau1m, X, Y, Xc, Xn, Xo, Cexcess, Z, 
                                               Teff0, omega, angvel, vtaneq, angmom, Rpol, Req, mbolmag, U_fSBmag, B_fSBmag, V_fSBmag, R_fSBmag, 
                                               I_fSBmag, J_fSBmag, H_fSBmag, K_fSBmag, U_fSB, U_f0, U_fk, U_i00, U_i05, U_i10, U_i15, U_i20, 
                                               U_i25, U_i30, U_i35, U_i40, U_i45, U_i50, U_i55, U_i60, U_i65, U_i70, U_i75, U_i80, U_i85, 
                                               U_i90, B_fSB, B_f0, B_fk, B_i00, B_i05, B_i10, B_i15, B_i20, B_i25, B_i30, B_i35, B_i40, 
                                               B_i45, B_i50, B_i55, B_i60, B_i65, B_i70, B_i75, B_i80, B_i85, B_i90, V_fSB, V_f0, V_fk, 
                                               V_i00, V_i05, V_i10, V_i15, V_i20, V_i25, V_i30, V_i35, V_i40, V_i45, V_i50, V_i55, V_i60, 
                                               V_i65, V_i70, V_i75, V_i80, V_i85, V_i90, R_fSB, R_f0, R_fk, R_i00, R_i05, R_i10, R_i15, 
                                               R_i20, R_i25, R_i30, R_i35, R_i40, R_i45, R_i50, R_i55, R_i60, R_i65, R_i70, R_i75, R_i80, 
                                               R_i85, R_i90, I_fSB, I_f0, I_fk, I_i00, I_i05, I_i10, I_i15, I_i20, I_i25, I_i30, I_i35, 
                                               I_i40, I_i45, I_i50, I_i55, I_i60, I_i65, I_i70, I_i75, I_i80, I_i85, I_i90, J_fSB, J_f0, 
                                               J_fk, J_i00, J_i05, J_i10, J_i15, J_i20, J_i25, J_i30, J_i35, J_i40, J_i45, J_i50, J_i55, 
                                               J_i60, J_i65, J_i70, J_i75, J_i80, J_i85, J_i90, H_fSB, H_f0, H_fk, H_i00, H_i05, H_i10, 
                                               H_i15, H_i20, H_i25, H_i30, H_i35, H_i40, H_i45, H_i50, H_i55, H_i60, H_i65, H_i70, H_i75, 
                                               H_i80, H_i85, H_i90, K_fSB, K_f0, K_fk, K_i00, K_i05, K_i10, K_i15, K_i20, K_i25, K_i30, 
                                               K_i35, K_i40, K_i45, K_i50, K_i55, K_i60, K_i65, K_i70, K_i75, K_i80, K_i85, K_i90
            physical_model (str): defines which data to use, either "MIST" or "PARSEC"
        """
        self.path = sanitize_path(path)

        if not isinstance(col_names, list):
            print("Error: col_names should be initialized as a list of strings.")
            sys.exit(1)
        self.col_names = col_names
        self.physical_model = physical_model

        self.all_MIST_col_names = ["log10_isochrone_age_yr", "initial_mass", "star_mass", "star_mdot", "he_core_mass", "c_core_mass", "log_L", "log_LH", 
                                   "log_LHe", "log_Teff", "log_R", "log_g", "surface_h1", "surface_he3", "surface_he4", "surface_c12", "surface_o16", 
                                   "log_center_T", "log_center_Rho", "center_gamma", "center_h1", "center_he4", "center_c12", "phase"] # + "metallicity" once the csv has been created
        
        self.all_PARSEC_col_names = ['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 'label', 'McoreTP', 'C_O', 'period0', 
                                     'period1', 'period2', 'period3', 'period4', 'pmode', 'Mloss', 'tau1m', 'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess', 'Z', 
                                     'Teff0', 'omega', 'angvel', 'vtaneq', 'angmom', 'Rpol', 'Req', 'mbolmag', 'U_fSBmag', 'B_fSBmag', 'V_fSBmag', 'R_fSBmag', 
                                     'I_fSBmag', 'J_fSBmag', 'H_fSBmag', 'K_fSBmag', 'U_fSB', 'U_f0', 'U_fk', 'U_i00', 'U_i05', 'U_i10', 'U_i15', 'U_i20', 
                                     'U_i25', 'U_i30', 'U_i35', 'U_i40', 'U_i45', 'U_i50', 'U_i55', 'U_i60', 'U_i65', 'U_i70', 'U_i75', 'U_i80', 'U_i85', 
                                     'U_i90', 'B_fSB', 'B_f0', 'B_fk', 'B_i00', 'B_i05', 'B_i10', 'B_i15', 'B_i20', 'B_i25', 'B_i30', 'B_i35', 'B_i40', 
                                     'B_i45', 'B_i50', 'B_i55', 'B_i60', 'B_i65', 'B_i70', 'B_i75', 'B_i80', 'B_i85', 'B_i90', 'V_fSB', 'V_f0', 'V_fk', 
                                     'V_i00', 'V_i05', 'V_i10', 'V_i15', 'V_i20', 'V_i25', 'V_i30', 'V_i35', 'V_i40', 'V_i45', 'V_i50', 'V_i55', 'V_i60', 
                                     'V_i65', 'V_i70', 'V_i75', 'V_i80', 'V_i85', 'V_i90', 'R_fSB', 'R_f0', 'R_fk', 'R_i00', 'R_i05', 'R_i10', 'R_i15', 
                                     'R_i20', 'R_i25', 'R_i30', 'R_i35', 'R_i40', 'R_i45', 'R_i50', 'R_i55', 'R_i60', 'R_i65', 'R_i70', 'R_i75', 'R_i80', 
                                     'R_i85', 'R_i90', 'I_fSB', 'I_f0', 'I_fk', 'I_i00', 'I_i05', 'I_i10', 'I_i15', 'I_i20', 'I_i25', 'I_i30', 'I_i35', 
                                     'I_i40', 'I_i45', 'I_i50', 'I_i55', 'I_i60', 'I_i65', 'I_i70', 'I_i75', 'I_i80', 'I_i85', 'I_i90', 'J_fSB', 'J_f0', 
                                     'J_fk', 'J_i00', 'J_i05', 'J_i10', 'J_i15', 'J_i20', 'J_i25', 'J_i30', 'J_i35', 'J_i40', 'J_i45', 'J_i50', 'J_i55', 
                                     'J_i60', 'J_i65', 'J_i70', 'J_i75', 'J_i80', 'J_i85', 'J_i90', 'H_fSB', 'H_f0', 'H_fk', 'H_i00', 'H_i05', 'H_i10', 
                                     'H_i15', 'H_i20', 'H_i25', 'H_i30', 'H_i35', 'H_i40', 'H_i45', 'H_i50', 'H_i55', 'H_i60', 'H_i65', 'H_i70', 'H_i75', 
                                     'H_i80', 'H_i85', 'H_i90', 'K_fSB', 'K_f0', 'K_fk', 'K_i00', 'K_i05', 'K_i10', 'K_i15', 'K_i20', 'K_i25', 'K_i30', 
                                     'K_i35', 'K_i40', 'K_i45', 'K_i50', 'K_i55', 'K_i60', 'K_i65', 'K_i70', 'K_i75', 'K_i80', 'K_i85', 'K_i90']
 # + "metallicity" once the csv has been created, we do not use Zini to be consistent with MIST data
        # quelles colonnes on veut : ['log10_isochrone_age_yr', 'log_Teff', 'log_g', 'phase', 'metallicity', 'star_mass', 'log_R']
        # quelles colonnes on a    : ['logAge', 'logTe' 'logg', 'label', 'metallicity', 'Mass', 'Rpol'] (metallicity dans le nom du fichier) (rayon n'est pas en log)
        # 0 = PMS, pre main sequence
        # 1 = MS, main sequence
        # 2 = SGB, subgiant branch, or Hertzsprung gap for more intermediate+massive stars
        # 3 = RGB, red giant branch, or the quick stage of red giant for intermediate+massive stars
        # 4 = CHEB, core He-burning for low mass stars, or the very initial stage of CHeB for intermediate+massive stars
        # 5 = still CHEB, the blueward part of the Cepheid loop of intermediate+massive stars
        # 6 = still CHEB, the redward part of the Cepheid loop of intermediate+massive stars
        # 7 = EAGB, the early asymptotic giant branch, or a quick stage of red giant for massive stars
        # 8 = TPAGB, the thermally pulsing asymptotic giant branch
        # 9 = post-AGB (in preparation!)
    
    def get_isochrone_dataframe(self, path : str=None, col_names : list[str]=None, override : bool=False) -> pd.DataFrame:
        """
        Uses the internal functions to give a dataframe containing either MIST or PARSEC data.
        To be usable, the PARSEC isochrone files' 15th line needs to be uncommented so that the column names can be extracted

        Parameters:
            path (str) : path to the directory containing the MIST isochrone files
            col_names (list of str) : names of the columns to be extracted. If set to an empty list, uses all the columns.
                Possible names for MIST are: log10_isochrone_age_yr, initial_mass, star_mass, star_mdot, he_core_mass, c_core_mass, log_L, log_LH, 
                                             log_LHe, log_Teff, log_R, log_g, surface_h1,surface_he3, surface_he4, surface_c12, surface_o16, 
                                             log_center_T, log_center_Rho, center_gamma, center_h1, center_he4, center_c12, phase, metallicity
                
                Possible names for PARSEC are: Zini, MH, logAge, Mini, int_IMF, Mass, logL, logTe, logg, label, McoreTP, C_O, period0, 
                                               period1, period2, period3, period4, pmode, Mloss, tau1m, X, Y, Xc, Xn, Xo, Cexcess, Z, 
                                               Teff0, omega, angvel, vtaneq, angmom, Rpol, Req, mbolmag, U_fSBmag, B_fSBmag, V_fSBmag, R_fSBmag, 
                                               I_fSBmag, J_fSBmag, H_fSBmag, K_fSBmag, U_fSB, U_f0, U_fk, U_i00, U_i05, U_i10, U_i15, U_i20, 
                                               U_i25, U_i30, U_i35, U_i40, U_i45, U_i50, U_i55, U_i60, U_i65, U_i70, U_i75, U_i80, U_i85, 
                                               U_i90, B_fSB, B_f0, B_fk, B_i00, B_i05, B_i10, B_i15, B_i20, B_i25, B_i30, B_i35, B_i40, 
                                               B_i45, B_i50, B_i55, B_i60, B_i65, B_i70, B_i75, B_i80, B_i85, B_i90, V_fSB, V_f0, V_fk, 
                                               V_i00, V_i05, V_i10, V_i15, V_i20, V_i25, V_i30, V_i35, V_i40, V_i45, V_i50, V_i55, V_i60, 
                                               V_i65, V_i70, V_i75, V_i80, V_i85, V_i90, R_fSB, R_f0, R_fk, R_i00, R_i05, R_i10, R_i15, 
                                               R_i20, R_i25, R_i30, R_i35, R_i40, R_i45, R_i50, R_i55, R_i60, R_i65, R_i70, R_i75, R_i80, 
                                               R_i85, R_i90, I_fSB, I_f0, I_fk, I_i00, I_i05, I_i10, I_i15, I_i20, I_i25, I_i30, I_i35, 
                                               I_i40, I_i45, I_i50, I_i55, I_i60, I_i65, I_i70, I_i75, I_i80, I_i85, I_i90, J_fSB, J_f0, 
                                               J_fk, J_i00, J_i05, J_i10, J_i15, J_i20, J_i25, J_i30, J_i35, J_i40, J_i45, J_i50, J_i55, 
                                               J_i60, J_i65, J_i70, J_i75, J_i80, J_i85, J_i90, H_fSB, H_f0, H_fk, H_i00, H_i05, H_i10, 
                                               H_i15, H_i20, H_i25, H_i30, H_i35, H_i40, H_i45, H_i50, H_i55, H_i60, H_i65, H_i70, H_i75, 
                                               H_i80, H_i85, H_i90, K_fSB, K_f0, K_fk, K_i00, K_i05, K_i10, K_i15, K_i20, K_i25, K_i30, 
                                               K_i35, K_i40, K_i45, K_i50, K_i55, K_i60, K_i65, K_i70, K_i75, K_i80, K_i85, K_i90

            override (bool): recomputes the dataframe and save it in a csv file if set to True. 
                             Otherwise, it only computes the dataframe if the file does not exist and returns the saved dataframe if it does.
        
        Returns:
            pandas.DataFrame : a pandas dataframe containing the data of the isochrones with the requested columns
        """
        if path is None:
            path = self.path
        if col_names is None:
            if len(self.col_names) == 0:
                col_names = self.all_MIST_col_names
            else:
                col_names = self.col_names

        if self.physical_model == "MIST":
            iso_df = self._full_MIST_data_to_panda(path, col_names, override)
        elif self.physical_model == "PARSEC":
            iso_df = self._full_PARSEC_data_to_panda(path, col_names, override)
        return iso_df
    
    # TODO mettre les deux fonctions en une vu que le seul changement c'est la façon dont on créé les données (dans une autre fct), le nom des fichiers et quelques variables
    def _full_PARSEC_data_to_panda(self, path : str, col_names : list[str], override : bool) -> pd.DataFrame:
        """
        Reads all PARSEC isochrone files in the given directory and creates a pandas dataframe of all the data with the requested columns.
        The dataframe, with all columns, is saved in the directory under the name "PARSEC_iso_full_data.csv".
        To be usable, the isochrone files' 15th line needs to be uncommented so that the column names can be extracted

        Parameters:
            path (str) : path to the directory containing the PARSEC isochrone files
            col_names (list of str) : names of the columns to be extracted. If set to an empty list, uses all the columns.
                Possible names are: log10_isochrone_age_yr, initial_mass, star_mass, star_mdot, he_core_mass, c_core_mass, log_L, log_LH, log_LHe, log_Teff, log_R, log_g, surface_h1,
                                    surface_he3, surface_he4, surface_c12, surface_o16, log_center_T, log_center_Rho, center_gamma, center_h1, center_he4, center_c12, phase
            override (bool) : recomputes the dataframe and save it in a csv file if set to True. 
                              Otherwise, it only computes the dataframe if the file does not exist and returns the saved dataframe if it does.
                                    
        Returns:
            pandas.DataFrame : a pandas dataframe containing the data of the isochrones with the requested columns
        """
        if override | (not os.path.exists(path + "PARSEC_iso_full_data.csv")):
            # creates a dictionary containing an empty list for each given column 
            col_dict = {key: [] for key in self.all_PARSEC_col_names}
            col_dict["metallicity"] = []
            full_iso_df = pd.DataFrame.from_dict(col_dict)
            
            # reads all the .iso files in the directory and appends their data to the dataframe
            for filename in os.listdir(path):
                if filename.endswith(".dat"):
                    iso_df = self._PARSEC_data_to_panda(path + filename, self.all_PARSEC_col_names)
                    full_iso_df = pd.concat([full_iso_df, iso_df], ignore_index=True)
            
            # saves the dataframe in a csv file
            print("Writing PARSEC dataframe to csv file...")
            full_iso_df.to_csv(path + "PARSEC_iso_full_data.csv", sep=',', encoding='utf-8', index=False, header=True)
        else:
            print("Reading PARSEC dataframe from csv file...")
            full_iso_df = pd.read_csv(path + "PARSEC_iso_full_data.csv")
        
        return full_iso_df[col_names]
    
    def _PARSEC_data_to_panda(self, path : str, col_names : list[str]) -> pd.DataFrame:
        """
        Reads a PARSEC isochrone file and returns a pandas dataframe with the requested columns.
        To be usable, the isochrone files' 15th line needs to be uncommented so that the column names can be extracted

        Parameters:
            path (str) : path to the MIST isochrone file
            col_names (list of str) : names of the columns to be extracted.
                Possible names are: log10_isochrone_age_yr, initial_mass, star_mass, star_mdot, he_core_mass, c_core_mass, log_L, log_LH, log_LHe, log_Teff, log_R, log_g, surface_h1,
                                    surface_he3, surface_he4, surface_c12, surface_o16, log_center_T, log_center_Rho, center_gamma, center_h1, center_he4, center_c12, phase
        
        Returns:
            pandas.DataFrame : a pandas dataframe containing the data of the isochrones with the requested columns
        """
        print(f"Reading file '{path}'")
        parsec_table = Table.read(path, format='ascii')

        # creates a dictionary containing an empty list for each given column 
        col_dict = {key: [] for key in col_names}
        filename = os.path.basename(os.path.normpath(path))
        # the files contain either "m" + number or "p" + number in their name
        metallicity = float(filename[12:16]) if filename[11] == "p" else -float(filename[12:16])

        for col_name in parsec_table.colnames:
            col_dict[col_name].extend(parsec_table[col_name])
        
        iso_df = pd.DataFrame.from_dict(col_dict)
        iso_df["metallicity"] = metallicity

        return iso_df


    
    def _full_MIST_data_to_panda(self, path : str, col_names : list[str], override : bool) -> pd.DataFrame:
        """
        Reads all MIST isochrones files in the given directory and creates a pandas dataframe of all the data with the requested columns.
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
        if override | (not os.path.exists(path + "MIST_iso_full_data.csv")):
            # creates a dictionary containing an empty list for each given column 
            col_dict = {key: [] for key in self.all_MIST_col_names}
            col_dict["metallicity"] = []
            full_iso_df = pd.DataFrame.from_dict(col_dict)
            
            # reads all the .iso files in the directory and appends their data to the dataframe
            for filename in os.listdir(path):
                if filename.endswith(".iso"):
                    iso_df = self._MIST_data_to_panda(path + filename, self.all_MIST_col_names)
                    full_iso_df = pd.concat([full_iso_df, iso_df], ignore_index=True)
            
            # saves the dataframe in a csv file
            print("Writing MIST dataframe to csv file...")
            full_iso_df.to_csv(path + "MIST_iso_full_data.csv", sep=',', encoding='utf-8', index=False, header=True)

        else:
            print("Reading MIST dataframe from csv file...")
            full_iso_df = pd.read_csv(path + "MIST_iso_full_data.csv")
        
        return full_iso_df[col_names]

    def _MIST_data_to_panda(self, path : str, col_names : list[str]) -> pd.DataFrame:
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

    # test_class = Iso_data_handler("C:/Users/antoi/Code/unif/MA2/thesis/data/PARSEC/",
    #                               ['logAge', 'logTe', 'logg', 'label', 'metallicity', 'Mass', 'Rpol'],
    #                               "PARSEC")
    # test_df = test_class._PARSEC_data_to_panda("C:/Users/antoi/Code/unif/MA2/thesis/data/PARSEC/PARSEC_feh_m0.25.dat", ['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 'label', 'McoreTP', 'C_O', 'period0', 
    #                                  'period1', 'period2', 'period3', 'period4', 'pmode', 'Mloss', 'tau1m', 'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess', 'Z', 
    #                                  'Teff0', 'omega', 'angvel', 'vtaneq', 'angmom', 'Rpol', 'Req', 'mbolmag', 'U_fSBmag', 'B_fSBmag', 'V_fSBmag', 'R_fSBmag', 
    #                                  'I_fSBmag', 'J_fSBmag', 'H_fSBmag', 'K_fSBmag', 'U_fSB', 'U_f0', 'U_fk', 'U_i00', 'U_i05', 'U_i10', 'U_i15', 'U_i20', 
    #                                  'U_i25', 'U_i30', 'U_i35', 'U_i40', 'U_i45', 'U_i50', 'U_i55', 'U_i60', 'U_i65', 'U_i70', 'U_i75', 'U_i80', 'U_i85', 
    #                                  'U_i90', 'B_fSB', 'B_f0', 'B_fk', 'B_i00', 'B_i05', 'B_i10', 'B_i15', 'B_i20', 'B_i25', 'B_i30', 'B_i35', 'B_i40', 
    #                                  'B_i45', 'B_i50', 'B_i55', 'B_i60', 'B_i65', 'B_i70', 'B_i75', 'B_i80', 'B_i85', 'B_i90', 'V_fSB', 'V_f0', 'V_fk', 
    #                                  'V_i00', 'V_i05', 'V_i10', 'V_i15', 'V_i20', 'V_i25', 'V_i30', 'V_i35', 'V_i40', 'V_i45', 'V_i50', 'V_i55', 'V_i60', 
    #                                  'V_i65', 'V_i70', 'V_i75', 'V_i80', 'V_i85', 'V_i90', 'R_fSB', 'R_f0', 'R_fk', 'R_i00', 'R_i05', 'R_i10', 'R_i15', 
    #                                  'R_i20', 'R_i25', 'R_i30', 'R_i35', 'R_i40', 'R_i45', 'R_i50', 'R_i55', 'R_i60', 'R_i65', 'R_i70', 'R_i75', 'R_i80', 
    #                                  'R_i85', 'R_i90', 'I_fSB', 'I_f0', 'I_fk', 'I_i00', 'I_i05', 'I_i10', 'I_i15', 'I_i20', 'I_i25', 'I_i30', 'I_i35', 
    #                                  'I_i40', 'I_i45', 'I_i50', 'I_i55', 'I_i60', 'I_i65', 'I_i70', 'I_i75', 'I_i80', 'I_i85', 'I_i90', 'J_fSB', 'J_f0', 
    #                                  'J_fk', 'J_i00', 'J_i05', 'J_i10', 'J_i15', 'J_i20', 'J_i25', 'J_i30', 'J_i35', 'J_i40', 'J_i45', 'J_i50', 'J_i55', 
    #                                  'J_i60', 'J_i65', 'J_i70', 'J_i75', 'J_i80', 'J_i85', 'J_i90', 'H_fSB', 'H_f0', 'H_fk', 'H_i00', 'H_i05', 'H_i10', 
    #                                  'H_i15', 'H_i20', 'H_i25', 'H_i30', 'H_i35', 'H_i40', 'H_i45', 'H_i50', 'H_i55', 'H_i60', 'H_i65', 'H_i70', 'H_i75', 
    #                                  'H_i80', 'H_i85', 'H_i90', 'K_fSB', 'K_f0', 'K_fk', 'K_i00', 'K_i05', 'K_i10', 'K_i15', 'K_i20', 'K_i25', 'K_i30', 
    #                                  'K_i35', 'K_i40', 'K_i45', 'K_i50', 'K_i55', 'K_i60', 'K_i65', 'K_i70', 'K_i75', 'K_i80', 'K_i85', 'K_i90'])

    # test_df = test_class.get_isochrone_dataframe(override=False)
    # print(test_df)
