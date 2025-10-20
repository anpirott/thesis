import numpy as np
import scipy.stats as stats
import astropy as ap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import sys
import os

from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error,\
                            root_mean_squared_error, median_absolute_error
from scipy.stats import pearsonr

import read_mist_models


 # TODO? avoir des méthodes qui peuvent dire dans quelle isochrone un point de donnée est, 
 # TODO? des plots(?), une meilleur façon de garder les noms des colonnes pour les choisir, de quoi regrouper les données en un set, etc
# TODO rajouter les exceptions pour les erreurs
class Iso_data_handler():
    def __init__(self, directory : str, col_names : list[str]):
        """
        Initializes the Iso_data_handler class.

        Parameters:
            directory (str) : complete path to the directory containing the MIST isochrone files
            col_names (list of str) : names of the columns to be extracted. If set to an empty list, uses all the columns.
                Possible names are: log10_isochrone_age_yr, initial_mass, star_mass, star_mdot, he_core_mass, c_core_mass, log_L, log_LH, log_LHe, log_Teff, log_R, log_g, surface_h1,
                                    surface_he3, surface_he4, surface_c12, surface_o16, log_center_T, log_center_Rho, center_gamma, center_h1, center_he4, center_c12, phase
        """

        if not isinstance(directory, str):
            print("Error: directory should be initialized as a string.")
            sys.exit(1)
        if "\\" in directory:
            directory = directory.replace("\\", "/")
        if not directory.endswith("/"):
            directory += "/"
        if not os.path.exists(directory):
            print("Error: directory does not exist.")
            sys.exit(1)
        self.directory = directory

        if not isinstance(col_names, list):
            print("Error: col_names should be initialized as a list of strings.")
            sys.exit(1)
        self.col_names = col_names

        self.all_col_names = ["log10_isochrone_age_yr", "initial_mass", "star_mass", "star_mdot", "he_core_mass", "c_core_mass", "log_L", "log_LH", 
                              "log_LHe", "log_Teff", "log_R", "log_g", "surface_h1", "surface_he3", "surface_he4", "surface_c12", "surface_o16", 
                              "log_center_T", "log_center_Rho", "center_gamma", "center_h1", "center_he4", "center_c12", "phase", "metallicity"]
    
    def full_iso_data_to_panda(self, directory : str=None, col_names : list[str]=None, override : bool=False) -> pd.DataFrame:
        """
        Reads all .iso files in the given directory and creates a pandas dataframe of all the data with the requested columns.
        The dataframe, with all columns, is saved in the directory under the name "MIST_iso_full_data.csv".

        Parameters:
            directory (str) : path to the directory containing the MIST isochrone files
            col_names (list of str) : names of the columns to be extracted. If set to an empty list, uses all the columns.
                Possible names are: log10_isochrone_age_yr, initial_mass, star_mass, star_mdot, he_core_mass, c_core_mass, log_L, log_LH, log_LHe, log_Teff, log_R, log_g, surface_h1,
                                    surface_he3, surface_he4, surface_c12, surface_o16, log_center_T, log_center_Rho, center_gamma, center_h1, center_he4, center_c12, phase
            override (bool) : recomputes the dataframe and save it in a csv file if set to True. 
                              Otherwise, it only computes the dataframe if the file does not exist and returns the saved dataframe if it does.
                                    
        Returns:
            pandas.DataFrame : a pandas dataframe containing the data of the isochrones with the requested columns
        """
        if directory is None:
            directory = self.directory
        if col_names is None:
            if len(self.col_names) == 0:
                col_names = self.all_col_names
            else:
                col_names = self.col_names
        
        if override | (not os.path.exists(directory + "MIST_iso_full_data.csv")):
            # creates a dictionary containing an empty list for each given column 
            col_dict = {key: [] for key in self.all_col_names}
            col_dict["metallicity"] = []
            full_iso_df = pd.DataFrame.from_dict(col_dict)
            
            # reads all the .iso files in the directory and appends their data to the dataframe
            for filename in os.listdir(directory):
                if filename.endswith(".iso"):
                    iso_df = Iso_data_handler.iso_data_to_panda(directory + filename, self.all_col_names)
                    full_iso_df = pd.concat([full_iso_df, iso_df], ignore_index=True)
            
            # saves the dataframe in a csv file
            print("Writing dataframe to csv file...")
            full_iso_df.to_csv(directory + "MIST_iso_full_data.csv", sep=',', encoding='utf-8', index=False, header=True)

        else:
            print("Reading dataframe from csv file...")
            full_iso_df = pd.read_csv(directory + "MIST_iso_full_data.csv")
        
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
        filename = os.path.basename(os.path.normpath(path))
        # the files contain either "m" + number or "p" + number in their name
        metallicity = float(filename[15:19]) if filename[14] == "p" else -float(filename[15:19])

        for iso_ind in range(len(iso.isos)):
            for keys in col_dict.keys():
                col_dict[keys].extend(iso.isos[iso_ind][keys])
        
        iso_df = pd.DataFrame.from_dict(col_dict)
        iso_df["metallicity"] = metallicity

        return iso_df


# TODO rajouter les exceptions pour les erreurs
class Model_evaluator():
    def __init__(self, model_name : str, model : str=None, truth : np.ndarray=None, preds : np.ndarray=None, path : str=None, rve : bool=True, 
                 rmse : bool=True, mae : bool=True, medae : bool=True, corr : bool=True, maxe : bool=True, percentile : list[int]=(75, 90, 95, 99), 
                 predicted_truth : bool=True, residuals_plot : bool=True, error_boxplot : bool=True, error_histogram : bool=True, qq_plot : bool=True): # TODO rajouter save ici? rajouter path pour ce qu'on sauvegarde?
        """
        Initializes the Model_evaluator class.

        Parameters:
            model_name (str) : name of the model being evaluated
            model (str) : path to the trained machine learning model
            truth (numpy.ndarray) : true values
            preds (numpy.ndarray) : predicted values
            every bool variable : whether to compute/plot the corresponding metric
            percentile (list of int) : list of percentiles to compute
        """
        self.model_name = model_name
        self.model = model
        self.truth = truth
        self.preds = preds
        self.path = path

        self.rve = rve
        self.rmse = rmse
        self.mae = mae
        self.medae = medae
        self.corr = corr
        self.maxe = maxe
        self.percentile = percentile
        self.predicted_truth = predicted_truth
        self.residuals_plot = residuals_plot
        self.error_boxplot = error_boxplot
        self.error_histogram = error_histogram
        self.qq_plot = qq_plot

        self.metrics_dict = dict()
    
    # TODO fct qui set tout à faux, fct qui set tout à vrai, fct pour set chaque valeur individuellement

    # TODO? faire une fonction qui calcule les metriques et une qui les print
    def print_model_metrics(self, parameter_name : str, truth : np.ndarray=None, preds : np.ndarray=None, path : str=None): # TODO? enlever save
        """
        Prints various metrics for the given model predictions.

        Parameters:
            parameter_name (str) : name of the parameter being evaluated
            truth (numpy.ndarray) : true values
            preds (numpy.ndarray) : predicted values
            path (str) : the path to save the metrics, if set to None, does not save
        """
        if truth is None:
            if self.truth is None:
                print("Error: truth values not provided.")
                sys.exit(1)
            truth = self.truth
        if preds is None:
            if self.preds is None:
                print("Error: predicted values not provided.")
                sys.exit(1)
            preds = self.preds

        self.metrics_dict[parameter_name] = dict() # TODO? p-ê gérer si ça existe déjà
        
        print()
        print(f"{parameter_name} results:")
        if self.rve:
            self.metrics_dict[parameter_name]['RVE'] = explained_variance_score(truth, preds)
            print("Explained variance (RVE): ",explained_variance_score(truth, preds))
        if self.rmse:
            self.metrics_dict[parameter_name]['RMSE'] = root_mean_squared_error(truth, preds)
            print("Root mean squared error (RMSE): ",root_mean_squared_error(truth, preds))
        if self.mae:
            self.metrics_dict[parameter_name]['MAE'] = mean_absolute_error(truth, preds)
            print("Mean absolute error (MAE): ",mean_absolute_error(truth, preds))  
        if self.medae:
            self.metrics_dict[parameter_name]['MedAE'] = median_absolute_error(truth, preds)
            print("Median absolute error (MedAE): ",median_absolute_error(truth, preds))
        if corr:
            self.metrics_dict[parameter_name]['CORR'], _ = pearsonr(truth, preds)
            corr, pval=pearsonr(truth, preds)
            print("Pearson's correlation coefficient (CORR): ",corr)
        if self.maxe:
            self.metrics_dict[parameter_name]['MAX_ER'] = max_error(truth, preds)
            print("Maximum error (MAX_ER): ",max_error(truth, preds))
        if isinstance(self.percentile, tuple) and len(self.percentile) > 0:
            self.metrics_dict[parameter_name]['Percentiles'] = dict()
            absolute_residuals = np.abs(truth - preds)
            for p in self.percentile:
                if p >= 0 and p <= 100:
                    self.metrics_dict[parameter_name]['Percentiles'][p] = np.percentile(absolute_residuals, p)
                    print(f"{p}th percentile : ", np.percentile(absolute_residuals, p))
        

    def plot_model_metrics(self, parameter_name : str, truth : np.ndarray=None, preds : np.ndarray=None, path : str=None): # TODO? enlever save
        """
        Plots various metrics for the given model predictions.

        Parameters:
            parameter_name (str) : name of the parameter being evaluated
            truth (numpy.ndarray) : true values
            preds (numpy.ndarray) : predicted values
            path (str) : the path to save the metrics, if set to None, does not save
        """
        residuals = truth - preds
        absolute_residuals = np.abs(truth - preds)

        if self.predicted_truth: # TODO rajouter dans le mémoire
            # plot showing the predicted vs true values : the x axis is the true values, the y axis is the predicted values
            # a point on the line means that the difference between the prediction and the truth is 0, meaning the prediction is perfect
            # thus the red line represents perfect predictions
            # the further away from the line, the larger the error
            # a point above the line means an overestimation, a point below means an underestimation
            plt.figure(figsize=(6,6))
            plt.scatter(truth, preds, alpha=0.5)
            plt.plot([min(truth), max(truth)], [min(truth), max(truth)], color='red', linestyle='--')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Predicted vs True Values for {parameter_name}')
            plt.show()
        if self.residuals_plot: # TODO rajouter dans le mémoire si je le garde
            # similaire à celui du dessus, le garde?
            plt.figure(figsize=(6,6))
            plt.scatter(truth, residuals, alpha=0.5)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('True Values')
            plt.ylabel('Residuals')
            plt.title(f'Residuals vs Predicted Values for {parameter_name}')
            plt.show()
        if self.error_boxplot: # TODO rajouter dans le mémoire si je le garde
            plt.figure(figsize=(6,6))
            sns.boxplot(y=residuals) # without log scale
            plt.ylabel('Residuals')
            plt.title(f'Box Plot of Residuals for {parameter_name}')
            plt.show()
        if self.error_boxplot: # TODO rajouter dans le mémoire si je le garde
            plt.figure(figsize=(6,6))
            sns.boxplot(y=absolute_residuals, log_scale=True)
            plt.ylabel('Residuals')
            plt.title(f'Box Plot of Residuals for {parameter_name}')
            plt.show()
        if self.error_histogram: # TODO rajouter dans le mémoire si je le garde
            plt.figure(figsize=(6,6))
            plt.hist(residuals, bins=60)
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Residuals for {parameter_name}')
            plt.show()
        if self.qq_plot: # TODO rajouter dans le mémoire si je le garde
            plt.figure(figsize=(6,6))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('Normal Q-Q plot')
            plt.xlabel('Theoretical quantiles')
            plt.ylabel('Ordered Values')
            plt.grid(True)
            plt.show()
    
    def save_model_metrics(self, truth : np.ndarray=None, preds : np.ndarray=None, path : str=None): # TODO tester le save
        """
        Saves various metrics for the given model predictions to a CSV file.

        Parameters:
            truth (numpy.ndarray) : true values
            preds (numpy.ndarray) : predicted values
            path (str) : the path to save the metrics, if set to None, uses the path provided during initialization
        """
        if truth is None:
            if self.truth is None:
                print("Error: truth values not provided.")
                sys.exit(1)
            truth = self.truth
        if preds is None:
            if self.preds is None:
                print("Error: predicted values not provided.")
                sys.exit(1)
            preds = self.preds
        if path is None:
            if self.path is None:
                print("Error: path not provided.")
                sys.exit(1)
            path = self.path
        
        metrics_df = pd.DataFrame.from_dict(self.metrics_dict, orient='index')
        metrics_df.to_csv(path + f"{self.model_name}_metrics.csv", sep=',', encoding='utf-8', index=True, header=True)


    def evaluate_model(self, model, X_test : pd.DataFrame, y_test : pd.DataFrame):
        """
        Evaluates the given model on the test data and prints the metrics.

        Parameters:
            model : trained machine learning model
            X_test (pandas.DataFrame) : test features
            y_test (pandas.DataFrame) : test targets
        """
        y_pred = model.predict(X_test)
        for i, col in enumerate(y_test.columns):
            self.print_model_metrics(y_test[col].values, y_pred[:, i], col)



if __name__ == "__main__":
    # test_class = Iso_data_handler("C:/Users/antoi/Code/unif/MA2/Thèse/data/MIST_v1.2_vvcrit0.0_basic_isos/",
    #                               ['log10_isochrone_age_yr', 'log_Teff', 'log_g', 'star_mass', 'phase'])
    # test_df = test_class.full_iso_data_to_panda(override=False)
    # print(test_df)
    pass

