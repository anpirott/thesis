import numpy as np
import scipy.stats as stats
import astropy as ap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import sys
import os
from collections.abc import Callable

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error,\
                            root_mean_squared_error, median_absolute_error
from sklearn.decomposition import PCA

from scipy.stats import pearsonr

import read_mist_models


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
        # path sanitization 
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
        self.path = path

        if not isinstance(col_names, list):
            print("Error: col_names should be initialized as a list of strings.")
            sys.exit(1)
        self.col_names = col_names

        self.all_col_names = ["log10_isochrone_age_yr", "initial_mass", "star_mass", "star_mdot", "he_core_mass", "c_core_mass", "log_L", "log_LH", 
                              "log_LHe", "log_Teff", "log_R", "log_g", "surface_h1", "surface_he3", "surface_he4", "surface_c12", "surface_o16", 
                              "log_center_T", "log_center_Rho", "center_gamma", "center_h1", "center_he4", "center_c12", "phase", "metallicity"]
    
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
class Data_preparator():
    """
    Class which contains methods for preparing data for machine learning models.

    Methods:
        filter_data : filters the given dataframe according to the given filter dictionary
        split_data : splits the given dataframe into training and testing sets
        pca_preparation : applies PCA to the given training and IVS data
    """
    @staticmethod
    def filter_data(data_df : pd.DataFrame, filter_dict : dict) -> pd.DataFrame:
        """
        Filters the given dataframe according to the given filter dictionary.

        Parameters:
            data_df (pandas.DataFrame) : dataframe to be filtered
            filter_dict (dict) : dictionary containing the filter conditions. The keys are the column names and the values are the filter values.
                The filter values can either be a list of values or a tuple of size two. If it is a tuple, the first element is the operator ("<", "<=", ">", ">=", "==", "!=") and the second element is the value to compare to.
                For example, to filter the dataframe to only include rows where the phase is 0, 2, 3, 4 or 5 and a mass smaller than 30, the filter_dict would be:
                    filter_dict = {"phase": [0, 2, 3, 4, 5], "mass": ("<", 30)}
        
        Returns:
            pandas.DataFrame : a pandas dataframe containing the filtered data
        """
        for key in filter_dict.keys():
            if isinstance(filter_dict[key], tuple) and len(filter_dict[key]) == 2:
                operator, value = filter_dict[key]
                if operator == "<":
                    data_df = data_df[data_df[key] < value].dropna().reset_index(drop=True)
                elif operator == "<=":
                    data_df = data_df[data_df[key] <= value].dropna().reset_index(drop=True)
                elif operator == ">":
                    data_df = data_df[data_df[key] > value].dropna().reset_index(drop=True)
                elif operator == ">=":
                    data_df = data_df[data_df[key] >= value].dropna().reset_index(drop=True)
                elif operator == "==":
                    data_df = data_df[data_df[key] == value].dropna().reset_index(drop=True)
                elif operator == "!=":
                    data_df = data_df[data_df[key] != value].dropna().reset_index(drop=True)
                else:
                    print(f"Error: invalid operator '{operator}' in filter_dict for key '{key}'.")
                    sys.exit(1)
            elif isinstance(filter_dict[key], list):
                data_df = data_df[data_df[key].isin(filter_dict[key])].dropna().reset_index(drop=True)
            else:
                print(f"Error: invalid filter value '{filter_dict[key]}' in filter_dict for key '{key}'.")
                sys.exit(1)
        return data_df

    @staticmethod
    def split_data(data_df : pd.DataFrame, x_cols : list, y_cols : list, test_size : float=0.25, shuffle : bool=True, random_state : int=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the given dataframe into training and testing sets.

        Parameters:
            data_df (pandas.DataFrame) : dataframe to be split
            x_cols (list) : list of column names to be used as features
            y_cols (list) : list of column names to be used as targets
            test_size (float) : proportion of the data to be used as test set
            shuffle (bool) : whether to shuffle the data before splitting
            random_state (int) : random seed for shuffling the data
        
        Returns:
            tuple of four numpy.ndarrays : the training features (X_train), testing features (X_test), training targets (y_train) and testing targets (y_test)
        """
        X = data_df[x_cols].to_numpy()
        y = data_df[y_cols].to_numpy()

        return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    
    def show_data_stats(): # TODO? p-ê pas utiles
        pass
    
    @staticmethod
    def pca_preparation(X_train : np.ndarray, X_test : np.ndarray, verbose : bool=False) -> tuple[np.ndarray, np.ndarray]:
        """
        applies PCA to the given training and test data.

        Parameters:
            X_train (numpy.ndarray) : training data
            X_test (numpy.ndarray) : test data
            verbose (bool) : whether to print the explained variance ratio of each principal component and the shape of the transformed data
        
        Returns:
            tuple of two numpy.ndarrays : the transformed training and test data
        """
        pca = PCA(n_components=4) # maybe try with less or more components
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_ivs_pca = pca.transform(X_test)

        if verbose:
            tve=0
            for i, ve in enumerate(pca.explained_variance_ratio_):
                tve+=ve
                print("PC%d - Variance explained: %7.4f - Total Variance: %7.4f" % (i, ve, tve))
            print()
            print(X_train_pca.shape)

        return X_train_pca, X_ivs_pca

# TODO rajouter les exceptions pour les erreurs
class Model_trainer():
    """
    Class which contains methods for training machine learning models.

    Methods:
        Kfold_pipeline : performs K-fold cross-validation on the given model and training data
    """
    # TODO faire que la fonction puisse accepter autant d'output qu'on veut
    # TODO mettre les outputs (preds) du modèle dans un fichier csv pour utiliser plus tard
    # TODO mettre la possibilité de rajouter des paramètres à tester dans le modèle
    # TODO rajouter le calcul du temps et le rajouter dans le csv
    @staticmethod
    def Kfold_pipeline(model : Callable, x_train_data : np.ndarray, y_train_data : np.ndarray, n_splits : int=10, 
                       shuffle : bool=True, random_state : int=120) -> tuple[list, list]:
        """
        Performs K-fold cross-validation on the given model and training data.

        Parameters:
            model (Callable) : machine learning model to be trained
            x_train_data (numpy.ndarray) : training features
            y_train_data (numpy.ndarray) : training targets
            n_splits (int) : number of folds for cross-validation
            shuffle (bool) : whether to shuffle the data before splitting into folds
            random_state (int) : random seed for shuffling the data
        
        Returns:
            tuple of two lists : a tuple containing the true values for each target across all folds and a tuple containing the predicted values for each target across all folds
        """
        truth = list(None for _ in range(y_train_data.shape[1]))
        preds = list(None for _ in range(y_train_data.shape[1]))

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        counter = 0
        print("split", end=' ')
        for train_index, test_index in kf.split(x_train_data):
            counter += 1
            print(str(counter), end=' ')
            X_train, X_test = x_train_data[train_index], x_train_data[test_index]
            y_train, y_test = y_train_data[train_index], y_train_data[test_index]

            mdl = model()
            mdl.fit(X_train, y_train)
            fold_preds = mdl.predict(X_test)

            for i in range(y_train_data.shape[1]):
                if truth[i] is None:
                    truth[i] = y_test[:, i]
                    preds[i] = fold_preds[:, i]
                else:
                    truth[i] = np.hstack((truth[i], y_test[:, i]))
                    preds[i] = np.hstack((preds[i], fold_preds[:, i]))
        
        return truth, preds


# TODO rajouter les exceptions pour les erreurs
class Model_evaluator():
    """
    Class for evaluating machine learning models by calculating various metrics and generating plots.

    Methods:
        __init__ : initializes the Model_evaluator class
        set_metrics_values : sets the metrics to be computed/plots to be generated
        calculate_model_evaluation : calculates various metrics and plots for the given model predictions
        show_model_evaluation : shows various metrics and plots for the given model predictions
        save_model_evaluation : saves various metrics and plots for the given model predictions to a CSV file and images in a directory
    """
    def __init__(self, model_name : str, model : str=None, truth : np.ndarray=None, preds : np.ndarray=None, path : str=None, rve : bool=True, 
                 rmse : bool=True, mae : bool=True, medae : bool=True, corr : bool=True, maxe : bool=True, percentile : list[int]=(75, 90, 95, 99), 
                 predicted_truth_plot : bool=True, residuals_truth_plot : bool=True, residuals_boxplot : bool=True, residuals_histogram : bool=True, qq_plot : bool=True): # TODO rajouter save ici? rajouter path pour ce qu'on sauvegarde?
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
        # path sanitization 
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
        self.path = path

        self.rve = rve
        self.rmse = rmse
        self.mae = mae
        self.medae = medae
        self.corr = corr
        self.maxe = maxe
        self.percentile = percentile
        self.predicted_truth_plot = predicted_truth_plot
        self.residuals_truth_plot = residuals_truth_plot
        self.residuals_boxplot = residuals_boxplot
        self.residuals_histogram = residuals_histogram
        self.qq_plot = qq_plot

        self.metrics_dict = dict()
        self.plot_dict = dict()
    
    def set_metrics_values(self, all=True, rve=None, rmse=None, mae=None, medae=None, corr=None, maxe=None, percentile=None,
                    predicted_truth_plot=None, residuals_truth_plot=None, residuals_boxplot=None, residuals_histogram=None, qq_plot=None):
        """
        Sets the metrics to be computed/plots to be generated.

        Parameters:
            all (bool) : if set to True, sets all metrics/plots to True
            every other parameter (bool or list of int) : if not set to None, sets the corresponding metric/plot to the given value
        """
        if all:
            self.rve = True
            self.rmse = True
            self.mae = True
            self.medae = True
            self.corr = True
            self.maxe = True
            self.percentile = (75, 90, 95, 99)
            self.predicted_truth_plot = True
            self.residuals_truth_plot = True
            self.residuals_boxplot = True
            self.residuals_histogram = True
            self.qq_plot = True
            return
        
        if rve is not None:
            self.rve = rve
        if rmse is not None:
            self.rmse = rmse
        if mae is not None:
            self.mae = mae
        if medae is not None:
            self.medae = medae
        if corr is not None:
            self.corr = corr
        if maxe is not None:
            self.maxe = maxe
        if percentile is not None:
            self.percentile = percentile
        if predicted_truth_plot is not None:
            self.predicted_truth_plot = predicted_truth_plot
        if residuals_truth_plot is not None:
            self.residuals_truth_plot = residuals_truth_plot
        if residuals_boxplot is not None:
            self.residuals_boxplot = residuals_boxplot
        if residuals_histogram is not None:
            self.residuals_histogram = residuals_histogram
        if qq_plot is not None:
            self.qq_plot = qq_plot

    def calculate_model_evaluation(self, parameter_name : str, truth : np.ndarray=None, preds : np.ndarray=None) -> tuple[dict, dict]:
        """
        Calculates various metrics and plots for the given model predictions.

        Parameters:
            parameter_name (str) : name of the parameter being evaluated
            truth (numpy.ndarray) : true values
            preds (numpy.ndarray) : predicted values
        
        Returns:
            tuple of two dicts : a dictionary containing the calculated metrics and a dictionary containing the generated plots
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

        if self.rve:
            self.metrics_dict[parameter_name]['RVE'] = explained_variance_score(truth, preds)
        if self.rmse:
            self.metrics_dict[parameter_name]['RMSE'] = root_mean_squared_error(truth, preds)
        if self.mae:
            self.metrics_dict[parameter_name]['MAE'] = mean_absolute_error(truth, preds)
        if self.medae:
            self.metrics_dict[parameter_name]['MedAE'] = median_absolute_error(truth, preds)
        if self.corr:
            self.metrics_dict[parameter_name]['CORR'], _ = pearsonr(truth, preds)
        if self.maxe:
            self.metrics_dict[parameter_name]['MAX_ER'] = max_error(truth, preds)
        if isinstance(self.percentile, tuple) and len(self.percentile) > 0:
            self.metrics_dict[parameter_name]['Percentiles'] = dict()
            absolute_residuals = np.abs(truth - preds)
            for p in self.percentile:
                if p >= 0 and p <= 100:
                    self.metrics_dict[parameter_name]['Percentiles'][p] = np.percentile(absolute_residuals, p)
        
        residuals = truth - preds
        absolute_residuals = np.abs(truth - preds)

        self.plot_dict[parameter_name] = dict() # TODO? gérer si ça existe déjà

        if self.predicted_truth_plot: # TODO rajouter dans le mémoire
            # plot showing the predicted vs true values : the x axis is the true values, the y axis is the predicted values
            # a point on the line means that the difference between the prediction and the truth is 0, meaning the prediction is perfect
            # thus the red line represents perfect predictions
            # the further away from the line, the larger the error
            # a point above the line means an overestimation, a point below means an underestimation

            plotted_predicted_truth = plt.figure(figsize=(6,6))
            plt.scatter(truth, preds, alpha=0.01)
            plt.plot([min(truth), max(truth)], [min(truth), max(truth)], color='red', linestyle='--')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Predicted vs True Values for {parameter_name}')
            self.plot_dict[parameter_name]['predicted_truth_plot'] = plotted_predicted_truth

            # self.plot_dict[parameter_name]['predicted_truth_plot'] = \
            #     self._test(xlabel='True Values', ylabel='Predicted Values', title=f'Predicted vs True Values for {parameter_name}', grid=False,
            #                funcs=[plt.scatter(truth, preds, alpha=0.5), plt.plot([min(truth), max(truth)], [min(truth), max(truth)], color='red', linestyle='--')])
        if self.residuals_truth_plot: # TODO rajouter dans le mémoire si je le garde
            # similaire à celui du dessus, le garde?
            plotted_residuals_truth = plt.figure(figsize=(6,6))
            plt.scatter(truth, residuals, alpha=0.01)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('True Values')
            plt.ylabel('Residuals')
            plt.title(f'Residuals vs Predicted Values for {parameter_name}')
            self.plot_dict[parameter_name]['residuals_truth_plot'] = plotted_residuals_truth
        if self.residuals_boxplot: # TODO rajouter dans le mémoire si je le garde
            plotted_boxplot_no_log = plt.figure(figsize=(6,6))
            sns.boxplot(y=residuals) # without log scale
            plt.ylabel('Residuals')
            plt.title(f'Box Plot of Residuals for {parameter_name}')
            self.plot_dict[parameter_name]['residuals_boxplot_no_log'] = plotted_boxplot_no_log
        if self.residuals_boxplot: # TODO rajouter dans le mémoire si je le garde
            plotted_boxplot_log = plt.figure(figsize=(6,6))
            sns.boxplot(y=absolute_residuals, log_scale=True) # TODO p-ê une erreur, ou alors juste à cause des valeurs du test
            plt.ylabel('Residuals')
            plt.title(f'Box Plot of Residuals for {parameter_name}')
            self.plot_dict[parameter_name]['residuals_boxplot_log'] = plotted_boxplot_log
        if self.residuals_histogram: # TODO rajouter dans le mémoire si je le garde
            plotted_histogram = plt.figure(figsize=(6,6))
            plt.hist(residuals, bins=60)
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Residuals for {parameter_name}')
            self.plot_dict[parameter_name]['residuals_histogram_plot'] = plotted_histogram
        if self.qq_plot: # TODO rajouter dans le mémoire si je le garde
            plotted_qq = plt.figure(figsize=(6,6))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.xlabel('Theoretical quantiles')
            plt.ylabel('Ordered Values')
            plt.title('Normal Q-Q plot')
            plt.grid(True)
            self.plot_dict[parameter_name]['qq_plot'] = plotted_qq
        
        return self.metrics_dict[parameter_name], self.plot_dict[parameter_name]


    # def _test_fonctionne_pas(self, xlabel : str, ylabel : str, title : str, grid : bool, funcs : list[Callable]) -> plt.Figure:
    #     plot = plt.figure(figsize=(6,6))
    #     for func in funcs:
    #         func
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.title(title)
    #     plt.grid(grid)
    #     return plot

    # TODO? ajouter une facon d'avoir toutes les metrics d'un coup, pas paramètres par paramètres, pour pouvoir voir en même temps (pas comparer parce que ça sert à rien entre différents paramètres)
    def show_model_evaluation(self, parameter_name : str=None, metrics_dict : dict=None, plot_dict : dict=None):
        """
        Shows various metrics and plots for the given model predictions.

        Parameters:
            parameter_name (str) : name of the parameter which evaluation needs to be printed. If set to None, prints all parameters
            metrics_dict (dict) : dictionary containing the metrics to be printed
            plot_dict (dict) : dictionary containing the plots to be shown
        """
        if metrics_dict is None:
            if self.metrics_dict is None:
                print("Error: metrics_dict not provided.")
                sys.exit(1)
            metrics_dict = self.metrics_dict
        
        if plot_dict is None:
            if self.plot_dict is None:
                print("Error: plot_dict not provided.")
                sys.exit(1)
            plot_dict = self.plot_dict
        
        for param in metrics_dict.keys():
            if parameter_name is not None and param != parameter_name:
                continue # skip to next parameter
            print()
            print(f"{param} results:")
            for key in metrics_dict[param].keys():
                if key != "Percentiles":
                    print(f"{key} : ", metrics_dict[param][key])
                else:
                    print("Percentiles : ")
                    for p in metrics_dict[param][key].keys():
                        print(f"  {p}th percentile : ", metrics_dict[param][key][p])
            print()

            for plot_name in plot_dict[param].keys():
                plt.show()
                # plot_dict[param][plot_name].show()
    
    def save_model_evaluation(self, model_name : str=None, path : str=None, metrics_dict : dict=None, plot_dict : dict=None): # TODO tester le save
        """
        Saves various metrics and plots for the given model predictions to a CSV file and images in a directory.

        Parameters:
            truth (numpy.ndarray) : true values
            preds (numpy.ndarray) : predicted values
            path (str) : the path to the directory in which the metrics will be saved. If set to None, uses the path provided during initialization
        """
        if model_name is None:
            if self.model_name is None:
                print("Error: model_name not provided.")
                sys.exit(1)
            model_name = self.model_name
        if path is None:
            if self.path is None:
                print("Error: path not provided.")
                sys.exit(1)
            path = self.path
        # path sanitization 
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
        self.path = path

        if metrics_dict is None:
            if self.metrics_dict is None:
                print("Error: metrics_dict not provided.")
                sys.exit(1)
            metrics_dict = self.metrics_dict
        if plot_dict is None:
            if self.plot_dict is None:
                print("Error: plot_dict not provided.")
                sys.exit(1)
            plot_dict = self.plot_dict
        
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        if not os.path.exists(path + f"{self.model_name}/"):
            os.makedirs(path + f"{self.model_name}/")
        metrics_df.to_csv(path + f"{self.model_name}/" + f"metrics.csv", sep=',', encoding='utf-8', index=True, header=True)

        for parameter_name in plot_dict.keys():
            for plot_name in plot_dict[parameter_name].keys():
                plot_dict[parameter_name][plot_name].savefig(path + f"{self.model_name}/" + f"{parameter_name}_{plot_name}.png")

    # TODO! ne fonctionne pas
    def evaluate_model(self, model, X_test : np.ndarray, y_test : np.ndarray):
        """
        Evaluates the given model on the test data and prints the metrics.

        Parameters:
            model : trained machine learning model
            X_test (np.ndarray) : dataset of the test features
            y_test (np.ndarray) : dataset of the test targets
        """
        y_pred = model.predict(X_test)
        for i, col in enumerate(y_test.columns):
            self.calculate_model_evaluation(col, y_test[col].values, y_pred[:, i])
            self.show_model_evaluation(col)
            # self.print_model_metrics(y_test[col].values, y_pred[:, i], col)
    
    def evaluate_predictions(self, truth : np.ndarray, preds : np.ndarray, parameter_name : str, save : bool=True):
        """
        Evaluates the given predictions and prints the metrics.

        Parameters:
            y_true (np.ndarray) : true values
            y_pred (np.ndarray) : predicted values
        """
        self.calculate_model_evaluation(parameter_name, truth=truth, preds=preds)
        self.show_model_evaluation(parameter_name)

        if save:
            self.save_model_evaluation()



if __name__ == "__main__":
    # test_class = Iso_data_handler("C:/Users/antoi/Code/unif/MA2/Thèse/data/MIST_v1.2_vvcrit0.0_basic_isos/",
    #                               ['log10_isochrone_age_yr', 'log_Teff', 'log_g', 'star_mass', 'phase'])
    # test_df = test_class.full_iso_data_to_panda(override=False)
    # print(test_df)
    # initialize data of lists.
    test_evaluator = Model_evaluator("test_model", path="C:/Users/antoi/Code/unif/MA2/Thèse/results/K_fold/", residuals_truth_plot=False, residuals_boxplot=False,
                                     residuals_histogram=False,qq_plot=False)

    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, -0.5, 2.0, 8.0])

    test_evaluator.calculate_model_evaluation("test_parameter", truth=y_true, preds=y_pred)
    test_evaluator.show_model_evaluation("test_parameter")

    test_evaluator.save_model_evaluation()

