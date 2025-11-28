import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import time
import csv
from collections.abc import Callable
from IPython.display import Image
from IPython.display import display

from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error,\
                            root_mean_squared_error, median_absolute_error
from scipy.stats import pearsonr

from ML.utils.utils import sanitize_path
from ML.utils.Model_trainer import Model_trainer


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
    def __init__(self, model_name : str, model : str=None, physical_model : str=None, truth : np.ndarray=None, preds : np.ndarray=None, path : str=None, rve : bool=True, # TODO? rajouter le temps dans le dict?
                 rmse : bool=True, mae : bool=True, medae : bool=True, corr : bool=True, maxe : bool=True, percentile : list[int]=(75, 90, 95, 99), 
                 predicted_truth_plot : bool=True, residuals_truth_plot : bool=True, residuals_boxplot : bool=True, residuals_histogram : bool=True, qq_plot : bool=True): # TODO rajouter save ici? rajouter path pour ce qu'on sauvegarde?
        """
        Initializes the Model_evaluator class.

        Parameters:
            model_name (str) : name of the model being evaluated
            model (str) : path to the trained machine learning model
            physical_model (str) : what physical model was used to train the model (i.e. "MIST" or "PARSEC")
            truth (numpy.ndarray) : true values
            preds (numpy.ndarray) : predicted values
            every bool variable : whether to compute/plot the corresponding metric
            percentile (list of int) : list of percentiles to compute
        """
        self.model_name = model_name
        self.model = model
        self.physical_model = physical_model
        self.truth = truth
        self.preds = preds
        self.path = sanitize_path(path)

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
    
    # TODO décrire tous les plots
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
        
        residuals = preds - truth
        absolute_residuals = np.abs(preds - truth)

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
            for p in self.percentile:
                if p >= 0 and p <= 100:
                    self.metrics_dict[parameter_name]['Percentiles'][p] = np.percentile(absolute_residuals, p)

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
            sns.boxplot(y=residuals, fliersize=2, whis=(10, 90)) # without log scale
            plt.ylabel('Residuals')
            plt.title(f'Box Plot of Residuals for {parameter_name}')
            self.plot_dict[parameter_name]['residuals_boxplot_no_log'] = plotted_boxplot_no_log
        if self.residuals_boxplot: # TODO rajouter dans le mémoire si je le garde
            # le IQR se calcule sur les points de données, pas sur les pourcentages (https://www.geeksforgeeks.org/machine-learning/box-plot/)
            plotted_boxplot_log = plt.figure(figsize=(6,6)) 
            sns.boxplot(y=absolute_residuals, log_scale=True, fliersize=2, whis=(10,90)) # change les whiskers pour qu'ils soient à des percentiles précis
            plt.ylabel('Residuals') # TODO p-ê une erreur, ou alors juste à cause des valeurs du test
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
            # compare la distribution des résidus à une autre distribution (normal, exponentielle, etc.), pas utile si la distribution des résidus ne nous intéresse pas => RMSE vs MAE?
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
    
    def save_model_evaluation(self,  tag : str, model_name : str=None, physical_model : str=None, path : str=None, metrics_dict : dict=None, plot_dict : dict=None, time : float=None, train_method : str=None):
        """
        Saves various metrics and plots for the given model predictions to a CSV file and images in a directory.

        Parameters:
            tag (str) : tag for the type of data used (e.g., "Base", "PCA", etc.)
            model_name (str) : name of the model being evaluated. If set to None, uses the name provided during initialization
            physical_model (str) : what physical model was used to train the model (i.e. "MIST" or "PARSEC")
            path (str) : the path to the directory in which the metrics will be saved. If set to None, uses the path provided during initialization
            metrics_dict (dict) : dictionary containing the metrics to be saved
            plot_dict (dict) : dictionary containing the plots to be saved
        """
        if model_name is None:
            if self.model_name is None:
                print("Error: model_name not provided.")
                sys.exit(1)
            model_name = self.model_name
        if physical_model is None:
            if self.physical_model is None:
                print("Error: physical_model not provided.")
                sys.exit(1)
            physical_model = self.physical_model
        if path is None:
            if self.path is None:
                print("Error: path not provided.")
                sys.exit(1)
            path = self.path
        else:
            path = sanitize_path(path)

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
        
        full_path = path + f"{model_name}/{physical_model}/{tag}/"
        
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        metrics_df.to_csv(full_path + f"metrics.csv", sep=',', encoding='utf-8', index=True, header=True)

        if time is not None:
            with open(full_path + "time_taken.txt", 'w') as file:
                file.write(f"Time,method\n{time},{train_method}")

        for parameter_name in plot_dict.keys():
            for plot_name in plot_dict[parameter_name].keys():
                plot_dict[parameter_name][plot_name].savefig(full_path + f"{parameter_name}_{plot_name}.png")

    # TODO! ne fonctionne pas, pas encore fini
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
    
    def evaluate_predictions(self, truth : np.ndarray, preds : np.ndarray, parameter_name : str, tag : str, save : bool=True, time : float=None, train_method : str=None):
        """
        Evaluates the given predictions and prints the metrics.

        Parameters:
            truth (np.ndarray) : true values
            preds (np.ndarray) : predicted values
            parameter_name (str) : name of the parameter being evaluated
            tag (str) : tag for the type of data used (e.g., "Base", "PCA", etc.)
            save (bool) : whether to save the metrics and plots
            time (float) : time taken for the model to have been trained
            train_method (str) : what method was used to train the model (i.e. "K_fold" or "normal")
        """
        self.calculate_model_evaluation(parameter_name, truth=truth, preds=preds)
        self.show_model_evaluation(parameter_name)

        if save:
            self.save_model_evaluation(tag=tag, time=time, train_method=train_method)
    
    def evaluate_Kfold_results(self, model : Callable, X_train_data : np.ndarray, y_train_data : np.ndarray, path : str, tag : str, n_splits : int=10, random_state : int=12, override : bool=False, use_preds : bool = False, **kwargs):
        """
        Generates K-fold cross-validation results for the given model and training data.

        Parameters:
            model (Callable) : machine learning model to be trained
            X_train_data (numpy.ndarray) : training features
            y_train_data (numpy.ndarray) : training targets
            tag (str) : tag for the type of data used (e.g., "Base", "PCA", etc.)
            path (str) : path to the directory in which the predictions and truths will be saved
            override (bool) : whether to override existing results or use the existing ones
            use_preds (bool) : whether to use existing predictions instead of generating new ones
            **kwargs : additional arguments which will be passed to the model during training
        """
        print(f"\n{tag} train data :")
        if not use_preds:
            if not override and self.check_existing_results(tag):
                self.show_existing_results(tag)
            else:
                start = time.time()
                truth, preds = Model_trainer.Kfold_pipeline(model, X_train_data=X_train_data, y_train_data=y_train_data, n_splits=n_splits, random_state=random_state, **kwargs)
                end = time.time()
                self.save_numpy_array(preds, path, f"{tag}_predictions.npy")
                self.save_numpy_array(truth, path, f"{tag}_truths.npy")
                self.evaluate_predictions(truth[0], preds[0], "mass", tag, time=end-start, train_method="K_fold") # TODO? pas de façon de le faire dans un loop parce qu'on ne connait pas le paramètre (mass ou radius)
                self.evaluate_predictions(truth[1], preds[1], "radius", tag, time=end-start, train_method="K_fold") # TODO? rajouter une liste des parameter_name à la classe?
        if use_preds:
            if override:
                print("Error: cannot override when using existing predictions. Set either override or use_preds to False.")
                sys.exit(1)
            elif not override:
                if not self.check_existing_results(tag): # TODO? faire une fonctions qui check si spécifiquement les predictions existent
                    print("Error: predictions do not exist.")
                    sys.exit(1)
                preds = self.load_numpy_array(path, f"{tag}_predictions.npy")
                truth = self.load_numpy_array(path, f"{tag}_truths.npy")
                self.evaluate_predictions(truth[0], preds[0], "mass", tag)
                self.evaluate_predictions(truth[1], preds[1], "radius", tag)
    
    def check_existing_results(self, tag : str, model_name : str=None, path : str=None) -> bool:
        """
        Checks if the results for the given tag already exist.

        Parameters:
            tag (str) : tag for the type of data used (e.g., "Base", "PCA", etc.)
        
        Returns:
            bool : True if the results already exist, False otherwise
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
        else:
            path = sanitize_path(path)
        if os.path.exists(path + f"{model_name}/{tag}/metrics.csv"):
            return True
        return False
    
    def show_existing_results(self, tag : str, model_name : str=None, path : str=None):
        """
        Loads the existing results for the given tag.

        Parameters:
            tag (str) : tag for the type of data used (e.g., "Base", "PCA", etc.)
            model_name (str) : name of the model being evaluated. If set to None, uses the name provided during initialization
            path (str) : the path to the directory in which the metrics are saved. If set to None, uses the path provided during initialization
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
        else:
            path = sanitize_path(path)
        full_path = path + f"{model_name}/{tag}/"

        for filename in os.listdir(full_path):
            if filename.endswith(".csv"):
                with open(full_path + filename, 'r') as file:
                    results = csv.DictReader(file)
                    for line_dict in results:
                        print(f"{line_dict['']} results:")
                        for key in line_dict.keys():
                            if key == "":
                                continue
                            if key != "Percentiles":
                                print(f"{key} : ", line_dict[key])
                            elif key == "Percentiles":
                                print("Percentiles : ")
                                percentiles_dict = eval(line_dict[key])
                                for p in percentiles_dict.keys():
                                    print(f"  {p}th percentile : ", percentiles_dict[p])
                        print()

            elif filename.endswith(".png"):
                display(Image(filename=full_path + filename)) # TODO ne fonctionne p-ê que dans un notebook

    def save_numpy_array(self, arr : np.ndarray, path : str, filename : str, model_name : str=None):
        """
        Saves the given numpy array to a .npy file.

        Parameters:
            preds (np.ndarray) : predicted values
            path (str) : path to the directory in which the numpy array will be saved
            filename (str) : name of the file in which the numpy array will be saved
        """
        if model_name is None:
            if self.model_name is None:
                print("Error: model_name not provided.")
                sys.exit(1)
            model_name = self.model_name
        path = sanitize_path(path)
        full_path = path + model_name + "/"
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        np.savetxt(full_path + filename, arr)
    
    def load_numpy_array(self, path : str, filename : str, model_name : str=None) -> np.ndarray: # TODO à tester
        """
        Loads the numpy array from a .npy file.

        Parameters:
            path (str) : path to the directory in which the numpy array are saved
            filename (str) : name of the file in which the numpy array are saved
            model_name (str) : name of the model whose numpy array are to be loaded
        """
        if model_name is None:
            if self.model_name is None:
                print("Error: model_name not provided.")
                sys.exit(1)
            model_name = self.model_name
        path = sanitize_path(path)
        full_path = path + model_name + "/"
        return np.loadtxt(full_path + filename)


if __name__ == "__main__":
    pass
    # test_evaluator = Model_evaluator("test_model", path="C:/Users/antoi/Code/unif/MA2/Thèse/results/K_fold/", residuals_truth_plot=False, residuals_boxplot=False,
    #                                  residuals_histogram=False,qq_plot=False)

    # y_true = np.array([3.0, -0.5, 2.0, 7.0])
    # y_pred = np.array([2.5, -0.5, 2.0, 8.0])

    # test_evaluator.calculate_model_evaluation("test_parameter", truth=y_true, preds=y_pred)
    # test_evaluator.show_model_evaluation("test_parameter")

    # test_evaluator.save_model_evaluation()