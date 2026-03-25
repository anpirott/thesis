import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import time
import csv
import copy
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
    def __init__(self, model_name : str, output_parameters : list[str], model : str=None, physical_model : str=None, truth : np.ndarray=None, preds : np.ndarray=None, categories : np.ndarray=None, # TODO? rajouter le temps dans le dict?
                 path : str=None, rve : bool=True, rmse : bool=True, mae : bool=True, medae : bool=True, corr : bool=True, maxe : bool=True, percentile : list[int]=(75, 90, 95, 99), 
                 predicted_truth_plot : bool=True, residuals_truth_plot : bool=True, residuals_boxplot : bool=True, residuals_histogram : bool=True, category_residuals_histogram : bool=True, qq_plot : bool=True, 
                 preds_plot : bool=True, category_preds_plot : bool=True, neg_preds_plot : bool=True): # TODO rajouter save ici? rajouter path pour ce qu'on sauvegarde?
        """
        Initializes the Model_evaluator class.

        Parameters:
            model_name (str) : name of the model being evaluated
            output_parameters (list[str]) : list containing the name of the output parameters. To match the right data to the output name, 
                                            the list needs to be in the same order as when splitting the data into the train and test set.
            model (str) : path to the trained machine learning model
            physical_model (str) : what physical model was used to train the model (i.e. "MIST" or "PARSEC")
            truth (numpy.ndarray) : true values
            preds (numpy.ndarray) : predicted values
            categories (numpy.ndarray) : categories of the values
            every bool variable : whether to compute/plot the corresponding metric
            percentile (list of int) : list of percentiles to compute
        """
        self.model_name = model_name
        self.output_parameters = output_parameters
        self.model = model
        self.physical_model = physical_model
        self.truth = truth
        self.preds = preds
        self.categories = categories
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
        self.category_residuals_histogram = category_residuals_histogram
        self.qq_plot = qq_plot
        self.preds_plot = preds_plot
        self.category_preds_plot = category_preds_plot
        self.neg_preds_plot = neg_preds_plot

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

    # TODO rajouter un plot qui montre plus en détail les hautes erreurs (jsp comment : les pourcentiles?, un cut sur les données pour voir les X plus grandes?, ...)
    # TODO faire un plot qui montre l'erreur proportionnelement à la valeur qu'il fallait prédire
    def calculate_model_evaluation(self, parameter_name : str, categories_name : list[str], truth : np.ndarray=None, preds : np.ndarray=None, categories : np.ndarray=None) -> tuple[dict, dict]: #, metrics_dict : dict=None, plot_dict : dict=None) -> tuple[dict, dict]:
        """
        Calculates various metrics and plots for the given model predictions.

        Parameters:
            parameter_name (str) : name of the parameter being evaluated
            truth (numpy.ndarray) : true values
            preds (numpy.ndarray) : predicted values
            categories (numpy.ndarray) : categories of the values
        
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
        if categories is None:
            if self.categories is None:
                print("Error: predicted values not provided.")
                sys.exit(1)
            categories = self.categories

        residuals = preds - truth
        absolute_residuals = np.abs(preds - truth)

        # if metrics_dict is None: # TODO? pour améliorer le code et pas tjs appeler self.metrics_dict
        #     metrics_dict = copy.deepcopy(self.metrics_dict)

        # metrics_dict[parameter_name][category_name][category_value][metric]
        self.metrics_dict[parameter_name] = self.calculate_metrics_by_category(categories_name, truth, preds, categories)
        # TODO? faire un bar chart des différentes catégories et de leur métriques (une métrique, toutes les valeurs groupé l'une à côté de l'autre par catégorie avec différentes couleurs)
        # plt.bar()

        # if plot_dict is None:  # TODO? pour améliorer le code et pas tjs appeler self.metrics_dict
        #     plot_dict = copy.deepcopy(self.plot_dict)

        self.plot_dict[parameter_name] = dict()

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
            plotted_boxplot_log = plt.figure(figsize=(6,6)) # TODO? erreur dans ce boxplot quand j'utilise PARSEC
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
            self.plot_dict[parameter_name]['residuals_hist_plot'] = plotted_histogram
        if self.category_residuals_histogram: # TODO rajouter dans le mémoire si je le garde
            self.plot_dict[parameter_name]['cat_residuals_hist_plot'] = dict()
            for i, cat_name in enumerate(categories_name):
                if cat_name == "Global":
                    continue

                num_cat = len(np.unique(categories[i]))
                a = int(np.round(np.sqrt(num_cat))) # number of rows
                b = (num_cat + a - 1) // a # number of columns

                category_plotted_histogram = plt.figure(figsize=(5*a,4*b))
                axes = category_plotted_histogram.subplots(a, b)
                for j, cat_value in enumerate(np.unique(categories[i])):
                    mask = categories[i] == cat_value
                    cat_truth = truth[mask] # all the truth from a certain category
                    cat_preds = preds[mask] # all the preds from a certain category
                    cat_residuals = cat_preds - cat_truth

                    axes_x, axes_y = j//b, j%b
                    axes[axes_x, axes_y].hist(cat_residuals, bins=60)
                    axes[axes_x, axes_y].set_xlabel('Residuals')
                    axes[axes_x, axes_y].set_ylabel('Frequency')
                    axes[axes_x, axes_y].set_title(f"{cat_name}_{cat_value}")
                # plt.title(f'Histogram of Residuals for {parameter_name} for category {cat_name}')
                self.plot_dict[parameter_name]['cat_residuals_hist_plot'][cat_name] = category_plotted_histogram
        if self.qq_plot: # TODO rajouter dans le mémoire si je le garde
            # compare la distribution des résidus à une autre distribution (normal, exponentielle, etc.), pas utile si la distribution des résidus ne nous intéresse pas => RMSE vs MAE?
            plotted_qq = plt.figure(figsize=(6,6))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.xlabel('Theoretical quantiles')
            plt.ylabel('Ordered Values')
            plt.title('Normal Q-Q plot')
            plt.grid(True)
            self.plot_dict[parameter_name]['qq_plot'] = plotted_qq
        if self.preds_plot:
            plotted_preds = plt.figure(figsize=(6,6))
            plt.hist(preds, bins=50, log=True)
            plt.xlabel('Predictions')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of predictions for {parameter_name}')
            self.plot_dict[parameter_name]['preds_plot'] = plotted_preds
        if self.category_preds_plot:
            self.plot_dict[parameter_name]['cat_preds_plot'] = dict()
            for i, cat_name in enumerate(categories_name):
                if cat_name == "Global":
                    continue

                num_cat = len(np.unique(categories[i]))
                a = int(np.round(np.sqrt(num_cat))) # number of rows
                b = (num_cat + a - 1) // a # number of columns

                category_plotted_preds = plt.figure(figsize=(5*a,4*b))
                axes = category_plotted_preds.subplots(a, b)
                for j, cat_value in enumerate(np.unique(categories[i])):
                    mask = categories[i] == cat_value
                    cat_preds = preds[mask] # all the preds from a certain category

                    axes_x, axes_y = j//b, j%b
                    axes[axes_x, axes_y].hist(cat_preds, bins=50, log=True)
                    axes[axes_x, axes_y].set_xlabel('Predictions')
                    axes[axes_x, axes_y].set_ylabel('Frequency')
                    axes[axes_x, axes_y].set_title(f"{cat_name}_{cat_value}")
                # plt.title(f'Histogram of Predictions for {parameter_name} for category {cat_name}')
                self.plot_dict[parameter_name]['cat_preds_plot'][cat_name] = category_plotted_preds
        if self.neg_preds_plot:
            mask = preds <= 0  # [False, True, False, True, False, True, False, True]
            filtered_preds = preds[mask]
            if len(filtered_preds) == 0:
                print("no negative values")
            else:
                plotted_neg_preds = plt.figure(figsize=(6,6))
                plt.hist(filtered_preds, bins=20)
                plt.xlabel('Predictions')
                plt.ylabel('Frequency')
                plt.title(f'Histogram of negative predictions for {parameter_name}')
                self.plot_dict[parameter_name]['preds_plot'] = plotted_neg_preds
        
        return self.metrics_dict[parameter_name], self.plot_dict[parameter_name]
    
    def calculate_metrics_by_category(self, categories_name : list[str], truth : np.ndarray, preds : np.ndarray, categories : np.ndarray):
        """
        Calculates the metrics for each category of the values.

        Parameters:
            categories_name (list[str]) : list containing the names of the different categories in the same order as in categories_train
            truth (numpy.ndarray) : true values
            preds (numpy.ndarray) : predicted values
            categories (numpy.ndarray) : categories of the values

        Returns:
            dict : a dictionary containing the calculated metrics for each category
        """
        sub_metrics_dict = dict()

        for i, cat_name in enumerate(categories_name):
            sub_metrics_dict[cat_name] = dict()

            for cat_value in np.unique(categories[i]):
                mask = categories[i] == cat_value
                cat_truth = truth[mask] # all the truth from a certain category
                cat_preds = preds[mask] # all the preds from a certain category
                # cat_residuals = cat_preds - cat_truth
                cat_absolute_residuals = np.abs(cat_preds - cat_truth)

                sub_metrics_dict[cat_name][cat_value] = dict()

                sub_metrics_dict[cat_name][cat_value]["value range"] = f"{np.min(cat_truth)} - {np.max(cat_truth)}"
                if self.rve:
                    sub_metrics_dict[cat_name][cat_value]['RVE'] = explained_variance_score(cat_truth, cat_preds)
                if self.rmse:
                    sub_metrics_dict[cat_name][cat_value]['RMSE'] = root_mean_squared_error(cat_truth, cat_preds)
                if self.mae:
                    sub_metrics_dict[cat_name][cat_value]['MAE'] = mean_absolute_error(cat_truth, cat_preds)
                if self.medae:
                    sub_metrics_dict[cat_name][cat_value]['MedAE'] = median_absolute_error(cat_truth, cat_preds)
                if self.corr:
                    sub_metrics_dict[cat_name][cat_value]['CORR'], _ = pearsonr(cat_truth, cat_preds)
                if self.maxe:
                    sub_metrics_dict[cat_name][cat_value]['MAX_ER'] = max_error(cat_truth, cat_preds)
                if isinstance(self.percentile, tuple) and len(self.percentile) > 0:
                    sub_metrics_dict[cat_name][cat_value]['Percentiles'] = dict()
                    for p in self.percentile:
                        if p >= 0 and p <= 100:
                            sub_metrics_dict[cat_name][cat_value]['Percentiles'][p] = np.percentile(cat_absolute_residuals, p) 
        
        return copy.deepcopy(sub_metrics_dict)

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
    def show_model_evaluation(self, categories_name : list[str], parameter_name : str=None, metrics_dict : dict=None, plot_dict : dict=None):
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
        
        print()
        for param in metrics_dict.keys():
            if parameter_name is not None and param != parameter_name:
                continue # skip to next parameter
            for cat_name in metrics_dict[param].keys():
                for cat_value in metrics_dict[param][cat_name].keys():
                    print(f"{param} results for the {cat_name} category with a value of {cat_value}")
                    for metric in metrics_dict[param][cat_name][cat_value].keys():
                        if metric != "Percentiles":
                            print(f"  {metric} : ", metrics_dict[param][cat_name][cat_value][metric])
                        else:
                            print(f"  Percentiles : ")
                            for p in metrics_dict[param][cat_name][cat_value][metric].keys():
                                print(f"    {p}th percentile : ", metrics_dict[param][cat_name][cat_value][metric][p])
                    print()

            for plot_name in plot_dict[param].keys():
                if isinstance(plot_dict[param][plot_name], dict):
                    for cat_plot in plot_dict[param][plot_name]:
                        plt.show()
                else:
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
        
        plot_path = path + f"{model_name}/{physical_model}/{tag}/"
        metrics_path = path + f"{model_name}/{physical_model}/{tag}/metrics/"

        if not os.path.exists(metrics_path): # creates the path for both the metrics and the plots
            os.makedirs(metrics_path)
                
        temp_dict = dict()
        for param in metrics_dict.keys():
            for cat_name in metrics_dict[param].keys():
                for cat_value in metrics_dict[param][cat_name].keys():
                    if f"{cat_name}_{cat_value}" not in temp_dict.keys():
                        temp_dict[f"{cat_name}_{cat_value}"] = dict()
                    temp_dict[f"{cat_name}_{cat_value}"][f"{param}"] = copy.deepcopy(metrics_dict[param][cat_name][cat_value])

        for key in temp_dict.keys():
            cat_name, cat_value = key.split("_")
            temp_df = pd.DataFrame.from_dict(temp_dict[key], orient='index')
            temp_df.to_csv(metrics_path + f"{cat_name}_{cat_value}_metrics.csv", sep=',', encoding='utf-8', index=True, header=True)

        if time is not None:
            with open(plot_path + "time_taken.txt", 'w') as file:
                file.write(f"Time,method\n{time},{train_method}")

        for parameter_name in plot_dict.keys():
            for plot_name in plot_dict[parameter_name].keys():
                if isinstance(plot_dict[param][plot_name], dict):
                    for cat_plot in plot_dict[param][plot_name]:
                        print(os.path.exists(plot_path))
                        plot_dict[param][plot_name][cat_plot].savefig(plot_path + f"{parameter_name}_{plot_name}_{cat_plot}.png")
                else:
                    plot_dict[parameter_name][plot_name].savefig(plot_path + f"{parameter_name}_{plot_name}.png")

    # TODO! ne fonctionne pas, pas encore fini
    def evaluate_model(self, model, X_test : np.ndarray, y_test : np.ndarray):
        """
        Evaluates the given model on the test data and prints the metrics.

        Parameters:
            model : trained machine learning model
            X_test (np.ndarray) : dataset of the test features
            y_test (np.ndarray) : dataset of the test targets
        """
        # y_pred = model.predict(X_test)
        # for i, col in enumerate(y_test.columns):
        #     self.calculate_model_evaluation(col, y_test[col].values, y_pred[:, i])
        #     self.show_model_evaluation(col)
        pass
    
    def evaluate_predictions(self, truth : np.ndarray, preds : np.ndarray, categories : np.ndarray, categories_name : list[str], parameter_name : str, show : bool=True):
        """
        Evaluates the given predictions and prints the metrics.

        Parameters:
            truth (np.ndarray) : true values
            preds (np.ndarray) : predicted values
            categories (np.ndarray) : categories of the values
            categories_name (list[str]) : list containing the names of the different categories in the same order as in categories_train
            parameter_name (str) : name of the parameter being evaluated
            tag (str) : tag for the type of data used (e.g., "Base", "PCA", etc.)
            save (bool) : whether to save the metrics and plots
            time (float) : time taken for the model to have been trained
            train_method (str) : what method was used to train the model (i.e. "K_fold" or "normal")
            show (bool) : whether to show the metrics and plots
        """
        self.calculate_model_evaluation(parameter_name, truth=truth, preds=preds, categories=categories, categories_name=categories_name)
        # self.metrics_dict, self.plot_dict = self.calculate_model_evaluation(parameter_name, truth=truth, preds=preds) # TODO? si je change le code comme le todo dans la fct
        if show:
            self.show_model_evaluation(categories_name=categories_name, parameter_name=parameter_name)
        else:
            plt.close('all')
    
    def evaluate_Kfold_results(self, model : Callable, X_train : np.ndarray, y_train : np.ndarray, categories_train : np.ndarray, categories_name : list[str], 
                               path : str, tag : str, n_splits : int=10, random_state : int=12, override : bool=False, use_preds : bool=False, 
                               show_depth : bool=True, save : bool=True, add_global : bool=True, show : bool=True, **kwargs):
        """
        Generates K-fold cross-validation results for the given model and training data.
        Also saves the prediction, truth and categories in a ".npy" file.

        Parameters:
            model (Callable) : machine learning model to be trained
            X_train (numpy.ndarray) : training features
            y_train (numpy.ndarray) : training targets
            categories_train (numpy.ndarray) : training categories of the values
            categories_name (list[str]) : list containing the names of the different categories in the same order as in categories_train
            tag (str) : tag for the type of data used (e.g., "Base", "PCA", etc.)
            path (str) : path to the directory in which the predictions and truths will be saved
            override (bool) : whether to override existing results or use the existing ones
            use_preds (bool) : whether to use existing predictions instead of generating new ones
            save (bool) : wether or not to save the results
            add_global (bool) : whether to add the global metrics (i.e. not separated by category) to the metrics_dict
            show (bool) : whether to show the metrics and plots
            **kwargs : additional arguments which will be passed to the model during training
        """
        if add_global: 
            categories_train = np.append(categories_train, [[1] for _ in range(len(categories_train))], axis=1)
            if "Global" not in categories_name: # second condition for running the same cell multiple times in a jupyter notebook
                categories_name.append("Global")
            # adds a column to categories_name with the value 1 for all rows

        print(f"\n{tag} train data :")
        if not use_preds: # not using the existing predictions
            if not override and self.check_existing_results(tag): # not overriding, results need to exist and we show if that is the case, results are not recreated
                self.show_existing_results(tag) # TODO encore ça à faire et la fonction de comparaison
            else: # overriding, creating the results and predictions
                print("Performing K-fold...")
                start = time.time()
                truth, preds, categories = Model_trainer.Kfold_pipeline(model, X=X_train, y=y_train, categories=categories_train, n_splits=n_splits, random_state=random_state, **kwargs)
                end = time.time()
                print("Evaluating predictions...")
                for i, output_param in enumerate(self.output_parameters):
                    self.evaluate_predictions(truth[i], preds[i], categories, categories_name, output_param, show=show)
                if save:
                    print("Saving predictions and results...")
                    self.save_numpy_array(preds, path, f"{tag}_predictions.npy")
                    self.save_numpy_array(truth, path, f"{tag}_truths.npy")
                    self.save_numpy_array(categories, path, f"{tag}_categories.npy")
                    self.save_model_evaluation(tag=tag, time=end-start, train_method="K_fold")
        if use_preds: # using the existing predictions
            if override: # overriding, we do not allow this case
                print("Error: cannot override when using existing predictions. Set either override or use_preds to False.")
                sys.exit(1)
            elif not override: # not overriding, we use the existing predictions if they exist and do not save the results
                if not os.path.exists(path + f"{self.model_name}"): # TODO? utilisation de self.model_name, si pas spécifié ça créé un problème
                    print("Error: predictions do not exist.")
                    sys.exit(1)
                preds = self.load_numpy_array(path, f"{tag}_predictions.npy")
                truth = self.load_numpy_array(path, f"{tag}_truths.npy")
                categories = self.load_numpy_array(path, f"{tag}_categories.npy")
                print("Evaluating predictions...")
                for i, output_param in enumerate(self.output_parameters):
                    self.evaluate_predictions(truth[i], preds[i], categories, categories_name, output_param, show=show)
    
    def check_existing_results(self, tag : str, model_name : str=None, path : str=None, physical_model : str=None) -> bool:
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
        if physical_model is None:
            if self.physical_model is None:
                print("Error: physical_model not provided.")
                sys.exit(1)
            physical_model = self.physical_model
        
        exists = False
        for filename in os.listdir(path + f"{model_name}/{physical_model}/{tag}/metrics"):
            if not exists and filename.endswith("_metrics.csv"):
                exists = True
        
        if exists:
            return True
        return False
    
    def show_existing_results(self, tag : str, model_name : str=None, path : str=None, physical_model : str=None):
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
        if physical_model is None:
            if self.physical_model is None:
                print("Error: physical_model not provided.")
                sys.exit(1)
            physical_model = self.physical_model

        plot_path = path + f"{model_name}/{physical_model}/{tag}/"
        metrics_path = path + f"{model_name}/{physical_model}/{tag}/metrics/"

        for filename in os.listdir(metrics_path):
            if filename.endswith(".csv"):
                with open(metrics_path + filename, 'r') as file:
                    results = csv.DictReader(file)
                    cat_name, cat_value = filename.split("_")[0], filename.split("_")[1]
                    for line_dict in results:
                        print(f"{line_dict['']} results for the {cat_name} category with a value of {cat_value}")
                        for metric in line_dict.keys():
                            if metric == "":
                                continue
                            if metric != "Percentiles":
                                print(f"  {metric} : ", line_dict[metric])
                            elif metric == "Percentiles":
                                print("  Percentiles : ")
                                percentiles_dict = eval(line_dict[metric])
                                for p in percentiles_dict.keys():
                                    print(f"    {p}th percentile : ", percentiles_dict[p])
                        print()

        for filename in os.listdir(plot_path):
            if filename.endswith(".png"):
                display(Image(filename=plot_path + filename)) # TODO ne fonctionne p-ê que dans un notebook

    def save_numpy_array(self, arr : np.ndarray, path : str, filename : str, model_name : str=None, physical_model : str=None):
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
        if physical_model is None:
            if self.physical_model is None:
                print("Error: physical_model not provided.")
                sys.exit(1)
            physical_model = self.physical_model

        path = sanitize_path(path)
        full_path = path + f"{model_name}/{physical_model}/"
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        np.savetxt(full_path + filename, arr)
    
    def load_numpy_array(self, path : str, filename : str, model_name : str=None, physical_model : str=None) -> np.ndarray:
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
        if physical_model is None:
            if self.physical_model is None:
                print("Error: physical_model not provided.")
                sys.exit(1)
            physical_model = self.physical_model
        
        path = sanitize_path(path)
        full_path = path + f"{model_name}/{physical_model}/"
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