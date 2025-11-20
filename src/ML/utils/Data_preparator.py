import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


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
    def split_data(data_df : pd.DataFrame, x_cols : list, y_cols : list, test_size : float=0.25, shuffle : bool=True, random_state : int=None, print_stats : bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the given dataframe into training and testing sets.

        Parameters:
            data_df (pandas.DataFrame) : dataframe to be split
            x_cols (list) : list of column names to be used as features
            y_cols (list) : list of column names to be used as targets
            test_size (float) : proportion of the data to be used as test set
            shuffle (bool) : whether to shuffle the data before splitting
            random_state (int) : random seed for shuffling the data
            print_stats (bool) : whether to print some statistics of the train and test sets
        
        Returns:
            tuple of four numpy.ndarrays : the training features (X_train), testing features (X_test), training targets (y_train) and testing targets (y_test)
        """
        X = data_df[x_cols].to_numpy()
        y = data_df[y_cols].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

        if print_stats:
            print("Training set statistics:")
            Data_preparator.show_data_stats(y_train, "train")
            print("Testing set statistics:")
            Data_preparator.show_data_stats(y_test, "test")

        return X_train, X_test, y_train, y_test
    
    @staticmethod    
    def show_data_stats(data : np.ndarray, set_type : str): # TODO? changer pour que ça fasse pour un nombre indéterminé de paramètres (pas juste mass et radius)
        """
        Shows some statistics of the given data.

        Parameters:
            data (numpy.ndarray) : data to be analyzed
            set (str) : name of the dataset (e.g. "train" or "test")
        """
        print(f"Range in {set_type} data for the mass parameter : {min(data[:, 0])} - {max(data[:, 0])}")
        print(f"Median value in {set_type} data for the mass parameter: {np.median(data[:, 0])}")
        print(f"Mean value in {set_type} data for the mass parameter: {np.mean(data[:, 0])}")

        print(f"Range in {set_type} data for the radius parameter : {min(data[:, 1])} - {max(data[:, 1])}")
        print(f"Median value in {set_type} data for the radius parameter: {np.median(data[:, 1])}")
        print(f"Mean value in {set_type} data for the radius parameter: {np.mean(data[:, 1])}\n")
    
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