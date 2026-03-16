import numpy as np
from collections.abc import Callable
import sys

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

# TODO rajouter les exceptions pour les erreurs
class Model_trainer():
    """
    Class which contains methods for training machine learning models.

    Methods:
        Kfold_pipeline : performs K-fold cross-validation on the given model and training data
    """
    # mettre les outputs (preds) du modèle dans un fichier csv pour utiliser plus tard, mis en pause parce que ça prend pas mal de place
    # TODO mettre la possibilité de rajouter des paramètres à tester dans le modèle
    # TODO? rajouter le calcul du temps et le rajouter dans le csv
    @staticmethod
    def Kfold_pipeline(model : Callable, X : np.ndarray, y : np.ndarray, categories : np.ndarray, n_splits : int=10, 
                       shuffle : bool=True, random_state : int=12, verbose : bool=True, **kwargs) -> tuple[list, list]:
        """
        Performs K-fold cross-validation on the given model and training data.

        Parameters:
            model (Callable) : machine learning model to be trained
            X (numpy.ndarray) : training features
            y (numpy.ndarray) : training targets
            categories (numpy.ndarray) : training categories of the values
            n_splits (int) : number of folds for cross-validation
            shuffle (bool) : whether to shuffle the data before splitting into folds
            random_state (int) : random seed for shuffling the data
            verbose (bool) : whether information during the training is shown or not
        
        Returns:
            tuple of three lists : a list containing the true values for each target across all folds,
                                   a list containing the predicted values for each target across all folds
                                   and a list containing the associated categories for each value across all folds
        """
        truth = list(None for _ in range(y.shape[1]))
        preds = list(None for _ in range(y.shape[1]))
        shuffled_categories = list(None for _ in range(categories.shape[1]))

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state) # TODO? si random_state=None je fais une ligne à part
        counter = 0
        if verbose:
            print("split", end=' ')
        for train_index, test_index in kf.split(X):
            counter += 1
            if verbose:
                print(str(counter), end=' ')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            categories_train, categories_test = categories[train_index], categories[test_index]

            mdl = model(**kwargs)
            mdl.fit(X_train, y_train)
            if "batch_size" in kwargs:
                plt.plot(range(len(mdl.loss_curve_)), mdl.loss_curve_)
                print(mdl.best_loss_)
            fold_preds = mdl.predict(X_test)

            # if show_depth:
            #     max_depth = list()
            #     for tree in mdl.estimators_:
            #         max_depth.append(tree.tree_.max_depth)
            #     print("avg max depth %0.1f" % (sum(max_depth) / len(max_depth)))

            for i in range(y.shape[1]):
                if truth[i] is None:
                    truth[i] = y_test[:, i]
                    preds[i] = fold_preds[:, i]
                else:
                    truth[i] = np.hstack((truth[i], y_test[:, i]))
                    preds[i] = np.hstack((preds[i], fold_preds[:, i]))
            for i in range(categories.shape[1]):
                if shuffled_categories[i] is None:
                    shuffled_categories[i] = categories_test[:, i]
                else:
                    shuffled_categories[i] = np.hstack((shuffled_categories[i], categories_test[:, i]))
        
        return truth, preds, shuffled_categories

    # TODO quand je devrai train des modèles tous seuls avec paramètres
    def train_model():
        pass

if __name__ == "__main__":
    pass

    # def test(a, b):
    #     print(a+b)
    
    # params = ["a=1", "b=2"]

    # test(eval(params[0]), eval(params[1]))

    # def test(a, b, **kwargs):
    #     print(a+b)
    #     print(type(kwargs))
    #     print(kwargs)
    #     print(*kwargs)
    #     print(kwargs.items())
    
    # test(1, 2, c=3, d=4)