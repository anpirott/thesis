import numpy as np
from collections.abc import Callable

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
    def Kfold_pipeline(model : Callable, X_train_data : np.ndarray, y_train_data : np.ndarray, n_splits : int=10, 
                       shuffle : bool=True, random_state : int=12, **kwargs) -> tuple[list, list]:
        """
        Performs K-fold cross-validation on the given model and training data.

        Parameters:
            model (Callable) : machine learning model to be trained
            X_train_data (numpy.ndarray) : training features
            y_train_data (numpy.ndarray) : training targets
            n_splits (int) : number of folds for cross-validation
            shuffle (bool) : whether to shuffle the data before splitting into folds
            random_state (int) : random seed for shuffling the data
        
        Returns:
            tuple of two lists : a tuple containing the true values for each target across all folds and a tuple containing the predicted values for each target across all folds
        """
        truth = list(None for _ in range(y_train_data.shape[1]))
        preds = list(None for _ in range(y_train_data.shape[1]))

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state) # TODO? si random_state=None je fais une ligne à part
        counter = 0
        print("split", end=' ')
        for train_index, test_index in kf.split(X_train_data):
            counter += 1
            print(str(counter), end=' ')
            X_train, X_test = X_train_data[train_index], X_train_data[test_index]
            y_train, y_test = y_train_data[train_index], y_train_data[test_index]

            mdl = model(**kwargs)
            mdl.fit(X_train, y_train)
            if "batch_size" in kwargs:
                plt.plot(mdl.loss_curve_, range(len(mdl.loss_curve_)))
                print(mdl.best_loss_)
            fold_preds = mdl.predict(X_test)

            for i in range(y_train_data.shape[1]):
                if truth[i] is None:
                    truth[i] = y_test[:, i]
                    preds[i] = fold_preds[:, i]
                else:
                    truth[i] = np.hstack((truth[i], y_test[:, i]))
                    preds[i] = np.hstack((preds[i], fold_preds[:, i]))
        
        return truth, preds

    # TODO quand je devrai train des modèles tous seuls avec paramètres
    def train_model():
        pass

if __name__ == "__main__":
    # pass

    # def test(a, b):
    #     print(a+b)
    
    # params = ["a=1", "b=2"]

    # test(eval(params[0]), eval(params[1]))

    def test(a, b, **kwargs):
        print(a+b)
        print(type(kwargs))
        print(kwargs)
        print(*kwargs)
        print(kwargs.items())
    
    test(1, 2, c=3, d=4)