import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas
from typing import NoReturn, Tuple, List


# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    
    """
    dataframe = pandas.read_csv(path_to_csv, header=0)
    dataframe = dataframe.sample(frac=1)  # перемешал строки
    dataframe_np = np.array(dataframe.values.tolist())
    x = dataframe_np[:, 1:] # убрал столбец label
    x = x.astype(float)
    y = dataframe_np[:, 0]  # взял  стобец label
    y[y == 'M'] = 1  # делал и '1', '0', местами тоже менял
    y[y == 'B'] = 0
    y = y.astype(float)
    return x, y


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    
    """
    dataframe = pandas.read_csv(path_to_csv, header=0)
    dataframe = dataframe.sample(frac=1)
    dataframe_np = np.array(dataframe.values.tolist())
    x = dataframe_np[:, :-1]
    x = x.astype(float)
    y = dataframe_np[:, -1]
    y = y.astype(float)
    # y = y.reshape(1, -1)
    return x, y

    
# Task 2


def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    len_train = int(len(X) * ratio)
    x_train = X[:len_train]
    x_test = X[len_train:]
    y_train = y[:len_train]
    y_test = y[len_train:]
    return x_train, y_train, x_test, y_test

# Task 3


def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    precision, recall, accuracy = [], [], 0
    accuracy = (y_pred == y_true).sum()/len(y_pred)
    classes = np.unique(y_true)
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()
    for elem in classes:
        tp, tn, fp, fn = 0, 0, 0, 0
        # tp = len(y_true[y_true[y_true == y_pred] == elem])
        for i in range(len(y_pred)):
            if y_true[i] == elem and y_pred[i] == elem:
                tp += 1
            elif y_true[i] != elem and y_pred[i] != elem:
                tn += 1
            elif y_true[i] == elem and y_pred[i] != elem:
                fn += 1
            elif y_true[i] != elem and y_pred[i] == elem:
                fp += 1
        precision.append(tp/(tp+fp))
        recall.append(tp/(tp+fn))
        # if accuracy is None:
        #     accuracy = float((tp+tn)/(tp+tn+fp+fn))

    return np.array(precision), np.array(recall), accuracy

# Task 4


class Node:
    def __int__(self, points, med, div_feature):
        self.right = None
        self.left = None
        self.points = points
        self.med = med
        self.div_feature = div_feature


class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        self.leaf_size = leaf_size
        self.points = X
        self.root = self.make_tree(Node(X,))

    def make_tree(self, Node):
        num_of_features = len(self.points[0])
        root =
        cur = root
        while True:
            div_feature = cur.div_feature
            l_points = []
            r_points = []
            if len(cur.points) > self.leaf_size:
                ftrs = cur.points[:, div_feature]
                mean = ftrs.mean()
                for point in cur.points:
                    if point[div_feature] <= mean:
                        l_points.append(point)
                    else:
                        r_points.append(point)





        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области, 
            в которых не меньше leaf_size точек).

        Returns
        -------

        """


    def query(self, X: np.array, k: int = 1) -> List[List]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k): 
            индексы k ближайших соседей для всех точек из X.

        """

        
# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
    
    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        
    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """

        
    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        np.array
            Вектор предсказанных классов.
            

        """
        return np.argmax(self.predict_proba(X), axis=1)


# def main():
#     x, y = read_cancer_dataset('cancer.csv')
#     x_train, y_train, x_test, y_test = train_test_split(x, y, 0.9)
#     p, r, a = get_precision_recall_accuracy(np.array([1,0,1,0]), np.array([0,1,1,0]))
#
# if __name__ == "__main__":
#     main()
