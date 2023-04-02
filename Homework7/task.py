import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False

# Task 1

class LinearSVM:
    def __init__(self, C: float):
        """
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        """
        self.C = C
        self.w = None
        self.b = None
        self.support = None     # индексы опорных элементов

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1).
        """
        X_norm = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
        m = len(X_norm)
        y = y.reshape(m, 1).astype(float)

        P = matrix((y * X_norm).dot((y * X_norm).T))
        q = matrix(np.ones((m, 1)) * -1)
        G = matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = matrix(y.reshape(1, m))
        b = matrix(np.zeros(1))

        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol["x"])

        self.support = np.where((alpha > 1e-4) & (alpha < self.C - 1e-4))[0]
        self.w = np.sum(alpha * y * X_norm, axis=0)

        self.b = np.mean(-y[self.support] + X_norm[self.support].dot(self.w))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).
        """
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        return np.ravel(X_norm.dot(self.w.reshape(-1, 1)) - self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))
    
# Task 2


def get_polynomial_kernel(c=1, power=2):
    "Возвращает полиномиальное ядро с заданной константой и степенью"
    def ker(X1,X2):
        if X2.ndim == 1:
            X2 = X2[None, :]
            t = (c + X1.dot(X2.T))**power
            return np.ravel(t)
        else:
            return (c + X1.dot(X2.T))**power
    return ker
    #return lambda X1, X2: (c + X1.dot(X2.T))**power


def get_gaussian_kernel(sigma=1.):
    "Возвращает ядро Гаусса с заданным коэффицинтом сигма"
    def ker(X, y):
        rvl = False
        if y.ndim == 1:
            rvl = True
        #norm_sq = np.sum((X - y[None, :]) ** 2, axis=-1)
        norm_sq = np.sum((X[:, None, :] - np.array(y)) ** 2, axis=-1)
        #norm_sq = np.sum((X[:, None, :] - y) ** 2, axis=-1)
        #e = np.exp(-1/sigma**2 * norm_sq)
        e = np.exp(-sigma * norm_sq)
        if rvl:
            return np.ravel(e)
        return e
    return ker

# Task 3

class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        kernel : Callable
            Функция ядра.
        
        """
        self.C = C
        self.kernel = kernel
        self.support = None
        self.b = None
        self.y = None
        self.alpha = None
        self.X_norm = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1). 
        
        """
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        m = len(X_norm)
        y = y.reshape(m, 1).astype(float)

        K = self.kernel(X_norm, X_norm)
        P = matrix(y * K * y.T)
        q = matrix(np.ones((m, 1)) * -1)
        G = matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = matrix(y.reshape(1, m))
        b = matrix(np.zeros(1))

        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol["x"])

        self.support = np.where((alpha > 1e-4) & (alpha < self.C - 1e-4))[0]

        K_sup = self.kernel(X_norm[self.support], X_norm[self.support])
        wx_sup = np.sum(alpha[self.support] * y[self.support] * K_sup, axis=0).reshape(-1, 1)
        self.b = np.mean(-y[self.support] + wx_sup)

        self.y = y
        self.alpha = alpha
        self.X_norm = X_norm

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).     
        
        """
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        K = self.kernel(self.X_norm, X_norm)
        wx = np.sum(self.alpha * self.y * K, axis=0).reshape(-1, 1)
        ans = (wx - self.b)
        return np.ravel(ans)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))


# def main():
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
#
#     def generate_dataset(moons=False):
#         if moons:
#             X, y = make_moons(1000, noise=0.075, random_state=42)
#             return X, 2 * y - 1
#         X, y = make_blobs(1000, 2, centers=[[0, 0], [-4, 2], [3.5, -2.0], [3.5, 3.5]], random_state=42)
#         y = 2 * (y % 2) - 1
#         return X, y
#         # return make_classification(1000, 2, 2, 0, flip_y=0.001, class_sep=1.2, scale=0.9, random_state=42)
#
#     #false - разделимые
#     X, y = generate_dataset(False)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
#
#     svm = KernelSVM(1, kernel=get_polynomial_kernel(0, 1))
#     #svm = KernelSVM(1, kernel=get_gaussian_kernel(0.4))
#     svm.fit(X_train, y_train)
#     y_pred = svm.predict(X_test)
#     acc = accuracy_score(y_pred, y_test)
#     print(acc)
#
#
#     # X = np.ones((3, 2))*5
#     # y = np.array([[1, 2],[3,4]])
#     # ker_g = get_gaussian_kernel(sigma=1.)
#     # ker_p = get_polynomial_kernel()
#     # f_p = ker_p(X, y)
#     # f_g = ker_g(X, y)
#     # f_g
#
# if __name__ == "__main__":
#     main()
