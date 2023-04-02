import numpy as np
from sklearn.model_selection import train_test_split
import copy
from typing import NoReturn


# Task 1


class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """
        self.w = None
        self.iterations = iterations
        self.labels = None


    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон. 
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        self.labels = np.unique(y)
        y_binary = np.where(y == self.labels[0], -1, 1)

        self.w = np.zeros(X.shape[1] + 1)
        bias_feature = np.ones((X.shape[0], 1))
        X_with_bias = np.append(bias_feature, X, axis=1)

        for _ in range(self.iterations):
            ind_dif = np.where(np.sign(X_with_bias.dot(self.w)) != y_binary)[0]  #индексы, где неравенства
            # self.w += np.sum(X_with_bias[ind_dif]*y_binary[ind_dif].reshape(-1, 1), axis=0)
            self.w += y_binary[ind_dif].dot(X_with_bias[ind_dif])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        bias_feature = np.ones((X.shape[0], 1))
        X_with_bias = np.append(bias_feature, X, axis=1)
        bin_labels = np.sign(X_with_bias.dot(self.w))

        return np.where(bin_labels == -1, self.labels[0], self.labels[1])


# Task 2

class PerceptronBest:

    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """
        self.w = None
        self.iterations = iterations
        self.labels = None
        self.min_error = None
        self.best_w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона, 
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса, 
        при которых значение accuracy было наибольшим.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        self.labels = np.unique(y)
        y_binary = np.where(y == self.labels[0], -1, 1)

        self.w = np.zeros(X.shape[1] + 1)
        bias_feature = np.ones((X.shape[0], 1))
        X_with_bias = np.append(bias_feature, X, axis=1)

        self.min_error = len(y_binary)
        for _ in range(self.iterations):
            ind_dif = np.where(np.sign(X_with_bias.dot(self.w)) != y_binary)[0]
            wrong_predictions = len(ind_dif)
            if wrong_predictions < self.min_error:
                self.min_error = wrong_predictions
                self.best_w = np.copy(self.w)
            # self.w += np.sum(X_with_bias[ind_dif] * y_binary[ind_dif].reshape(-1, 1), axis=0)
            self.w += y_binary[ind_dif].dot(X_with_bias[ind_dif])


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        bias_feature = np.ones((X.shape[0], 1))
        X_with_bias = np.append(bias_feature, X, axis=1)
        bin_labels = np.sign(X_with_bias.dot(self.best_w))

        return np.where(bin_labels == -1, self.labels[0], self.labels[1])


# Task 3


def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.
        
    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    res = np.array([])
    for image in images:
        res = np.append(res, np.sum(np.flip(image, axis=1) == image)/(image.shape[0]*image.shape[1]))
        # res = np.append(res, np.sum(np.flip(image, axis=0) == image)/(image.shape[0]*image.shape[1]))
        res = np.append(res, max(np.sum(image, axis=0)))
    return np.array(res).reshape(-1, 2)



