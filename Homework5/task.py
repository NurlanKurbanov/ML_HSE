import numpy as np
import copy
from typing import List, NoReturn
import torch
from torch import nn
import torch.nn.functional as F
from math import ceil

# Task 1


class Module:
    """
    Абстрактный класс. Его менять не нужно. Он описывает общий интерфейс взаимодествия со слоями нейронной сети.
    """
    def forward(self, x):
        pass
    
    def backward(self, d):
        pass
        
    def update(self, alpha):
        pass
    
    
class Linear(Module):
    """
    Линейный полносвязный слой.
    """
    def __init__(self, in_features: int, out_features: int):
        np.random.seed(0)
        self.w = np.random.normal(0, 1/np.sqrt(in_features+out_features), [in_features, out_features])
        self.x = None
        np.random.seed(1)
        self.bias = np.random.normal(0, 1/np.sqrt(in_features+out_features), [1, out_features])
        self.dL_dw = None
        self.dL_db = None
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int 
            Размер выхода.
    
        Notes
        -----
        W и b инициализируются случайно.
        """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).

        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        if x.ndim == 1:
            x = np.array([x])
        self.x = x
        return (self.x).dot(self.w) + self.bias

    
    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        if d.ndim == 1:
            d = np.array([d])

        new_d = d.dot((self.w).T)          # k_i
        self.dL_dw = (self.x.T).dot(d)     # k_i*k_j = i_j
        self.dL_db = np.sum(d, axis=0)
        return new_d

    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        self.w -= alpha * self.dL_dw # /self.x.shape[0]
        self.bias -= alpha * self.dL_db # /self.x.shape[0]
    

class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU. Данная функция возвращает новый массив, в котором значения меньшие 0 заменены на 0.
    """
    def __init__(self):
        self.der = None
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
    
        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self.x = x
        x_pos = (x >= 0).astype(int)
        self.der = x_pos
        return x*x_pos
        
    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        new_d = d * self.der
        return new_d


def SoftMaxCrossEntropyLoss(batch, y_batch_one_hot=None):
    str_max = np.array([np.max(batch, axis=1)]).T
    exps = np.exp(batch - str_max)                    # чтобы не было переполнения
    exps_str_sum = np.array([np.sum(exps, axis=1)]).T
    stable_softmax = exps / exps_str_sum
    if y_batch_one_hot is not None:                   # когда делаем fit нужно только значение d
        d = (stable_softmax - y_batch_one_hot)  #/ batch.shape[0]  # need to divide?
        return d
    return stable_softmax                             # когда предсказываем нужны только вероятности

# Task 2


class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01, batch_size: int = 32):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и 
            описывающий слои нейронной сети. 
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обучения.
        alpha : float
            Cкорость обучения.
        batch_size : int
            Размер батча, используемый в процессе обучения.
        """
        self.modules = modules
        self.epochs_num = epochs
        self.learning_rate = alpha
        self.batch_size = batch_size
            
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох. 
        В каждой эпохе необходимо использовать cross-entropy loss для обучения, 
        а так же производить обновления не по одному элементу, а используя батчи (иначе обучение будет нестабильным и полученные результаты будут плохими.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        """
        y_one_hot = np.zeros((len(y), max(y)+1))
        y_one_hot[np.arange(len(y)), y] = 1

        batches_num = ceil(len(X) / self.batch_size)

        for _ in range(self.epochs_num):
            idx = np.random.permutation(len(X))
            X, y_one_hot = X[idx], y_one_hot[idx]

            batches = np.array_split(X, batches_num)
            batches_y = np.array_split(y_one_hot, batches_num)

            for batch, batch_y in list(zip(batches, batches_y)):
                batch_after_layer = batch
                for module in self.modules:
                    batch_after_layer = module.forward(batch_after_layer)

                d = SoftMaxCrossEntropyLoss(batch_after_layer, batch_y)

                for module in self.modules[::-1]:
                    d = module.backward(d)

                for module in self.modules:
                    module.update(self.learning_rate)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)
        
        """
        batch = X
        for module in self.modules:
            batch = module.forward(batch)

        stable_softmax = SoftMaxCrossEntropyLoss(batch)
        return stable_softmax

    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Вектор предсказанных классов
        
        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)
    
# Task 3

classifier_moons = MLPClassifier([Linear(2,8),ReLU(),Linear(8,2)]) # Нужно указать гиперпараметры
classifier_blobs = MLPClassifier([Linear(2,8),ReLU(),Linear(8,3)]) # Нужно указать гиперпараметры


# Task 4

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 30),
            nn.ReLU(),
            nn.Linear(30, 10),

            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).cuda()

    def load_model(self):
        """
        Используйте torch.load, чтобы загрузить обученную модель
        Учтите, что файлы решения находятся не в корне директории, поэтому необходимо использовать следующий путь:
        `__file__[:-7] +"model.pth"`, где "model.pth" - имя файла сохраненной модели `
        """
        torch.load(__file__[:-7] + "Model.pth", map_location=lambda storage, loc: storage)

    def save_model(self):
        """
        Используйте torch.save, чтобы сохранить обученную модель
        """
        torch.save(self.network, "Model.pth")


def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    """
    Cчитает cross-entropy.

    Parameters
    ----------
    X : torch.Tensor
        Данные для обучения.
    y : torch.Tensor
        Метки классов.
    model : Model
        Модель, которую будем обучать.

    """
    y_pred = model.forward(X)
    loss = F.cross_entropy(y_pred, y)
    return loss
