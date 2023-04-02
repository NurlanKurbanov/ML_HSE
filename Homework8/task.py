import numpy as np
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List
import time
# Task 1


def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    uniq_num_normed = np.unique(x, return_counts=True)[1] / x.shape[0]
    return np.sum(uniq_num_normed * (1 - uniq_num_normed))


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    uniq_num_normed = np.unique(x, return_counts=True)[1] / x.shape[0]
    return -np.sum(uniq_num_normed * np.log2(uniq_num_normed))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    if len(left_y) == 0 or len(right_y) == 0:
        return 0

    x1 = (left_y.shape[0]+right_y.shape[0]) * criterion(np.concatenate((left_y, right_y)))
    x2 = left_y.shape[0] * criterion(left_y)
    x3 = right_y.shape[0] * criterion(right_y)
    return x1 - x2 - x3


# Task 2

class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """
    def __init__(self, ys):
        y_uniq, y_freqs = np.unique(ys, return_counts=True)
        self.y = y_uniq[np.argmax(y_freqs)]
        self.probs = {label: prob for label, prob in list(zip(y_uniq, y_freqs / len(ys)))}


class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value.
    """
    def __init__(self, split_dim: int, split_value: float,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right


# Task 3


class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """
    def __init__(self, criterion : str = "gini",
                 max_depth : Optional[int] = None,
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        if criterion == "gini":
            self.criterion = gini
        else:
            self.criterion = entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        self.root = self._make_tree(X, y, 0)

    def _make_tree(self, X, y, depth):

        #y_uniq_num = len(np.unique(y))
        y_uniq_num = len(np.unique(y))
        samples_num = X.shape[0]


        if self.max_depth is not None:
            if (depth == self.max_depth) or (y_uniq_num == 1) or (samples_num == self.min_samples_leaf):
                return DecisionTreeLeaf(y)
        else:
            if (y_uniq_num == 1) or (samples_num == self.min_samples_leaf):
                return DecisionTreeLeaf(y)

        split_feat, split_feat_val, is_leaf, left_son_ind, right_son_ind = self._get_best_criteria(X, y)

        if is_leaf:
            return DecisionTreeLeaf(y)
        else:
            left_son = self._make_tree(X[left_son_ind, :], y[left_son_ind], depth + 1)
            right_son = self._make_tree(X[right_son_ind, :], y[right_son_ind], depth + 1)

            return DecisionTreeNode(split_feat, split_feat_val, left_son, right_son)

    def _get_best_criteria(self, X, y):
        # max_gain = -1
        # split_feat, split_feat_val = None, None
        # is_leaf = True
        # left_son_ind, right_son_ind = None, None
        #
        # for j in range(X.shape[1]):
        #
        #     vals = np.unique(X[:, j])
        #
        #     for val in vals:
        #         left_str_ind = np.argwhere(X[:, j] < val).flatten()
        #         right_str_ind = np.argwhere(X[:, j] >= val).flatten()
        #         inf_gain = gain(y[left_str_ind], y[right_str_ind], self.criterion)
        #
        #         if ((inf_gain > max_gain) and (len(left_str_ind) >= self.min_samples_leaf)
        #                 and (len(right_str_ind) >= self.min_samples_leaf)):
        #             is_leaf = False
        #             max_gain = inf_gain
        #             split_feat = j
        #             split_feat_val = val
        #             left_son_ind = left_str_ind
        #             right_son_ind = right_str_ind
        #
        # return split_feat, split_feat_val, is_leaf, left_son_ind, right_son_ind
        max_gain = -1
        best_criteria = {'split_feat': None, 'split_feat_val': None, 'is_leaf': True,
                         'left_son_ind': None, 'right_son_ind': None}

        for j in range(X.shape[1]):
            #vals = (val for val in np.round(np.unique(X[:, j])[::3], 1))
            vals = (val for val in np.unique(X[:, j])[::3])
            for val in vals:
                left_son_ind = np.where(X[:, j] < val)[0]
                right_son_ind = np.where(X[:, j] >= val)[0]
                inf_gain = gain(y[left_son_ind], y[right_son_ind], self.criterion)

                if inf_gain > max_gain and len(left_son_ind) >= self.min_samples_leaf and len(
                        right_son_ind) >= self.min_samples_leaf:
                    best_criteria.update({'is_leaf': False, 'split_feat': j, 'split_feat_val': val,
                                          'left_son_ind': left_son_ind, 'right_son_ind': right_son_ind})
                    max_gain = inf_gain

        return best_criteria['split_feat'], best_criteria['split_feat_val'], best_criteria['is_leaf'], \
            best_criteria['left_son_ind'], best_criteria['right_son_ind']

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь
            {метка класса -> вероятность класса}.
        """
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if isinstance(node, DecisionTreeLeaf):
            return node.probs

        if x[node.split_dim] < node.split_value:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X : np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]

# Task 4
task4_dtc = DecisionTreeClassifier(max_depth=6, min_samples_leaf=30)



# def main():
#
#     from sklearn.datasets import make_blobs, make_moons
#     noise = 0.35
#     #X, y = make_moons(1500, noise=noise)
#     #X_test, y_test = make_moons(200, noise=noise)
#     X = np.random.randn(3,4)
#     y = ['a', 'b', 'a']
#     tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30)
#     tree.fit(X, y)
#
#
# if __name__ == '__main__':
#     main()