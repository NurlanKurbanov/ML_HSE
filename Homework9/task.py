import numpy as np
from catboost import CatBoostClassifier
from collections import Counter
import copy
# Task 0


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


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion) -> float:
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

    x1 = (left_y.shape[0] + right_y.shape[0]) * criterion(np.concatenate((left_y, right_y)))
    x2 = left_y.shape[0] * criterion(left_y)
    x3 = right_y.shape[0] * criterion(right_y)
    return x1 - x2 - x3

# Task 1


def bagging(X, y):
    n_samples = X.shape[0]
    idx_in = np.random.choice(n_samples, n_samples, replace=True)
    idx_out = np.where(~np.isin(np.arange(n_samples), idx_in))[0]
    return X[idx_in], y[idx_in], X[idx_out], y[idx_out]


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
    def __init__(self, split_dim: int, split_value: float, left, right):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right


class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = np.sqrt(X.shape[1]).astype('int') if max_features == "auto" else max_features
        self.X = X
        self.y = y
        self.X_bag, self.y_bag, self.X_out, self.y_out = bagging(self.X, self.y)
        self.root = self._make_tree(self.X_bag, self.y_bag, 0)

    def _make_tree(self, X, y, depth):
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
        max_gain = -1
        best_criteria = {'split_feat': None, 'split_feat_val': None, 'is_leaf': True,
                         'left_son_ind': None, 'right_son_ind': None}
        used_feat_ind = np.random.choice(X.shape[1], self.max_features, replace=False)

        for j in used_feat_ind:
            vals = (val for val in np.unique(X[:, j]))
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

    def _traverse(self, x, node):
        if isinstance(node, DecisionTreeLeaf):
            return node.probs

        if x[node.split_dim] < node.split_value:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        # self.X_bag, self.y_bag, self.X_out, self.y_out = bagging(self.X, self.y)
        # root = self._make_tree(self.X_bag, self.y_bag, 0)

        #probs = np.array([self._traverse(x, root) for x in X])
        probs = np.array([self._traverse(x, self.root) for x in X])
        return [max(p.keys(), key=lambda k: p[k]) for p in probs]

    def oob_acc(self, shuffle=None, feat_ind=None):
        if shuffle is None:
            f = self.predict(self.X_out)
            ff = self.y_out.reshape(1,-1)
            f = (f == ff)
            f = np.sum(f)
            return np.sum(self.predict(self.X_out) == self.y_out.reshape(1,-1)) / self.y_out.shape[0]
        else:
            shuffled_idx = np.random.permutation(self.y_out.shape[0])
            X_shuffled = copy.deepcopy(self.X_out)
            X_shuffled[:, feat_ind] = X_shuffled[shuffled_idx, feat_ind]
            f = self.predict(X_shuffled)
            f = (f==self.y_out.reshape(1,-1))
            return np.sum(self.predict(X_shuffled) == self.y_out.reshape(1,-1)) / self.y_out.shape[0]
# Task 2


class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.forest = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTree(X, y, criterion=self.criterion, max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
            self.forest.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.forest]) # row=tree  column=samples
        most_common_vote = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=1, arr=tree_preds.T)
        return most_common_vote

    def get_feat_imp(self):
        n_feat = self.forest[0].X_out.shape[1]
        forest_acc = np.zeros((self.n_estimators, n_feat))
        for i, tree in enumerate(self.forest):
            oob_acc = tree.oob_acc()
            for j in range(n_feat):
                oob_shuffled_acc = tree.oob_acc(shuffle=True, feat_ind=j)
                forest_acc[i, j] = oob_acc - oob_shuffled_acc
        return np.mean(forest_acc, axis=0)


# Task 3

def feature_importance(rfc):
    return rfc.get_feat_imp()


# Task 4

rfc_age = RandomForestClassifier(criterion="gini", max_depth=20, min_samples_leaf=10, max_features="auto", n_estimators=3)
rfc_gender = RandomForestClassifier(criterion="gini", max_depth=20, min_samples_leaf=10, max_features="auto", n_estimators=2)

# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model
catboost_rfc_age = CatBoostClassifier().load_model(f'{__file__[:-7]}/catboost_rfc_age.cbm')
catboost_rfc_gender = CatBoostClassifier().load_model(f'{__file__[:-7]}/catboost_rfc_sex.cbm')




