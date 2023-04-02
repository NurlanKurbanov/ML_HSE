import numpy as np

# Task 1


def mse(y_true: np.ndarray, y_predicted: np.ndarray):
    # return np.square(np.subtract(y_true, y_predicted)).mean()
    return ((y_true-y_predicted)**2).mean()


def r2(y_true: np.ndarray, y_predicted: np.ndarray):
    u = np.sum(np.square(np.subtract(y_true, y_predicted)))
    y_m = np.sum(y_true)/len(y_true)
    v = np.sum(np.square(y_m - y_true))
    return 1-u/v

# Task 2


class NormalLR:
    def __init__(self):
        self.weights = None # Save weights here
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        # np.random.shuffle(X)
        new_feature = np.ones((len(X), 1))
        new_X = np.append(new_feature, X, axis=1)
        self.weights = np.linalg.inv((new_X.T).dot(new_X)).dot(new_X.T).dot(y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        new_feature = np.ones((len(X), 1))
        new_X = np.append(new_feature, X, axis=1)
        return np.array(new_X.dot(self.weights), dtype=float)
    
# Task 3

class GradientLR:
    def __init__(self, alpha:float, iterations=10000, l=0.):
        self.weights = None # Save weights here
        self.alpha = alpha
        self.iterations = iterations
        self.l = l

    def fit(self, X:np.ndarray, y:np.ndarray):
        self.weights = np.zeros(np.shape(X)[1] + 1)

        new_feature = np.ones((len(X), 1))
        new_X = np.append(new_feature, X, axis=1)

        # for i in range(new_X.shape[1]):
        #     new_X[:, i] = new_X[:, i] / max(new_X[:, i])

        for _ in range(self.iterations):
            dw = 2/len(new_X) * ((new_X.T).dot(new_X.dot(self.weights) - y)) + self.l * np.sign(self.weights)
            self.weights -= self.alpha * dw

    def predict(self, X:np.ndarray):
        new_feature = np.ones((len(X), 1))
        new_X = np.append(new_feature, X, axis=1)

        # for i in range(new_X.shape[1]):
        #     new_X[:, i] = new_X[:, i] / max(new_X[:, i])

        return new_X.dot(self.weights)

# Task 4


def get_feature_importance(linear_regression):
    return abs(linear_regression.weights[1:])


def get_most_important_features(linear_regression):
    return np.argsort(abs(linear_regression.weights[1:]))[::-1]
