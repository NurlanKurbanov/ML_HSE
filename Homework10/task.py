import numpy as np
import pandas
import random
import copy
# Task 1


def cyclic_distance(points, dist):
    total = 0
    prev = points[-1]
    for point in points:
        total += dist(prev, point)
        prev = point
    return total


def l2_distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))


def l1_distance(p1, p2):
    return np.sum(np.abs(p1-p2))


# Task 2

class HillClimb:
    def __init__(self, max_iterations, dist):
        self.max_iterations = max_iterations
        self.dist = dist # Do not change
    
    def optimize(self, X):
        return self.optimize_explain(X)[-1]

    def optimize_explain(self, X):
        permuts = []
        n_points = X.shape[0]

        perm_cur = np.array(range(n_points))
        permuts.append(perm_cur)
        path_len_cur = cyclic_distance(X[perm_cur], self.dist)

        for _ in range(self.max_iterations):
            neighbours = self._get_neighbours(perm_cur)
            perm_new, path_len_new = self._get_best_neighbour(X, neighbours, path_len_cur)

            if perm_new is None:
                break

            permuts.append(perm_new)
            perm_cur = perm_new
            path_len_cur = path_len_new

        return permuts

    def _get_neighbours(self, perm):
        neighbours = []
        for i in range(len(perm)):
            for j in range(i + 1, len(perm)):
                neighbour = copy.deepcopy(perm)
                neighbour[i] = perm[j]
                neighbour[j] = perm[i]
                neighbours.append(neighbour)
        return neighbours

    def _get_best_neighbour(self, X, neighbours, cur_best):
        best_neighbour = None
        best_path_len = cur_best

        for neighbour in neighbours:
            neighb_path_len = cyclic_distance(X[neighbour], self.dist)
            if neighb_path_len < best_path_len:
                best_path_len = neighb_path_len
                best_neighbour = neighbour
        return best_neighbour, best_path_len


# Task 3

class Genetic:
    def __init__(self, iterations, population, survivors, distance):
        self.pop_size = population
        self.surv_size = survivors
        self.dist = distance
        self.iters = iterations
    
    def optimize(self, X):
        last_population = self.optimize_explain(X)[-1]
        path_lens = [cyclic_distance(X[path], self.dist) for path in last_population]
        best_path_idx = np.argmin(path_lens)
        best_path = last_population[best_path_idx]
        return best_path

    def optimize_explain(self, X):
        n_points = X.shape[0]
        evolution = []

        population = np.array([np.random.permutation(n_points) for _ in range(self.pop_size)])

        for _ in range(self.iters//4):
            best_parents = self._get_best_parents(population, X)
            population_after_crossover = self._crossover(best_parents)
            population_mutated = self._mutation(population_after_crossover)
            evolution.append(population_mutated)

            #population = copy.deepcopy(population_mutated)
            #population = np.copy(population_mutated)
            population = population_mutated

        return evolution

    def _get_best_parents(self, population, X):
        path_lens = [cyclic_distance(X[path], self.dist) for path in population]
        best_parents_idx = np.argsort(path_lens)[:self.surv_size]
        best_parents = population[best_parents_idx]
        return best_parents

    def _crossover(self, survivors):
        #next_generation = survivors
        next_generation = [x for x in survivors]

        for _ in range(self.pop_size-self.surv_size):
            par1_idx, par2_idx = np.random.choice(survivors.shape[0], size=2, replace=False)
            left_cross_bound, right_cross_bound = np.random.choice(len(survivors[0]), size=2, replace=False)
            son_left_part = survivors[par1_idx][left_cross_bound:(right_cross_bound + 1)]
            son_right_part = [point for point in survivors[par2_idx] if point not in son_left_part]
            son = np.concatenate((son_left_part, son_right_part))
            next_generation.append(son)
        return np.array(next_generation, dtype="int")

    def _mutation(self, population):
        prob = 0.4
        n_mutated = int(len(population)*prob)
        idx_mutated = np.random.choice(len(population), size=n_mutated, replace=False)
        for ind in idx_mutated:
            i, j = np.random.choice(population.shape[1], size=2, replace=False)
            population[ind][i], population[ind][j] = population[ind][j], population[ind][i]
        return population


# Task 4
from nltk.stem.snowball import SnowballStemmer
import re


class BoW:
    def _clear(self, X, vocab=False):
        #stemmer = SnowballStemmer("english")
        X_stem = []

        for i in range(X.shape[0]):
            s = X[i].lower()
            s = re.sub(r"[^a-z]", ' ', s)
            s = (s.strip()).split()
            #s = [stemmer.stem(el) for el in s]
            if vocab:
                X_stem += s
            else:
                X_stem += [s]

        return X_stem

    def __init__(self, X: np.ndarray, voc_limit: int = 1000):
        """
        Составляет словарь, который будет использоваться для векторизации предложений.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ), 
            по которому будет составляться словарь.
        voc_limit : int
            Максимальное число слов в словаре.

        """
        self.voc_limit = voc_limit

        X_stem = self._clear(X, vocab=True)
        words, words_freq = np.unique(X_stem, return_counts=True)
        words = words[np.argsort(-words_freq)]
        #words = words[int(0.01*len(words)):]
        words = words[:voc_limit]

        self.vocab = {word: word_idx for word_idx, word in enumerate(words)}

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Векторизует предложения.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ), 
            который необходимо векторизовать.
        
        Return
        ------
        np.ndarray
            Матрица векторизованных предложений размерности (n_sentences, vocab_size)
        """
        X_stem = self._clear(X)

        vectorized = np.zeros((X.shape[0], self.voc_limit))

        for i, text in enumerate(X_stem):
            words, words_freq = np.unique(text, return_counts=True)
            for freq_idx, word in enumerate(words):
                if word in self.vocab:
                    word_idx = self.vocab[word]
                    vectorized[i, word_idx] += words_freq[freq_idx]
        return vectorized

# Task 5

class NaiveBayes:
    def __init__(self, alpha: float):
        """
        Parameters
        ----------
        alpha : float
            Параметр аддитивной регуляризации.
        """
        self.alpha = alpha
        self.classes = None
        self.y_prob = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Оценивает параметры распределения p(x|y) для каждого y.
        """
        reg = X.shape[1]

        y_uniq, y_freq = np.unique(y, return_counts=True)
        self.classes = y_uniq
        self.y_prob = y_freq / float(y.shape[0])

        self.prob_x_y = dict()
        for y_ in y_uniq:
            count_x_y = np.sum(X[y == y_], axis=0, dtype=np.float64)
            self.prob_x_y[y_] = (count_x_y + float(self.alpha)) / (np.sum(count_x_y, dtype=np.float64)
                                                                   + float(reg*self.alpha))

    def predict(self, X: np.ndarray) -> list:
        """
        Return
        ------
        list
            Предсказанный класс для каждого элемента из набора X.
        """
        return [self.classes[i] for i in np.argmax(self.log_proba(X), axis=1)]
    
    def log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return
        ------
        np.ndarray
            Для каждого элемента набора X - логарифм вероятности отнести его к каждому классу. 
            Матрица размера (X.shape[0], n_classes)
        """
        log_proba = np.zeros((X.shape[0], len(self.classes)))

        for j, y_ in enumerate(self.classes):
            #prob = np.log(self.prob_x_y[y_]) * X
            prob = np.multiply(np.log(self.prob_x_y[y_], dtype=np.float64), X, dtype=np.float64)
            log_proba[:, j] += np.sum(prob, axis=1, dtype=np.float64)

        return log_proba + np.log(self.y_prob, dtype=np.float64)

