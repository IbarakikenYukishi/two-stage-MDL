import sys
sys.path.append("../")
import numpy as np
import scipy.linalg as sl
from scipy import special
from copy import deepcopy


class Retrospective:
    """
    S-MDL (Retrospective)
    Diffierential MDL Change Statistics
    """

    def __init__(self, retrospective_first, retrospective_second):
        """
        Args:
            retrospective_first: SDMDL for the first stage
            retrospective_second: SDMDL for the second stage
        """

        self.__h_1 = retrospective_first.h
        self.__h_2 = retrospective_second.h
        self.__threshold=retrospective_second.threshold_0
        self.__retrospective_first = retrospective_first
        self.__retrospective_second = retrospective_second

    def calc_scores(self, X):
        """
        calculate scores

        Args:
            X: input data

        Returns:
            Union[ndarray, list]: scores of the input data
        """

        # data index: from 0 to n - 1
        # S-MDL at time t uses the data from time t- (h_1 - 1) to t + h_1
        # Therefore, S-MDL is defined from time h_1 - 1 to time n - h_1 - 1
        # r_t is defined from time h_1 to time n - h_1 - 1
        # FW2S-MDL is defined from time h_1 + h_2 - 1 to time n - h_1 - h_2 - 1

        # 1st stage scores
        n = len(X)
        eps = 1e-12
        probs = self.__retrospective_first.calc_prob(X)
        probs = probs[self.__h_1 - 1: n - self.__h_1]
        prob_lengths = -np.log(probs)
        prob_lengths = np.where(prob_lengths < eps, eps, prob_lengths)
        prob_lengths = np.array(
            [np.nan] * (self.__h_1 - 1) + list(prob_lengths) + [np.nan] * self.__h_1)

        # growth rates
        growth_rates = np.zeros(n - 2 * self.__h_1)
        for i in range(self.__h_1, n - self.__h_1):
            growth_rates[i - self.__h_1] = prob_lengths[i] / \
                prob_lengths[i - 1] - 1

        probs = np.array([np.nan] * (self.__h_1  - 1) + list(probs) + [np.nan] * self.__h_1)
        # 2nd stage scores
        scores = self.__retrospective_second.calc_scores(growth_rates)
        scores = np.array([np.nan] * self.__h_1 + list(scores) + [np.nan] * self.__h_1)

        # set np.nan if the probability decreases
        for j in range(self.__h_1 + self.__h_2 - 1, n - self.__h_1 - self.__h_2):
            if np.mean(probs[j - self.__h_2:j]) > np.mean(probs[j:j + self.__h_2]):
                scores[j] = np.nan

        return scores

    def make_alarms(self, X):
        """
        make alarms with the threshold

        Args:
            X: input data

        Returns:
            ndarray: indice of alarms
        """
        scores = self.calc_scores(X)

        # ignore warnings made by np.nan
        with np.errstate(invalid='ignore'):
            alarms = np.greater(
                scores,
                self.__threshold
            ).astype(int)

        alarms = np.where(alarms == 1)[0]

        return alarms
