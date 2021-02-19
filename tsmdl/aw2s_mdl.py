import numpy as np
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
        #self.__h_2 = retrospective_second.h
        self.__retrospective_first = retrospective_first
        self.__retrospective_second = retrospective_second

    def calc_all_stats(self, X):
        """
        calculate all statistics

        Args:
            X: input data

        Returns:
            ndarray, ndarray, ndarray, ndarray: alarms, scores, cutpoints, and window sizes
        """

        # data index: from 0 to n - 1
        # S-MDL at time t uses the data from time t- (h_1 - 1) to t + h_1
        # Therefore, S-MDL is defined from time h_1 - 1 to time n - h_1 - 1
        # r_t is defined from time h_1 to time n - h_1 - 1

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
        # 2nd stage statistics
        alarms, scores, cutpoints, window_size = self.__retrospective_second.calc_all_stats(
            growth_rates)
        window_size = np.array([np.nan] * self.__h_1 +
                               list(window_size) + [np.nan] * self.__h_1)
        for i in range(3):
            scores[i] = np.array([np.nan] * self.__h_1 +
                                 list(scores[i]) + [np.nan] * self.__h_1)
            alarms[i] += self.__h_1
            cutpoints[i] += self.__h_1

        alarms_processed=[]
        cutpoints_processed=[]

        # delete alarms and cutpoints if the probability decreases
        for i in range(len(alarms[0])):
            if i == 0:
                start = 0
            else:
                start = int(alarms[0][i - 1] + 1)
            cutpoint = int(cutpoints[0][i])
            end = int(alarms[0][i])
            if np.mean(probs[start:cutpoint+1]) <= np.mean(probs[cutpoint+1:end+1]):
                alarms_processed.append(alarms[0][i])
                cutpoints_processed.append(cutpoints[0][i])

        alarms_processed=np.array(alarms_processed)
        cutpoints_processed=np.array(cutpoints_processed)        

        return alarms_processed, scores[0], cutpoints_processed, window_size

    def calc_scores(self, X):
        """
        Calculate scores.

        Args:
            X: input data

        Returns:
            ndarray: scores of the input data
        """
        _, max_scores, _, _ = self.calc_all_stats(X)

        return max_scores

    def make_alarms(self, X):
        """
        Make alarms with the threshold.

        Args:
            X: input data

        Returns:
            ndarray: indice of alarms
        """
        alarms, _, _, _ = self.calc_all_stats(X)

        return alarms
