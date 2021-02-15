import numpy as np
import scipy.linalg as sl
from scipy import special


class Retrospective:
    """
    S-MDL (Retrospective)
    Diffierential MDL Change Statistics
    """

    def __init__(self, h, encoding_func, complexity_func, delta_0=0.05, delta_1=0.05, delta_2=0.05, order=-1):
        """
        Args:
            h: half window size
            encoding_func: encoding function
            complexity_func: parametric complexity of the encoding function
            delta_0: the upper bound on the Type-I error probability of 0th D-MDL
            delta_1: the upper bound on the Type-I error probability of 1st D-MDL
            delta_2: the upper bound on the Type-I error probability of 2nd D-MDL
            order: return which order of D-MDL. if order=-1, return all the statistics.
        """
        self.__h = h
        self.__encoding_func = encoding_func
        self.__complexity_func = complexity_func
        # thresholds are set so that the Type-I error probability is less than
        # delta
        self.__threshold_0 = self.__complexity_func(2 * self.__h) - np.log(delta_0)
        self.__threshold_1 = 2 * self.__complexity_func(self.__h) - np.log(delta_1)
        self.__threshold_2 = 2 * \
            (2 * self.__complexity_func(self.__h) - np.log(delta_2))
        self.__order = order

    def calc_scores(self, X):
        """
        calculate scores

        Args:
            X: input data

        Returns:
            Union[ndarray, list]: scores of the input data
        """
        detector = Prospective(
            h=self.__h, encoding_func=self.__encoding_func, complexity_func=self.__complexity_func)

        if X.ndim == 1:
            X = X.reshape((X.shape[0], 1))

        scores_0 = []
        scores_1 = []
        scores_2 = []

        for i, X_i in enumerate(X):
            score_0, score_1, score_2 = detector.update(X_i)
            if i >= self.__h:
                scores_0.append(score_0)
                scores_1.append(score_1)
                scores_2.append(score_2)

        scores_0 = scores_0 + [np.nan] * self.__h
        scores_1 = scores_1 + [np.nan] * self.__h
        scores_2 = scores_2 + [np.nan] * self.__h

        if self.__order == 0:
            return np.array(scores_0)
        elif self.__order == 1:
            return np.array(scores_1)
        elif self.__order == 2:
            return np.array(scores_2)
        else:
            return [np.array(scores_0), np.array(scores_1), np.array(scores_2)]

    def make_alarms(self, X):
        """
        make alarms with the threshold

        Args:
            X: input data

        Returns:
            Union[ndarray, list]: indice of alarms
        """
        scores_0, scores_1, scores_2 = self.calc_scores(X)
        alarms_0 = np.where(scores_0 >= self.__threshold_0, 1, 0)
        alarms_1 = np.where(scores_1 >= self.__threshold_1, 1, 0)
        alarms_2 = np.where(scores_2 >= self.__threshold_2, 1, 0)

        if self.__order == 0:
            return alarms_0
        elif self.__order == 1:
            return alarms_1
        elif self.__order == 2:
            return alarms_2
        else:
            return [alarms_0, alarms_1, alarms_2]


class Prospective:
    """
    S-MDL (Prospective)
    Diffierential MDL Change Statistics
    """

    def __init__(self, h, encoding_func, complexity_func):
        """
        Args:
            h: half window size
            encoding_func: encoding function
            complexity_func: parametric complexity of the encoding function
        """
        # パラメータ設定
        self.__h = h
        self.__encoding_func = encoding_func
        self.__complexity_func = complexity_func
        self.__stacked_data = np.array([])

    def update(self, x):
        """
        calculate the score of the input datum

        Args:
            x: input datum

        Returns:
            float, float, float: 0th, 1st, 2nd D-MDL
        """
        x = x.reshape((1, x.shape[0]))

        if len(self.__stacked_data) == 0:
            self.__stacked_data = np.copy(x)
        else:
            self.__stacked_data = np.append(self.__stacked_data, x, axis=0)

        # return np.nan if the number of data is less than 2h
        if self.__stacked_data.shape[0] < 2 * self.__h:
            return np.nan, np.nan, np.nan
        else:
            n = len(self.__stacked_data)
            # code length with no change
            stat_nochange = self.__encoding_func(self.__stacked_data)
            # code length with change at time t, t+1, and t-1
            stat_t = self.__encoding_func(self.__stacked_data[
                                          0:n // 2]) + self.__encoding_func(self.__stacked_data[n // 2:])
            stat_tp = self.__encoding_func(self.__stacked_data[
                                           0:n // 2 + 1]) + self.__encoding_func(self.__stacked_data[n // 2 + 1:])
            stat_tm = self.__encoding_func(self.__stacked_data[
                                           0:n // 2 - 1]) + self.__encoding_func(self.__stacked_data[n // 2 - 1:])

            score_0 = stat_nochange - stat_t
            score_1 = (stat_nochange - stat_tp) - (stat_nochange - stat_t)
            score_2 = ((stat_nochange - stat_tp) - (stat_nochange - stat_t)) - \
                ((stat_nochange - stat_t) - (stat_nochange - stat_tm))
            self.__stacked_data = np.delete(self.__stacked_data, obj=0, axis=0)

            return score_0, score_1, score_2
