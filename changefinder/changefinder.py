# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp


class _SDAR_1Dim(object):
    """
    Sequential discounting algorithm
    """

    def __init__(self, r, order):
        """
        Args:
            r: sequential discounting coefficient
            order: AR model's dimension
        """
        self.__r = r
        self.__order = order
        # initialize parameters
        self.__mu = np.random.random()
        self.__sigma = np.random.random()
        self.__cor = np.random.random(self.__order + 1) / 100.0

    def update(self, x, sub_sequence):
        """
        update the SDAR model

        Args:
            x: a datum
            sub_sequence: sub-time-series data

        Returns:
            (float, ndarray): outlier score
        """
        assert len(
            sub_sequence) >= self.__order, "the length of sub-sequence must be order or more"
        sub_sequence = np.array(sub_sequence)
        # update the mean
        self.__mu = self.__r * x + (1.0 - self.__r) * self.__mu

        # update the covariance vector
        for i in range(1, self.__order + 1):
            self.__cor[i] = (1 - self.__r) * self.__cor[i] + self.__r * \
                (x - self.__mu) * (sub_sequence[-i] - self.__mu)
        self.__cor[0] = (1 - self.__r) * self.__cor[0] + \
            self.__r * (x - self.__mu) ** 2

        # update parameters in AR model
        what = self._levinson_durbin_algorithm(self.__cor, self.__order)

        # prediction
        xhat = np.dot(-what[1:], (sub_sequence[::-1] - self.__mu)) + self.__mu

        # update sigma
        self.__sigma = self.__r * (x - xhat) ** 2 + \
            (1 - self.__r) * self.__sigma

        # outlier score
        outlier_score = (0.5 * (x - xhat)**2) / self.__sigma + \
            0.5 * (np.log(2 * np.pi) + np.log(self.__sigma))

        return outlier_score

    def _levinson_durbin_algorithm(self, r, ar_order):
        """
        Solve a Yule-Walker equation by the Levinson-Durbin algorithm.

        Args:
            r: the RHS of a Yule-Walker equation
            ar_order: AR model's dimension

        Returns:
            float: coefficient
        """

        coef = np.zeros(ar_order + 1, dtype=np.float64)
        error = np.zeros(ar_order + 1, dtype=np.float64)

        coef[0] = 1.0
        coef[1] = - r[1] / r[0]
        error[1] = r[0] + r[1] * coef[1]

        for k in range(1, ar_order):
            tau = 0.0
            for j in range(k + 1):
                tau -= coef[j] * r[k + 1 - j]
            tau /= error[k]

            U = [1]
            U.extend([coef[i] for i in range(1, k + 1)])
            U.append(0)

            V = [0]
            V.extend([coef[i] for i in range(k, 0, -1)])
            V.append(1)

            coef = np.array(U) + tau * np.array(V)
            error[k + 1] = error[k] * (1.0 - tau**2)

        return coef


class Retrospective():
    """
    ChangeFinder (Retrospective)
    """

    def __init__(self, r=0.5, order=1, smooth=7, threshold=1):
        """
        Args:
            r: sequential discounting coefficient.
            order: AR model's order.
            smooth: smoothing window size. The second stage's window size is smooth/2 to reduce hyper parameters
            threshold: threshold for alarms.
        """
        assert order > 0, "order must be 1 or more."
        assert smooth > 2, "term must be 3 or more."
        self.__smooth = smooth
        self.__order = order
        self.__r = r
        self.__threshold = threshold

    def calc_scores(self, X):
        """
        calculate scores

        Args:
            X: input data

        Returns:
            ndarray: scores of the input data
        """
        detector = Prospective(self.__r, self.__order, self.__smooth)
        scores = []
        for i in X:
            score = detector.update(i)
            scores.append(score)
        return np.array(scores)

    def make_alarms(self, X):
        """
        make alarms with the threshold

        Args:
            X: input data

        Returns:
            ndarray: binarized scores
        """
        scores = self.calc_scores(X)
        alarms = np.where(scores >= self.__threshold, 1, 0)
        return alarms


class Prospective():
    """
    ChangeFinder (Prospective)
    """

    def __init__(self, r=0.5, order=1, smooth=7):
        """
        Args:
            r: sequential discounting coefficient
            order: AR model's dimension
            smooth: second stage's smoothing window size
        """
        assert order > 0, "order must be 1 or more."
        assert smooth > 2, "term must be 3 or more."
        self.__smooth_first = smooth
        self.__smooth_second = int(round(self.__smooth_first / 2.0))
        self.__order = order
        self.__r = r
        self.__sub_sequence = []
        self.__first_scores = []
        self.__smoothed_scores = []
        self.__second_scores = []
        self.__sdar_first = _SDAR_1Dim(r, self.__order)
        self.__sdar_second = _SDAR_1Dim(r, self.__order)

    def update(self, x):
        """
        calculate the score of the input datum

        Args:
            x: input datum

        Returns:
            float: score of the input datum
        """
        if len(self.__sub_sequence) == self.__order:  # first stage
            first_score = self.__sdar_first.update(x, self.__sub_sequence)
            self._add_datum(first_score, self.__first_scores,
                            self.__smooth_first)
        self._add_datum(x, self.__sub_sequence, self.__order)

        second_smoothed = None
        if len(self.__first_scores) == self.__smooth_first:  # smoothing
            second_smoothed = self._smoothing(self.__first_scores)

        if second_smoothed and len(self.__smoothed_scores) == self.__order:  # secondstage
            second_score = self.__sdar_second.update(
                second_smoothed, self.__smoothed_scores)
            self._add_datum(second_score,
                            self.__second_scores, self.__smooth_second)
        if second_smoothed:
            self._add_datum(second_smoothed,
                            self.__smoothed_scores, self.__order)

        if len(self.__second_scores) == self.__smooth_second:
            return self._smoothing(self.__second_scores)
        else:
            return np.nan

    def _add_datum(self, datum, sub_sequence, size):
        """
        append a datum and pop the oldest datum

        Args:
            one: the input data
            sub_sequence: time-series data
            size: time-series data's maximum tolerance length
        """
        sub_sequence.append(datum)
        if len(sub_sequence) == size + 1:
            sub_sequence.pop(0)

    def _smoothing(self, sub_sequence):
        """
        smoothing

        Args:
            sub_sequence: input time-series data

        Returns:
            float: smoothed score        
        """
        return sum(sub_sequence) / float(len(sub_sequence))
