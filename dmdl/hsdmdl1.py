import numpy as np
import scipy.linalg as sl
from scipy import special
from collections import deque
import functools as fts


class Retrospective:
    """
    Hierarchical Sequential D-MDL algorithm with SCAW1 (Prospective)
    """

    def __init__(
        self,
        encoding_func,
        d,
        min_datapoints=1,
        delta_0=0.05,
        delta_1=0.05,
        delta_2=0.05,
        how_to_drop='all',
        order=-1,
        reliability=True
    ):
        """
        Args:
            encoding_func: encoding function
            d: dimension of the parametric class
            min_datapoints: minimum number of data points to calculate the NML code length
            delta_0: upper bound on the Type-I error probability of 0th D-MDL
            delta_1: upper bound on the Type-I error probability of 1st D-MDL
            delta_2: upper bound on the Type-I error probability of 2nd D-MDL
            how_to_drop: cut: drop the data before the optimal cut point, all: drop the all data
            order: determine which order of D-MDL will be calculated. Note that calculate all the D-MDLs
            if order==-1.
            reliability: use asymptotic reliability or not
        """
        self.__encoding_func = encoding_func
        self.__d = d
        self.__min_datapoints = min_datapoints
        self.__delta_0 = delta_0
        self.__delta_1 = delta_1
        self.__delta_2 = delta_2
        self.__how_to_drop = how_to_drop
        self.__buckets = deque([])
        self.__order = order
        self.__reliability = reliability
        assert how_to_drop in ('cutpoint', 'all')

    def calc_all_stats(self, X):
        """
        Calculate alarms, scores, cutpoints, and window sizes.

        Args:
            X: input data

        Returns:
            list, list, list, ndarray: alarms, scores, cutpoints, and window sizes
        """

        detector = Prospective(
            encoding_func=self.__encoding_func,
            d=self.__d,
            min_datapoints=self.__min_datapoints,
            delta_0=self.__delta_0,
            delta_1=self.__delta_1,
            delta_2=self.__delta_2,
            how_to_drop=self.__how_to_drop,
            order=self.__order,
            reliability=self.__reliability
        )

        # alarms
        alarms_0 = []
        alarms_1 = []
        alarms_2 = []
        # maximum scores
        max_scores_0 = []
        max_scores_1 = []
        max_scores_2 = []
        # optimal cut points
        cutpoints_0 = []
        cutpoints_1 = []
        cutpoints_2 = []
        # window sizes
        window_sizes = []

        for i, X_i in enumerate(X):
            alarms, max_scores, cutpoints, window_size = detector.update(
                X_i)
            alarms_0.append(alarms[0])
            alarms_1.append(alarms[1])
            alarms_2.append(alarms[2])
            max_scores_0.append(max_scores[0])
            max_scores_1.append(max_scores[1])
            max_scores_2.append(max_scores[2])
            cutpoints_0.append(cutpoints[0])
            cutpoints_1.append(cutpoints[1])
            cutpoints_2.append(cutpoints[2])
            window_sizes.append(window_size)

        return [np.array(alarms_0), np.array(alarms_1), np.array(alarms_2)], [np.array(max_scores_0), np.array(max_scores_1), np.array(max_scores_2)], [np.array(cutpoints_0), np.array(cutpoints_1), np.array(cutpoints_2)], np.array(window_sizes)

    def calc_scores(self, X):
        """
        Calculate scores. Return self.__order of D-MDL. 
        If self.__order==-1, return all order of D-MDL as a list.

        Args:
            X: input data

        Returns:
            Union[ndarray, list]: scores of the input data
        """
        _, max_scores, _, _ = self.calc_all_stats(X)
        if self.__order != -1:
            return max_scores[self.__order]
        else:
            return max_scores

    def make_alarms(self, X):
        """
        Make alarms with the threshold. Return self.__order of D-MDL. 
        If self.__order==-1, return all order of D-MDL as a list.

        Args:
            X: input data

        Returns:
            Union[ndarray, list]: binarized scores
        """
        alarms, _, _, _ = self.calc_all_stats(X)
        if self.__order != -1:
            return alarms[self.__order]
        else:
            return alarms


class Prospective:
    """
    Hierarchical Sequential D-MDL algorithm with SCAW1 (Prospective)
    """

    def __init__(
        self,
        encoding_func,
        d,
        min_datapoints=1,
        delta_0=0.05,
        delta_1=0.05,
        delta_2=0.05,
        how_to_drop='all',
        order=-1,
        reliability=True
    ):
        self.__encoding_func = encoding_func
        self.__d = d
        self.__min_datapoints = min_datapoints
        self.__delta_0 = delta_0
        self.__delta_1 = delta_1
        self.__delta_2 = delta_2
        self.__how_to_drop = how_to_drop
        self.__window = deque([])
        self.__order = order
        self.__reliability = reliability
        assert how_to_drop in ('cutpoint', 'all')

    def _combine(self, x):
        """
        combine datum to the window

        Args:
            x: input datum
        """

        newT = self._suffi_stats(x)

        if len(self.__window) == 0:
            self.__window.append([newT[0], newT[1], newT[0], newT[1]])
        else:
            sum_suffi_stat = self.__window[-1]
            self.__window.append([newT[0], newT[1], sum_suffi_stat[
                                 2] + newT[0], sum_suffi_stat[3] + newT[1]])

    def update(self, x):
        """
        calculate the score of the input datum

        Args:
            x: input datum

        Returns:
            list, list, list, ndarray: alarms, scores, cutpoints, window size
        """

        self._combine(x)  # combine new datum

        # the number of data contained in the window
        n = len(self.__window)

        # alarms for change and change signs
        alarm_0 = 0
        alarm_1 = 0
        alarm_2 = 0

        # cut points for change and change signs
        cutpoint_0 = np.nan
        cutpoint_1 = np.nan
        cutpoint_2 = np.nan

        # calculate all the order of D-MDL
        scores_0, scores_1, scores_2 = self._calc_stats()

        # calculate maximum scores of each order of D-MDL
        if np.isnan(scores_0).all():
            max_score_0 = np.nan
        else:
            max_score_0 = np.nanmax(scores_0)

        if np.isnan(scores_1).all():
            max_score_1 = np.nan
        else:
            max_score_1 = np.nanmax(scores_1)

        if np.isnan(scores_2).all():
            max_score_2 = np.nan
        else:
            max_score_2 = np.nanmax(scores_2)

        # max_score_0 is np.nan means the number of data points is not
        # sufficient to compute 0th D-MDL
        if np.isnan(max_score_0):
            ret_alarms = [alarm_0, alarm_1, alarm_2]
            ret_scores = [max_score_0 / n, max_score_1 / n, max_score_2 / n]
            ret_cutpoints = [cutpoint_0, cutpoint_1, cutpoint_2]
            return ret_alarms, ret_scores, ret_cutpoints, n

        # 0th alarm
        if max_score_0 >= self._calculate_threshold(nn=len(self.__window) - 1, n=n, order=0):
            alarm_0 = 1

        # 1st alarm
        if np.isnan(max_score_1) == False and max_score_1 >= self._calculate_threshold(nn=len(self.__window) - 1, n=n, order=1):
            cutpoint_1 = np.nanargmax(scores_1)
            alarm_1 = 1

        # 2nd alarm
        if np.isnan(max_score_2) == False and max_score_2 >= self._calculate_threshold(nn=len(self.__window) - 1, n=n, order=2):
            cutpoint_2 = np.nanargmax(scores_2)
            alarm_2 = 1

        if alarm_0 == 1:  # if 0th alarm was raised
            cutpoint_0 = np.nanargmax(scores_0)
            for j in range(0, cutpoint_0 + 1):
                self.__window.popleft()
            capacity_sum = len(self.__window)

            ret_alarms = [alarm_0, alarm_1, alarm_2]
            ret_scores = [max_score_0 / n, max_score_1 / n, max_score_2 / n]
            ret_cutpoints = [cutpoint_0, cutpoint_1, cutpoint_2]

            if self.__how_to_drop == 'cutpoint':
                return ret_alarms, ret_scores, ret_cutpoints, capacity_sum
            if self.__how_to_drop == 'all':
                self.__window = deque([])
                return ret_alarms, ret_scores, ret_cutpoints, 0
        else:
            ret_alarms = [alarm_0, alarm_1, alarm_2]
            ret_scores = [max_score_0 / n, max_score_1 / n, max_score_2 / n]
            ret_cutpoints = [cutpoint_0, cutpoint_1, cutpoint_2]
            return ret_alarms, ret_scores, ret_cutpoints, n

    def _suffi_stats(self, x):
        """
        compute sufficient statistics

        Args:
            x: input datum

        Returns:
            ndarray: a component of the sufficient statistics
        """

        x_ = x.reshape((1, -1))

        # x_1.shape = (1, m), x_2.shape = (m, m)
        x_1 = x_  # 1st order momoent
        x_2 = x_.reshape((-1, 1)) @ x_  # 2nd order moment

        return np.array([x_1, x_2])

    def _calc_stats(self):
        """
        Calculate all order of D-MDL. If self.__order=0, skip the computation of 1st and 2nd D-MDL.

        Returns:
            ndarray, ndarray, ndarray: 0th, 1st, 2nd D-MDL
        """

        # if the length of buckets is not sufficient to compute statistics,
        # return np.nans
        n = len(self.__window)
        if n == 1:
            return np.nan, np.nan, np.nan

        # statistics of D-MDL
        stats_0 = np.zeros(n - 1)
        stats_0[:] = np.nan
        stats_1 = np.zeros(n - 1)
        stats_1[:] = np.nan
        stats_2 = np.zeros(n - 1)
        stats_2[:] = np.nan

        Xmat = np.array(list(map((lambda x: x[0]), self.__window)))
        Xmat=Xmat.reshape((1 ,-1))
        Xmat = np.matrix(Xmat)

        n, m = Xmat.shape
        if n == 1:
            Xmat = Xmat.T
            n, m = Xmat.shape
        else:
            pass

        # calculate the NML code length with no change beforehand for the
        # computational efficiency.
        entire_stat = self.__encoding_func(
            Xmat, self.__window[n - 1][2], self.__window[n - 1][3])

        start_0, end_0 = self.__min_datapoints, n - self.__min_datapoints
        start_1, end_1 = self.__min_datapoints + 1, n - self.__min_datapoints
        start_2, end_2 = self.__min_datapoints + \
            1, n - (self.__min_datapoints + 1)

        # calculate statistics
        for cut in range(start_0, end_0 + 1):
            # 0th D-MDL
            former_stat_t = self.__encoding_func(
                Xmat[:cut],
                self.__window[cut - 1][2],
                self.__window[cut - 1][3]
            )
            latter_stat_t = self.__encoding_func(
                Xmat[cut:],
                self.__window[n - 1][2] - self.__window[cut - 1][2],
                self.__window[n - 1][3] - self.__window[cut - 1][3]
            )

            stat_t = former_stat_t + latter_stat_t
            stats_0[cut - 1] = entire_stat - stat_t

        for cut in range(start_1, end_1 + 1):
            stats_1[cut - 1] = stats_0[cut - 1] - stats_0[cut - 2]

        for cut in range(start_2, end_2 + 1):
            stats_2[cut - 1] = (stats_0[cut] - stats_0[cut - 1]) - (stats_0[cut - 1] - stats_0[cut - 2])

        return stats_0, stats_1, stats_2

    def _calculate_threshold(self, nn, n, order):
        """
        calculate threshold for an order of D-MDL.

        Args:
            nn: the number of possible cutpoints
            n: the number of data within the window
            order: 0th, 1st, or 2nd

        Returns:
            float: threshold
        """
        if order == 0:
            if self.__reliability:
                threshold = np.log(1 / self.__delta_0) + (self.__d /
                                                          2 + 1 + self.__delta_0) * np.log(n) + np.log(nn)
            else:
                threshold = self.__d / 2 * np.log(n) - np.log(self.__delta_0)

        elif order == 1:
            threshold = self.__d * np.log(n / 2) - np.log(self.__delta_1)

        else:
            threshold = 2 * (self.__d * np.log(n / 2) - np.log(self.__delta_2))

        return threshold
