import numpy as np
import scipy.linalg as sl
from scipy import special
from collections import deque
import functools as fts
import operator as op


class Retrospective:
    """
    Hierarchical Sequential D-MDL algorithm with SCAW2 (Prospective)
    """

    def __init__(self, encoding_func, d, M=5, min_datapoints=1, delta_0=0.05, delta_1=0.05, delta_2=0.05, how_to_drop='all', order=-1, reliability=True):
        """
        Args:
            encoding_func: encoding function
            d: dimension of the parametric class
            M: maximum number of buckets which have the same data points
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
        self.__M = M
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

        detector = Prospective(encoding_func=self.__encoding_func, d=self.__d, M=self.__M, min_datapoints=self.__min_datapoints,
                               delta_0=self.__delta_0, delta_1=self.__delta_1, delta_2=self.__delta_2, how_to_drop=self.__how_to_drop,
                               order=self.__order, reliability=self.__reliability)

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
    Hierarchical Sequential D-MDL algorithm with SCAW2 (Prospective)
    """

    def __init__(self, encoding_func, d, M=5, min_datapoints=1, delta_0=0.05, delta_1=0.05, delta_2=0.05, how_to_drop='all', order=-1, reliability=True):
        """
        Args:
            encoding_func: encoding function
            d: dimension of the parametric class
            M: maximum number of buckets which have the same data points
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
        self.__M = M
        self.__min_datapoints = min_datapoints
        self.__delta_0 = delta_0
        self.__delta_1 = delta_1
        self.__delta_2 = delta_2
        self.__how_to_drop = how_to_drop
        self.__buckets = deque([])
        self.__order = order
        self.__reliability = reliability
        assert how_to_drop in ('cutpoint', 'all')

    def update(self, x):
        """
        calculate the score of the input datum

        Args:
            x: input datum

        Returns:
            list, list, list, ndarray: alarms, scores, cutpoints, window size
        """

        # compute the sufficient statistics of the input datum
        newT = self._suffi_stats(x)
        self._combine(newT)  # combine buckets

        # the number of data contained in buckets
        n = sum(map((lambda x: 2 ** x[3]), self.__buckets))

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
        if max_score_0 >= self._calculate_threshold(nn=len(self.__buckets) - 1, n=n, order=0):
            alarm_0 = 1

        # 1st alarm
        if np.isnan(max_score_1) == False and max_score_1 >= self._calculate_threshold(nn=len(self.__buckets) - 1, n=n, order=1):
            cut_1_bucket = np.nanargmax(scores_1)
            cutpoint_1 = sum(map(
                (lambda x: 2 ** x[3]), [self.__buckets[i] for i in range(0, cut_1_bucket + 1)]))
            alarm_1 = 1

        # 2nd alarm
        if np.isnan(max_score_2) == False and max_score_2 >= self._calculate_threshold(nn=len(self.__buckets) - 1, n=n, order=2):
            cut_2_bucket = np.nanargmax(scores_2)
            cutpoint_2 = sum(map(
                (lambda x: 2 ** x[3]), [self.__buckets[i] for i in range(0, cut_2_bucket + 1)]))
            alarm_2 = 1

        if alarm_0 == 1:  # if 0th alarm was raised
            cut_0_bucket = np.nanargmax(scores_0)
            cutpoint_0 = sum(map(
                (lambda x: 2 ** x[3]), [self.__buckets[i] for i in range(0, cut_0_bucket + 1)]))
            for j in range(0, cut_0_bucket + 1):
                self.__buckets.popleft()
            capacity_sum = sum(map((lambda x: 2 ** x[3]), self.__buckets))

            ret_alarms = [alarm_0, alarm_1, alarm_2]
            ret_scores = [max_score_0 / n, max_score_1 / n, max_score_2 / n]
            ret_cutpoints = [cutpoint_0, cutpoint_1, cutpoint_2]

            if self.__how_to_drop == 'cutpoint':
                return ret_alarms, ret_scores, ret_cutpoints, capacity_sum
            if self.__how_to_drop == 'all':
                self.__buckets = deque([])
                return ret_alarms, ret_scores, ret_cutpoints, 0
        else:
            ret_alarms = [alarm_0, alarm_1, alarm_2]
            ret_scores = [max_score_0 / n, max_score_1 / n, max_score_2 / n]
            ret_cutpoints = [cutpoint_0, cutpoint_1, cutpoint_2]
            return ret_alarms, ret_scores, ret_cutpoints, n

    def _combine(self, newT):
        """
        combine a new datum to the buckets
        new_buckets according to the "bucket rule"

        Args:
            newT: input datum
        """

        # make a new datum into a bucket: form of
        # (suffi_stats, suffi_stats_right, suffi_stats_left log(length))
        # suffi_stats contains the sufficient statistics of the bucket
        # suffi_stats_right contains those of the bucket without the rightest missing datum
        # suffi_stats_left contains those of the bucket without the leftest
        # missing datum

        # append a datum
        empty = np.zeros(newT.shape)
        new = (newT, empty, empty, 0.0)
        self.__buckets.append(new)

        k = self.__buckets[0][3]  # get the largest capacity in the buckets
        # the number of buckets (not the number of data points)
        n = len(self.__buckets)
        indicator = n - 1  # indicate the point where we finish combining buckets

        # conduct bucketing.
        for i in range(0, int(k) + 1):
            counter = 0
            while indicator >= 0 and self.__buckets[indicator][3] == i:
                counter += 1
                indicator -= 1
            if counter > self.__M:
                indicator += 1

                # move unnecessary buckets to the rightest
                self.__buckets.rotate(n - 2 - indicator)
                left = self.__buckets[n - 2]
                right = self.__buckets[n - 1]
                # combine the leftest ones s.t. length = i
                combination = (left[0] + right[0], left[2] +
                               right[0], left[0] + right[1], i + 1)
                # delete two unnecessary buckets
                self.__buckets.pop()
                self.__buckets.pop()
                # append a combination bucket
                self.__buckets.append(combination)
                n -= 1
                # return to where they were
                self.__buckets.rotate(-(n - 1 - indicator))

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

    def _calc_stats(self):
        """
        Calculate all order of D-MDL. If self.__order=0, skip the computation of 1st and 2nd D-MDL.

        Returns:
            ndarray, ndarray, ndarray: 0th, 1st, 2nd D-MDL
        """

        # if the length of buckets is not sufficient to compute statistics,
        # return np.nans
        n = len(self.__buckets)
        if n == 1:
            return np.nan, np.nan, np.nan

        # statistics of D-MDL
        stat_list_0 = np.zeros(n - 1)
        stat_list_0[:] = np.nan
        stat_list_1 = np.zeros(n - 1)
        stat_list_1[:] = np.nan
        stat_list_2 = np.zeros(n - 1)
        stat_list_2[:] = np.nan

        # calculate the NML code length with no change beforehand for the
        # computational efficiency.
        entire = (sum(map((lambda x: x[0]), self.__buckets)), sum(
            map((lambda x: 2 ** x[3]), self.__buckets)))
        entire_stat = self.__encoding_func(entire)

        # specify the starting and ending bucket to compute the statistics
        num_datapoints = list(map((lambda x: 2 ** x[3]), self.__buckets))
        start_0, end_0 = self._acquire_endpoints(num_datapoints, order=0)
        start_1, end_1 = self._acquire_endpoints(num_datapoints, order=1)
        start_2, end_2 = self._acquire_endpoints(num_datapoints, order=2)

        # calculate statistics
        for cut in range(1, n):
            if start_0 <= cut <= end_0:  # 0th order
                # 0th D-MDL
                former_t = (sum(map((lambda x: x[0]), [self.__buckets[i] for i in range(0, cut)])), sum(
                    map((lambda x: 2 ** x[3]), [self.__buckets[i] for i in range(0, cut)])))
                latter_t = (sum(map((lambda x: x[0]), [self.__buckets[i] for i in range(cut, n)])), sum(
                    map((lambda x: 2 ** x[3]), [self.__buckets[i] for i in range(cut, n)])))

                former_t_stat = self.__encoding_func(former_t)
                latter_t_stat = self.__encoding_func(latter_t)

                stat_list_0[cut - 1] = entire_stat - \
                    former_t_stat - latter_t_stat

            if self.__order != 0:  # if order==0, skip the computation of 1st and 2nd D-MDL
                if start_1 <= cut <= end_1:  # 1st D-MDL
                    former_tm = (sum(map((lambda x: x[0]), [self.__buckets[i] for i in range(0, cut)])) - self.__buckets[cut - 1][0] + self.__buckets[cut - 1][1], sum(
                        map((lambda x: 2 ** x[3]), [self.__buckets[i] for i in range(0, cut)])) - 1)

                    latter_tm = (sum(map((lambda x: x[0]), [self.__buckets[i] for i in range(cut, n)])) + self.__buckets[cut - 1][0] - self.__buckets[cut - 1][1], sum(
                        map((lambda x: 2 ** x[3]), [self.__buckets[i] for i in range(cut, n)])) + 1)

                    former_tm_stat = self.__encoding_func(former_tm)
                    latter_tm_stat = self.__encoding_func(latter_tm)

                    stat_list_1[
                        cut - 1] = (entire_stat - former_t_stat - latter_t_stat) - (entire_stat - former_tm_stat - latter_tm_stat)

                if start_2 <= cut <= end_2:  # 2nd D-MDL
                    former_tp = (sum(map((lambda x: x[0]), [self.__buckets[i] for i in range(0, cut)])) + self.__buckets[cut][0] - self.__buckets[cut][2], sum(
                        map((lambda x: 2 ** x[3]), [self.__buckets[i] for i in range(0, cut)])) + 1)
                    latter_tp = (sum(map((lambda x: x[0]), [self.__buckets[i] for i in range(cut, n)])) - self.__buckets[cut][0] + self.__buckets[cut][2], sum(
                        map((lambda x: 2 ** x[3]), [self.__buckets[i] for i in range(cut, n)])) - 1)

                    former_tp_stat = self.__encoding_func(former_tp)
                    latter_tp_stat = self.__encoding_func(latter_tp)

                    stat_list_2[cut - 1] = ((entire_stat - former_tp_stat - latter_tp_stat) - (entire_stat - former_t_stat - latter_t_stat)) - (
                        (entire_stat - former_t_stat - latter_t_stat) - (entire_stat - former_tm_stat - latter_tm_stat))

        return stat_list_0, stat_list_1, stat_list_2

    def _acquire_endpoints(self, num_datapoints, order):
        """
        determine the starting and ending bucket to compute the statistics.
        0th D-MDL needs at least min_datapoints for both side of data
        1st D-MDL needs at least min_datapoints+1 for left side of data and min_datapoints for right side of data
        2nd D-MDL needs at least min_datapoints+1 for both side of data

        Args:
            num_datapoints: the list of each number of data points in each bucket
            order: which order of D-MDL

        Returns:
            int, int: starting and ending bucket
        """

        n = len(num_datapoints)
        start = n
        end = -1

        if order == 0:
            start_min_datapoints = self.__min_datapoints
            end_min_datapoints = self.__min_datapoints
        elif order == 1:
            start_min_datapoints = self.__min_datapoints + 1
            end_min_datapoints = self.__min_datapoints
        else:
            start_min_datapoints = self.__min_datapoints + 1
            end_min_datapoints = self.__min_datapoints + 1

        # specify the stating bucket
        total_datapoints = 0
        for i in range(min(start_min_datapoints, n)):
            total_datapoints += num_datapoints[i]
            if total_datapoints >= start_min_datapoints:
                start = i + 1
                break

        # specify the ending bucket
        total_datapoints = 0
        for i in range(min(end_min_datapoints, n)):
            total_datapoints += num_datapoints[n - 1 - i]
            if total_datapoints >= end_min_datapoints:
                end = n - i
                break

        return start, end
