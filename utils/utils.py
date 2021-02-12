import numpy as np
import pandas as pd
import multiprocessing as multi
from copy import deepcopy
from functools import partial


def Gradual_step(x, transition_period=300):
    """
    supplementary function of gradual changes

    Args:
        x: time point
        transition_peirod: length of the transition period

    Returns:
        float: return value
    """
    if x < 0:
        return 0.0
    elif 0 <= x < transition_period:
        return x / transition_period
    else:
        return 1.0



def mean_changing(transition_period):
    """
    return a mean-changing data sequence and the corresponding changepoints

    Args:
        transition_peirod: the length of the transition period

    Returns:
        (ndarray, ndarray): a data sequence, changepoints
    """

    data = np.random.normal(0, 1, 10000)
    for t in range(10000):
        for i in range(1, 10):
            data[t] += 0.6 * (10 - i) * Gradual_step(t - 1000 * i, transition_period=transition_period)

    changepoints = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000])
    return data, changepoints


def variance_changing(transition_period):
    """
    return a variance-changing data sequence and the corresponding changepoints

    Args:
        transition_peirod: the length of the transition period

    Returns:
        (ndarray, ndarray): a data sequence, changepoints
    """

    data = np.empty(10000)
    for t in range(10000):
        val = 0
        for i in range(1, 10):
            val += 0.3 * (10 - i) * Gradual_step(t - 1000 * i, transition_period=transition_period)
        val = np.exp(val)
        data[t] = np.random.normal(0, val)

    changepoints = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000])
    return data, changepoints


def create_dataset(n_samples, func, transition_period):
    """
    return a dataset

    Args:
        n_samples: the number of samples
        func: the function to generate a data sequence
        transition_peirod: the length of the transition period

    Returns:
        (ndarray, ndarray): dataset, changepoints
    """

    dataset = []
    changepoints = []

    for i in range(n_samples):
        data_sequence, cp = func(transition_period=transition_period)
        dataset.append(data_sequence)
        changepoints = cp

    return np.array(dataset), changepoints


def calc_AUC(scores, changepoints, tolerance_delay, div=100, both=True):
    """
    calculate the area under the curve (AUC) from scores and given changepoints.

    Args:
        scores: the change scores or change sign scores
        changepoints: changepoints or starting points of gradual changes
        tolerance_delay: tolerance delay for change or change sign detection
        div: sampling points for calculating the AUC
        both: how to define the section of benefits

    Returns:
        float: AUC

    """
    # basic statistics
    score_max = np.nanmax(scores)
    score_min = np.nanmin(scores)
    _scores = np.nan_to_num(scores, nan=score_min)

    benefit_sequence = np.zeros(len(scores))
    false_alarm_sequence = np.ones(len(scores))
    benefits = []
    false_alarms = []

    # benefitとfalse_alarmを計算する用の列を作成
    for changepoint in changepoints:
        if both:
            false_alarm_sequence[changepoint - tolerance_delay:changepoint + tolerance_delay + 1] = 0
            for j in range(changepoint - tolerance_delay, changepoint + tolerance_delay + 1):
                benefit_sequence[j] = max(1 - np.abs(j - changepoint) / tolerance_delay, benefit_sequence[j])
        else:
            false_alarm_sequence[changepoint:changepoint + tolerance_delay + 1] = 0
            for j in range(changepoint, changepoint + tolerance_delay + 1):
                benefit_sequence[j] = max(1 - np.abs(j - changepoint) / tolerance_delay, benefit_sequence[j])            

    # calculate benefits and false alarms with thresholds
    for i in range(div + 1):
        threshold = score_min + i * (score_max - score_min) / div
        alarms= np.where(_scores >= threshold, 1, 0)
        benefits.append(alarms.dot(benefit_sequence))
        false_alarms.append(alarms.dot(false_alarm_sequence))

    # regularization
    benefits = np.array(benefits)
    false_alarms = np.array(false_alarms)
    benefits /= np.max(benefits)
    false_alarm_rates = false_alarms/ np.max(false_alarms)

    # calculate AUC by numerical integration
    AUC = 0
    for i in range(div):
        AUC += (benefits[i] + benefits[i + 1]) * \
            (false_alarm_rates[i] - false_alarm_rates[i + 1]) / 2

    return AUC

def calc_F1(scores, changepoints, T, tuned_threshold=None, div=1000, both=True):
    # TODO: 修正する
    # tuned_thresholdがNoneのときは閾値のチューニングを、
    # そうでないときは与えられた閾値によるF値を計算する。
    if tuned_threshold == None:
        score_max = np.max(scores)
        score_min = np.min(scores)
        max_F_value = 0
        max_precision = 0  # 最大のprecisionというよりもむしろF値を最大たらしめるprecision
        max_recall = 0  # 最小のrecallというよりもむしろF値を最大たらしめるrecall
        opt_threshold = 0

        F_value_hist = np.zeros(div)
        precision_hist = np.zeros(div)
        recall_hist = np.zeros(div)
        threshold_hist = np.zeros(div)

        for i in range(div):
            threshold = score_min + i * (score_max - score_min) / (div - 1)
            a = np.where(scores >= threshold, 1, 0)
            a[0] = 0
            a[-1] = 0

            diff = np.diff(a)
            # estimated_changepoints_prev = np.where(
            #    diff == 1)[0]  # change-points

            estimated_changepoints_prev = []

            start = np.where(diff == 1)[0]
            end = np.where(diff == -1)[0]
            # print(start)
            # print(end)

            for j in range(start.size):
                s = start[j]
                e = end[j]
                # estimated_changepoints_prev.append(
                #    s + np.argmax(scores[s:e + 1]))
                estimated_changepoints_prev.append(s)

            estimated_changepoints_prev = np.array(estimated_changepoints_prev)

            TP = 0

            for c in changepoints:
                if both == True:
                    estimated_changepoints = np.where((estimated_changepoints_prev >= c - T) & (estimated_changepoints_prev <= c + T),
                                                      -1, estimated_changepoints_prev)
                elif both == False:
                    estimated_changepoints = np.where((estimated_changepoints_prev >= c) & (estimated_changepoints_prev <= c + T),
                                                      -1, estimated_changepoints_prev)

                if not (estimated_changepoints == estimated_changepoints_prev).all():
                    TP += 1
                estimated_changepoints_prev = np.copy(estimated_changepoints)

            FP = len(changepoints) - TP
            FN = len(np.where(estimated_changepoints_prev != -1)[0])

            if TP == 0 and FP == 0:
                precision = 0
            else:
                precision = TP / (TP + FP)
            if TP == 0 and FN == 0:
                recall = 0
            else:
                recall = TP / (TP + FN)

            if precision == 0 and recall == 0:
                F_value = 0
            else:
                F_value = 2 * precision * recall / (precision + recall)

            F_value_hist[i] = F_value
            precision_hist[i] = precision
            recall_hist[i] = recall
            threshold_hist[i] = threshold

#            if max_F_value < F_value:
#                max_F_value = F_value
#                max_precision = precision
#                max_recall = recall
#                opt_threshold = threshold

        max_F_value = np.max(F_value_hist)
        ind = np.where(F_value_hist == max_F_value)[0]
        middle_ind = int(np.mean(ind))
        max_precision = precision_hist[middle_ind]
        max_recall = recall_hist[middle_ind]
        opt_threshold = threshold_hist[middle_ind]

        return max_F_value, max_precision, max_recall, opt_threshold
    else:
        a = np.where(scores >= tuned_threshold, 1, 0)

        diff = np.diff(a)
        estimated_changepoints_prev = np.where(diff == 1)[0]  # change-points
        TP = 0

        for c in changepoints:
            if both == True:
                estimated_changepoints = np.where((estimated_changepoints_prev >= c - T) & (estimated_changepoints_prev <= c + T),
                                                  -1, estimated_changepoints_prev)
            elif both == False:
                estimated_changepoints = np.where((estimated_changepoints_prev >= c) & (estimated_changepoints_prev <= c + T),
                                                  -1, estimated_changepoints_prev)

            if not (estimated_changepoints == estimated_changepoints_prev).all():
                TP += 1
            estimated_changepoints_prev = np.copy(estimated_changepoints)

        FP = len(changepoints) - TP
        FN = len(np.where(estimated_changepoints_prev != -1)[0])

        if TP == 0 and FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        if TP == 0 and FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)

        if precision == 0 and recall == 0:
            F_value = 0
        else:
            F_value = 2 * precision * recall / (precision + recall)

        return F_value, precision, recall

def DJ(X):
    diff = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        if i == 0 or i == X.shape[0] - 1:
            diff[i] = 1
        else:
            diff[i] = X[i + 1] / X[i]
    diff -= 1
    return diff

