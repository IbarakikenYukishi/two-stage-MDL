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
            data[t] += 0.6 * (10 - i) * Gradual_step(t -
                                                     1000 * i, transition_period=transition_period)

    changepoints = np.array(
        [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000])
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
            val += 0.3 * (10 - i) * Gradual_step(t - 1000 * i,
                                                 transition_period=transition_period)
        val = np.exp(val)
        data[t] = np.random.normal(0, val)

    changepoints = np.array(
        [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000])
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


def calc_AUC(scores, changepoints, tolerance_delay, div=500, both=True):
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

    benefit_sequence = np.zeros(len(_scores))
    false_alarm_sequence = np.ones(len(_scores))
    benefits = []
    false_alarms = []

    # benefitとfalse_alarmを計算する用の列を作成
    for changepoint in changepoints:
        if both:
            false_alarm_sequence[
                changepoint - tolerance_delay:changepoint + tolerance_delay + 1] = 0
            for j in range(changepoint - tolerance_delay, changepoint + tolerance_delay + 1):
                benefit_sequence[j] = max(
                    1 - np.abs(j - changepoint) / tolerance_delay, benefit_sequence[j])
        else:
            false_alarm_sequence[
                changepoint:changepoint + tolerance_delay + 1] = 0
            for j in range(changepoint, changepoint + tolerance_delay + 1):
                benefit_sequence[j] = max(
                    1 - np.abs(j - changepoint) / tolerance_delay, benefit_sequence[j])

    # calculate benefits and false alarms with thresholds
    for i in range(div + 1):
        threshold = score_min + i * (score_max - score_min) / div
        alarms = np.where(_scores >= threshold, 1, 0)
        benefits.append(alarms.dot(benefit_sequence))
        false_alarms.append(alarms.dot(false_alarm_sequence))

    # regularization
    benefits = np.array(benefits)
    false_alarms = np.array(false_alarms)
    benefits /= np.max(benefits)
    false_alarm_rates = false_alarms / np.max(false_alarms)

    # calculate AUC by numerical integration
    AUC = 0
    for i in range(div):
        AUC += (benefits[i] + benefits[i + 1]) * \
            (false_alarm_rates[i] - false_alarm_rates[i + 1]) / 2

    return AUC


def calc_F1_score(scores, changepoints, tolerance_delay, tuned_threshold=None, div=500, both=True):
    """
    Calculate F1 score. If tuned_threshold is None, return the tuned threshold.

    Args:
        scores: the change scores or change sign scores
        changepoints: changepoints or starting points of gradual changes
        tolerance_delay: tolerance delay for change or change sign detection
        tuned_threshold: the tuned threshold. If it is None, tune the threshold and 
        return metrics with it.
        div: sampling points for calculating the AUC
        both: how to define the section of benefits

    Returns:
        float, float, float, Optional[float]: F1 score, precision, recall, tuned threshold

    """

    # basic statistics
    score_max = np.nanmax(scores)
    score_min = np.nanmin(scores)
    _scores = np.nan_to_num(scores, nan=score_min)  # fill nan

    if tuned_threshold == None:


        # statistics
        max_F1_score = 0
        max_precision = 0  # the precision it achieves maximum F1 score
        max_recall = 0  # the recall it achieves maximum F1 score
        opt_threshold = 0

        # statistics list
        F1_scores = np.zeros(div)
        precisions = np.zeros(div)
        recalls = np.zeros(div)
        thresholds = np.zeros(div)

        for i in range(div):
            threshold = score_min + i * (score_max - score_min) / (div - 1)
            alarms = np.where(_scores >= threshold, 1, 0)
            alarms[0] = 0
            alarms[-1] = 0

            diff = np.diff(alarms)
            estimated_changepoints_prev = []

            start = np.where(diff == 1)[0]
            end = np.where(diff == -1)[0]

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
                    estimated_changepoints = np.where((estimated_changepoints_prev >= c - tolerance_delay) & (estimated_changepoints_prev <= c + tolerance_delay),
                                                      -1, estimated_changepoints_prev)
                elif both == False:
                    estimated_changepoints = np.where((estimated_changepoints_prev >= c) & (estimated_changepoints_prev <= c + tolerance_delay),
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
                F1_score = 0
            else:
                F1_score = 2 * precision * recall / (precision + recall)

            F1_scores[i] = F1_score
            precisions[i] = precision
            recalls[i] = recall
            thresholds[i] = threshold

        max_F1_score = np.max(F1_scores)
        idx = np.where(F1_scores == max_F1_score)[0]
        middle_idx = int(np.mean(idx))
        max_precision = precisions[middle_idx]
        max_recall = recalls[middle_idx]
        opt_threshold = thresholds[middle_idx]

        return max_F1_score, max_precision, max_recall, opt_threshold
    else:
        alarms = np.where(_scores >= tuned_threshold, 1, 0)

        diff = np.diff(alarms)
        estimated_changepoints_prev = np.where(diff == 1)[0]  # change-points
        TP = 0

        for c in changepoints:
            if both == True:
                estimated_changepoints = np.where((estimated_changepoints_prev >= c - tolerance_delay) & (estimated_changepoints_prev <= c + tolerance_delay),
                                                  -1, estimated_changepoints_prev)

            elif both == False:
                estimated_changepoints = np.where((estimated_changepoints_prev >= c) & (estimated_changepoints_prev <= c + tolerance_delay),
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
            F1_score = 0
        else:
            F1_score = 2 * precision * recall / (precision + recall)

        return F1_score, precision, recall
