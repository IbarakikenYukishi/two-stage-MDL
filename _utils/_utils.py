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
