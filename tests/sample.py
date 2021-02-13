import sys
sys.path.append('./')
import numpy as np
import optuna
import changefinder
import bocpd
import dmdl.sdmdl as sdmdl
import dmdl.hsdmdl2 as hsdmdl2
import utils.sdmdl_nml as sdmdl_nml
import utils.hsdmdl2_nml as hsdmdl2_nml
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from utils.utils import Gradual_step, mean_changing, variance_changing, create_dataset, calc_AUC

# TODO: docstring

def one_mean_changing(transition_period):
    """
    return a mean-changing data sequence and the corresponding changepoints

    Args:
        transition_peirod: the length of the transition period

    Returns:
        (ndarray, ndarray): a data sequence, changepoints
    """

    data = np.random.normal(0, 1, 2000)
    for t in range(2000):
        data[t] += 10.0 * \
            Gradual_step(t - 1000, transition_period=transition_period)

    changepoints = np.array([1000])
    return data, changepoints


def _calc_metrics(data, changepoints, tolerance_delay, retrospective):
    _retrospective = deepcopy(retrospective)
    scores = _retrospective.calc_scores(data)
    AUC = calc_AUC(scores, changepoints, tolerance_delay)
    return AUC


def main():
    np.random.seed(0)

    tolerance_delay = 100
    transition_period = 200
    data, changepoints = one_mean_changing(transition_period=transition_period)

    print("ChangeFinder")
    retrospective = changefinder.Retrospective(r=0.1, order=5, smooth=5)
    AUC = _calc_metrics(
        data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)
    print("AUC: ", AUC)

    print("BOCPD")
    lam = 1000
    alpha = 0.5
    beta = 5
    kappa = 5
    mu = 0
    h = partial(bocpd.constant_hazard, lam)
    lik = bocpd.StudentT(alpha, beta, kappa, mu)
    retrospective = bocpd.Retrospective(hazard_func=h, likelihood_func=lik)
    AUC = _calc_metrics(
        data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)
    print("AUC: ", AUC)

    print("SDMDL 0th")
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=1e8,
                           div_min=1e-8, div_max=1e8)
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=1e8,
                                  div_min=1e-8, div_max=1e8)
    retrospective = sdmdl.Retrospective(h=100, encoding_func=nml_gaussian,
                                        complexity_func=complexity_gaussian, order=0)
    AUC = _calc_metrics(
        data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)
    print("AUC: ", AUC)

    print("SDMDL 1st")
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=1e8,
                           div_min=1e-8, div_max=1e8)
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=1e8,
                                  div_min=1e-8, div_max=1e8)
    retrospective = sdmdl.Retrospective(h=100, encoding_func=nml_gaussian,
                                        complexity_func=complexity_gaussian, order=1)
    AUC = _calc_metrics(
        data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)
    print("AUC: ", AUC)

    print("SDMDL 2nd")
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=1e8,
                           div_min=1e-8, div_max=1e8)
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=1e8,
                                  div_min=1e-8, div_max=1e8)
    retrospective = sdmdl.Retrospective(h=100, encoding_func=nml_gaussian,
                                        complexity_func=complexity_gaussian, order=2)
    AUC = _calc_metrics(
        data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)
    print("AUC: ", AUC)

    print("Hierarchical 0th")
    nml_gaussian = partial(hsdmdl2_nml.nml_gaussian)
    order=0
    min_datapoints = 5
    delta_0 =0.05
    delta_1 = 0.05
    delta_2 = 0.05
    retrospective = hsdmdl2.Retrospective(encoding_func=nml_gaussian, d=2, M=5, min_datapoints=min_datapoints, delta_0=delta_0,
                        delta_1=delta_1, delta_2=delta_2, how_to_drop='all', order=order, reliability=True)
    AUC = _calc_metrics(data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)
    print("AUC: ", AUC)

    print("Hierarchical 1st")
    nml_gaussian = partial(hsdmdl2_nml.nml_gaussian)
    order=1
    min_datapoints = 5
    delta_0 =0.05
    delta_1 = 0.05
    delta_2 = 0.05
    retrospective = hsdmdl2.Retrospective(encoding_func=nml_gaussian, d=2, M=5, min_datapoints=min_datapoints, delta_0=delta_0,
                        delta_1=delta_1, delta_2=delta_2, how_to_drop='all', order=order, reliability=True)
    AUC = _calc_metrics(data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)
    print("AUC: ", AUC)

    print("Hierarchical 2nd")
    nml_gaussian = partial(hsdmdl2_nml.nml_gaussian)
    order=2
    min_datapoints = 5
    delta_0 =0.05
    delta_1 = 0.05
    delta_2 = 0.05
    retrospective = hsdmdl2.Retrospective(encoding_func=nml_gaussian, d=2, M=5, min_datapoints=min_datapoints, delta_0=delta_0,
                        delta_1=delta_1, delta_2=delta_2, how_to_drop='all', order=order, reliability=True)
    AUC = _calc_metrics(data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)
    print("AUC: ", AUC)



if __name__ == "__main__":
    main()
