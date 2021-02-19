import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import os
import changefinder
import bocpd
import dmdl.sdmdl as sdmdl
import dmdl.hsdmdl1 as hsdmdl1
import dmdl.hsdmdl2 as hsdmdl2
import tsmdl.fw2s_mdl as fw2s_mdl
import tsmdl.aw2s_mdl as aw2s_mdl
import utils.sdmdl_nml as sdmdl_nml
import utils.hsdmdl1_nml as hsdmdl1_nml
import utils.hsdmdl2_nml as hsdmdl2_nml
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from utils.utils import Gradual_step, calc_AUC

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


def _calc_metrics_draw_figure(name, data, changepoints, tolerance_delay, retrospective):
    # Calculation of AUC
    print(name)
    _retrospective = deepcopy(retrospective)
    scores = _retrospective.calc_scores(data)
    AUC = calc_AUC(scores, changepoints, tolerance_delay, both=True)
    print("AUC: ", AUC)

    # Draw a figure
    output_path = "./figs/"
    os.makedirs(output_path, exist_ok=True)
    fontsize = 18
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

    axes[0].plot(data, label="Raw Data")
    axes[0].set_ylabel('Value', fontsize=fontsize)
    axes[0].set_ylim(np.nanmin(data) - 5, np.nanmax(data) + 5)
    axes[0].legend(loc='upper left', fontsize=fontsize)

    axes[1].plot(scores, label=name, color="blue")
    axes[1].set_ylabel('Score', fontsize=fontsize)
    axes[1].legend(loc='upper left', fontsize=fontsize)
    axes[1].set_ylim(np.nanmin(scores), np.nanmax(scores))

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    fig.savefig(output_path + name + ".png")


def main():
    np.random.seed(0)

    tolerance_delay = 100
    transition_period = 200
    data, changepoints = one_mean_changing(transition_period=transition_period)

    name = "ChangeFinder"
    retrospective = changefinder.Retrospective(r=0.1, order=5, smooth=5)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)

    name = "BOCPD"
    h = partial(bocpd.constant_hazard, 1000)
    lik = bocpd.StudentT(0.5, 5, 5, 0)
    retrospective = bocpd.Retrospective(hazard_func=h, likelihood_func=lik)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)

    name = "SDMDL_0th"
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=1e8,
                           div_min=1e-8, div_max=1e8)
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=1e8,
                                  div_min=1e-8, div_max=1e8)
    retrospective = sdmdl.Retrospective(h=100, encoding_func=nml_gaussian,
                                        complexity_func=complexity_gaussian, order=0)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)

    name = "SDMDL_1st"
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=1e8,
                           div_min=1e-8, div_max=1e8)
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=1e8,
                                  div_min=1e-8, div_max=1e8)
    retrospective = sdmdl.Retrospective(h=100, encoding_func=nml_gaussian,
                                        complexity_func=complexity_gaussian, order=1)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)

    name = "SDMDL_2nd"
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=1e8,
                           div_min=1e-8, div_max=1e8)
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=1e8,
                                  div_min=1e-8, div_max=1e8)
    retrospective = sdmdl.Retrospective(h=100, encoding_func=nml_gaussian,
                                        complexity_func=complexity_gaussian, order=2)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)

    name = "Hierarchical_SCAW1_0th"
    lnml_gaussian = partial(hsdmdl1_nml.lnml_gaussian)
    retrospective = hsdmdl1.Retrospective(encoding_func=lnml_gaussian, d=2, min_datapoints=5, delta_0=0.05,
                                          delta_1=0.05, delta_2=0.05, how_to_drop='all', order=0, reliability=True)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)

    name = "Hierarchical_SCAW1_1st"
    lnml_gaussian = partial(hsdmdl1_nml.lnml_gaussian)
    retrospective = hsdmdl1.Retrospective(encoding_func=lnml_gaussian, d=2, min_datapoints=5, delta_0=0.05,
                                          delta_1=0.05, delta_2=0.05, how_to_drop='all', order=1, reliability=True)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)

    name = "Hierarchical_SCAW1_2nd"
    lnml_gaussian = partial(hsdmdl1_nml.lnml_gaussian)
    retrospective = hsdmdl1.Retrospective(encoding_func=lnml_gaussian, d=2, min_datapoints=5, delta_0=0.05,
                                          delta_1=0.05, delta_2=0.05, how_to_drop='all', order=2, reliability=True)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)

    name = "Hierarchical_SCAW2_0th"
    nml_gaussian = partial(hsdmdl2_nml.nml_gaussian)
    retrospective = hsdmdl2.Retrospective(encoding_func=nml_gaussian, d=2, M=5, min_datapoints=5, delta_0=0.05,
                                          delta_1=0.05, delta_2=0.05, how_to_drop='all', order=0, reliability=True)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)

    name = "Hierarchical_SCAW2_1st"
    nml_gaussian = partial(hsdmdl2_nml.nml_gaussian)
    retrospective = hsdmdl2.Retrospective(encoding_func=nml_gaussian, d=2, M=5, min_datapoints=5, delta_0=0.05,
                                          delta_1=0.05, delta_2=0.05, how_to_drop='all', order=1, reliability=True)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)

    name = "Hierarchical_SCAW2_2nd"
    nml_gaussian = partial(hsdmdl2_nml.nml_gaussian)
    retrospective = hsdmdl2.Retrospective(encoding_func=nml_gaussian, d=2, M=5, min_datapoints=5, delta_0=0.05,
                                          delta_1=0.05, delta_2=0.05, how_to_drop='all', order=2, reliability=True)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)

    name = "FW2S_MDL"
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=1e8,
                           div_min=1e-8, div_max=1e8)
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=1e8,
                                  div_min=1e-8, div_max=1e8)
    retrospective_first = sdmdl.Retrospective(h=100, encoding_func=nml_gaussian,
                                              complexity_func=complexity_gaussian, order=0)
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=1e8,
                           div_min=1e-8, div_max=1e8)
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=1e8,
                                  div_min=1e-8, div_max=1e8)
    retrospective_second = sdmdl.Retrospective(h=100, encoding_func=nml_gaussian,
                                               complexity_func=complexity_gaussian, order=0)
    retrospective = fw2s_mdl.Retrospective(
        retrospective_first, retrospective_second)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)

    name = "AW2S_MDL"
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=1e8,
                           div_min=1e-8, div_max=1e8)
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=1e8,
                                  div_min=1e-8, div_max=1e8)
    retrospective_first = sdmdl.Retrospective(h=100, encoding_func=nml_gaussian,
                                              complexity_func=complexity_gaussian, order=0)
    lnml_gaussian = partial(hsdmdl2_nml.lnml_gaussian, sigma_given=0.3)
    retrospective_second = hsdmdl2.Retrospective(encoding_func=lnml_gaussian, d=2, M=5, min_datapoints=5, delta_0=0.05,
                                                 delta_1=0.05, delta_2=0.05, how_to_drop='all', order=0, reliability=True)

    retrospective = aw2s_mdl.Retrospective(
        retrospective_first, retrospective_second)
    _calc_metrics_draw_figure(
        name, data, changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)


if __name__ == "__main__":
    main()
