import sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as multi
import optuna
import changefinder
import bocpd
import dmdl.sdmdl as sdmdl
import dmdl.hsdmdl2 as hsdmdl2
import tsmdl.aw2s_mdl as aw2s_mdl
import utils.sdmdl_nml as sdmdl_nml
import utils.hsdmdl2_nml as hsdmdl2_nml
from multiprocessing import Pool
from functools import partial
from copy import deepcopy
from utils.utils import mean_changing, variance_changing, create_dataset, calc_F1_score


def _calc_metrics(idx_data, dataset, changepoints, tolerance_delay, threshold, retrospective):  # calculate the metrics
    _retrospective = deepcopy(retrospective)
    scores = _retrospective.calc_scores(dataset[idx_data])
    F1_score, precision, recall = calc_F1_score(
        scores, changepoints, tolerance_delay, threshold)
    return F1_score, precision, recall


# obtain the optimal threshold
def calc_opt_threshold(train, changepoints, tolerance_delay, retrospective):
    _retrospective = deepcopy(retrospective)
    scores = _retrospective.calc_scores(train)
    _, _, _, opt_threshold = calc_F1_score(
        scores, changepoints, tolerance_delay)
    return opt_threshold


def _objective_CF(trial, train, changepoints, tolerance_delay):  # ChangeFinder
    # hyperparameters
    r = trial.suggest_uniform('r', 0.01, 0.99)
    order = trial.suggest_int('order', 1, 20)
    smooth = trial.suggest_int('smooth', 3, 20)

    retrospective = changefinder.Retrospective(r=r, order=order, smooth=smooth)
    scores = retrospective.calc_scores(train)
    F1_score, _, _, _ = calc_F1_score(scores, changepoints, tolerance_delay)

    return -F1_score


def conduct_CF(n_trials, n_samples, dataset, changepoints, tolerance_delay):  # ChangeFinder
    # hyperparameter tuning
    objective_CF = partial(_objective_CF, train=dataset[0],
                           changepoints=changepoints, tolerance_delay=tolerance_delay)
    study = optuna.create_study()
    study.optimize(objective_CF, n_trials=n_trials, n_jobs=-1)
    opt_r = study.best_params['r']
    opt_order = study.best_params['order']
    opt_smooth = study.best_params['smooth']

    # optimal threshold
    retrospective = changefinder.Retrospective(
        r=opt_r, order=opt_order, smooth=opt_smooth)
    opt_threshold = calc_opt_threshold(train=dataset[0], changepoints=changepoints,
                                       tolerance_delay=tolerance_delay, retrospective=retrospective)

    # calculate metrics
    calc_metrics = partial(_calc_metrics, dataset=dataset, changepoints=changepoints,
                           tolerance_delay=tolerance_delay, threshold=opt_threshold, retrospective=retrospective)
    p = Pool(multi.cpu_count() - 1)
    args = list(range(1, n_samples))
    res = np.array(p.map(calc_metrics, args))
    p.close()

    # result
    print("F1 score:  ", np.mean(res[:, 0]), "±", np.std(res[:, 0]))
    print("precision:  ", np.mean(res[:, 1]), "±", np.std(res[:, 1]))
    print("recall:  ", np.mean(res[:, 2]), "±", np.std(res[:, 2]))

    row = pd.DataFrame({"method": ["ChangeFinder"], "F1_score_mean": np.mean(res[:, 0]), "F1_score_std": np.std(res[:, 0]),
                        "precision_mean": np.mean(res[:, 1]), "precision_std": np.std(res[:, 1]),
                        "recall_mean": np.mean(res[:, 2]), "recall_std": np.std(res[:, 2])})

    return row


def _objective_BOCPD(trial, train, changepoints, tolerance_delay):  # BOCPD
    lam = trial.suggest_int('lam', 2, 1000)
    alpha = trial.suggest_uniform('alpha', 0.01, 10)
    beta = trial.suggest_uniform('beta', 0.01, 10)
    kappa = trial.suggest_uniform('kappa', 0.01, 10)
    mu = trial.suggest_uniform('mu', 0.01, 10)

    h = partial(bocpd.constant_hazard, lam)
    lik = bocpd.StudentT(alpha, beta, kappa, mu)
    retrospective = bocpd.Retrospective(hazard_func=h, likelihood_func=lik)

    scores = retrospective.calc_scores(train)
    F1_score, _, _, _ = calc_F1_score(
        scores, changepoints, tolerance_delay)

    return -F1_score


def conduct_BOCPD(n_trials, n_samples, dataset, changepoints, tolerance_delay):  # BOCPD
    # hyperparameter tuning
    objective_BOCPD = partial(_objective_BOCPD, train=dataset[0],
                              changepoints=changepoints, tolerance_delay=tolerance_delay)
    study = optuna.create_study()
    study.optimize(objective_BOCPD, n_trials=n_trials, n_jobs=-1)
    opt_lam = study.best_params['lam']
    opt_alpha = study.best_params['alpha']
    opt_beta = study.best_params['beta']
    opt_kappa = study.best_params['kappa']
    opt_mu = study.best_params['mu']

    # optimal threshold
    h = partial(bocpd.constant_hazard, opt_lam)
    lik = bocpd.StudentT(opt_alpha, opt_beta, opt_kappa, opt_mu)
    retrospective = bocpd.Retrospective(hazard_func=h, likelihood_func=lik)
    opt_threshold = calc_opt_threshold(train=dataset[0], changepoints=changepoints,
                                       tolerance_delay=tolerance_delay, retrospective=retrospective)

    # calculate metrics
    calc_metrics = partial(_calc_metrics, dataset=dataset, changepoints=changepoints,
                           tolerance_delay=tolerance_delay, threshold=opt_threshold, retrospective=retrospective)
    p = Pool(multi.cpu_count() - 1)
    args = list(range(1, n_samples))
    res = np.array(p.map(calc_metrics, args))
    p.close()

    # result
    print("F1 score:  ", np.mean(res[:, 0]), "±", np.std(res[:, 0]))
    print("precision:  ", np.mean(res[:, 1]), "±", np.std(res[:, 1]))
    print("recall:  ", np.mean(res[:, 2]), "±", np.std(res[:, 2]))

    row = pd.DataFrame({"method": ["BOCPD"], "F1_score_mean": np.mean(res[:, 0]), "F1_score_std": np.std(res[:, 0]),
                        "precision_mean": np.mean(res[:, 1]), "precision_std": np.std(res[:, 1]),
                        "recall_mean": np.mean(res[:, 2]), "recall_std": np.std(res[:, 2])})

    return row


# Hierarchical
def _objective_Hierarchical(trial, train, changepoints, tolerance_delay, order):
    nml_gaussian = partial(hsdmdl2_nml.nml_gaussian)
    min_datapoints = 5

    # delta_0だけはいかなる場合でもチューニングしなければならない
    delta_0 = trial.suggest_uniform('delta_0', 0.000001, 10.00)

    if order == 1:
        delta_1 = trial.suggest_uniform('delta_1', 0.000001, 1.00)
    else:
        delta_1 = 0.05

    if order == 2:
        delta_2 = trial.suggest_uniform('delta_2', 0.000001, 1.00)
    else:
        delta_2 = 0.05

    retrospective = hsdmdl2.Retrospective(encoding_func=nml_gaussian, d=2, M=5, min_datapoints=min_datapoints, delta_0=delta_0,
                                          delta_1=delta_1, delta_2=delta_2, how_to_drop='all', order=order, reliability=True)
    alarms = retrospective.make_alarms(train)

    scores = np.zeros(len(train))
    scores[alarms] = 1

    F1_score, _, _ = calc_F1_score(
        scores, changepoints, tolerance_delay, tuned_threshold=0.5)

    return -F1_score


# calculate the metrics
def _calc_Hierarchical_metrics(idx_data, dataset, changepoints, tolerance_delay, threshold, retrospective, order):
    _retrospective = deepcopy(retrospective)
    alarms = _retrospective.make_alarms(dataset[idx_data])
    scores = np.zeros(len(dataset[idx_data]))
    scores[alarms] = 1

    F1_score, precision, recall = calc_F1_score(
        scores, changepoints, tolerance_delay, threshold)
    return F1_score, precision, recall


def conduct_Hierarchical(n_trials, n_samples, dataset, changepoints, tolerance_delay, order):  # Hierarchical
    # hyperparameter tuning
    objective_Hierarchical = partial(_objective_Hierarchical, train=dataset[0],
                                     changepoints=changepoints, tolerance_delay=tolerance_delay, order=order)
    study = optuna.create_study()
    study.optimize(objective_Hierarchical, n_trials=n_trials, n_jobs=-1)

    opt_delta_0 = study.best_params['delta_0']

    if order == 1:
        opt_delta_1 = study.best_params['delta_1']
    else:
        opt_delta_1 = 0.05

    if order == 2:
        opt_delta_2 = study.best_params['delta_2']
    else:
        opt_delta_2 = 0.05

    min_datapoints = 5
    nml_gaussian = partial(hsdmdl2_nml.nml_gaussian)
    retrospective = hsdmdl2.Retrospective(encoding_func=nml_gaussian, d=2, M=5, min_datapoints=min_datapoints, delta_0=opt_delta_0,
                                          delta_1=opt_delta_1, delta_2=opt_delta_2, how_to_drop='all', order=True, reliability=True)

    # calculate metrics
    calc_metrics = partial(_calc_Hierarchical_metrics, dataset=dataset, changepoints=changepoints,
                           tolerance_delay=tolerance_delay, threshold=0.5, retrospective=retrospective, order=order)
    p = Pool(multi.cpu_count() - 1)
    args = list(range(1, n_samples))
    res = np.array(p.map(calc_metrics, args))
    p.close()

    # result
    print("F1 score:  ", np.mean(res[:, 0]), "±", np.std(res[:, 0]))
    print("precision:  ", np.mean(res[:, 1]), "±", np.std(res[:, 1]))
    print("recall:  ", np.mean(res[:, 2]), "±", np.std(res[:, 2]))

    method_name = "Hierarchical"
    if order == 0:
        method_name += "_0th"
    elif order == 1:
        method_name += "_1st"
    else:
        method_name += "_2nd"

    row = pd.DataFrame({"method": [method_name], "F1_score_mean": np.mean(res[:, 0]), "F1_score_std": np.std(res[:, 0]),
                        "precision_mean": np.mean(res[:, 1]), "precision_std": np.std(res[:, 1]),
                        "recall_mean": np.mean(res[:, 2]), "recall_std": np.std(res[:, 2])})

    return row


def _objective_AW2S_MDL(trial, train, changepoints, tolerance_delay):
    window_size = trial.suggest_int('window_size', 10, 500)
    sigma_given = trial.suggest_uniform('sigma_given', 0.1, 2)

    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=1e8,
                           div_min=1e-8, div_max=1e8)
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=1e8,
                                  div_min=1e-8, div_max=1e8)
    retrospective_first = sdmdl.Retrospective(h=window_size, encoding_func=nml_gaussian,
                                              complexity_func=complexity_gaussian, order=0)

    lnml_gaussian = partial(hsdmdl2_nml.lnml_gaussian, sigma_given=sigma_given)
    retrospective_second = hsdmdl2.Retrospective(encoding_func=lnml_gaussian, d=2, M=5, min_datapoints=5, delta_0=0.05,
                                                 delta_1=0.05, delta_2=0.05, how_to_drop='all', order=0, reliability=True)

    retrospective = aw2s_mdl.Retrospective(
        retrospective_first, retrospective_second)

    alarms = retrospective.make_alarms(train)
    scores = np.zeros(len(train))
    scores[alarms] = 1

    F1_score, _, _ = calc_F1_score(
        scores, changepoints, tolerance_delay, tuned_threshold=0.5)

    return -F1_score

# calculate the metrics


def _calc_AW2S_MDL_metrics(idx_data, dataset, changepoints, tolerance_delay, threshold, retrospective):
    _retrospective = deepcopy(retrospective)
    alarms = _retrospective.make_alarms(dataset[idx_data])
    scores = np.zeros(len(dataset[idx_data]))
    scores[alarms] = 1

    F1_score, precision, recall = calc_F1_score(
        scores, changepoints, tolerance_delay, threshold)
    return F1_score, precision, recall


def conduct_AW2S_MDL(n_trials, n_samples, dataset, changepoints, tolerance_delay):  # AW2S-MDL
    # hyperparameter tuning
    objective_AW2S_MDL = partial(_objective_AW2S_MDL, train=dataset[0],
                                 changepoints=changepoints, tolerance_delay=tolerance_delay)
    study = optuna.create_study()
    study.optimize(objective_AW2S_MDL, n_trials=n_trials, n_jobs=-1)

    opt_window_size = study.best_params['window_size']
    opt_sigma_given = study.best_params['sigma_given']

    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=1e8,
                           div_min=1e-8, div_max=1e8)
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=1e8,
                                  div_min=1e-8, div_max=1e8)
    retrospective_first = sdmdl.Retrospective(h=opt_window_size, encoding_func=nml_gaussian,
                                              complexity_func=complexity_gaussian, order=0)

    lnml_gaussian = partial(hsdmdl2_nml.lnml_gaussian,
                            sigma_given=opt_sigma_given)
    retrospective_second = hsdmdl2.Retrospective(encoding_func=lnml_gaussian, d=2, M=5, min_datapoints=5, delta_0=0.05,
                                                 delta_1=0.05, delta_2=0.05, how_to_drop='all', order=0, reliability=True)

    retrospective = aw2s_mdl.Retrospective(
        retrospective_first, retrospective_second)

    # calculate metrics
    calc_metrics = partial(_calc_AW2S_MDL_metrics, dataset=dataset, changepoints=changepoints,
                           tolerance_delay=tolerance_delay, threshold=0.5, retrospective=retrospective)
    p = Pool(multi.cpu_count() - 1)
    args = list(range(1, n_samples))
    res = np.array(p.map(calc_metrics, args))
    p.close()

    # result
    print("F1 score:  ", np.mean(res[:, 0]), "±", np.std(res[:, 0]))
    print("precision:  ", np.mean(res[:, 1]), "±", np.std(res[:, 1]))
    print("recall:  ", np.mean(res[:, 2]), "±", np.std(res[:, 2]))

    row = pd.DataFrame({"method": ["AW2S_MDL"], "F1_score_mean": np.mean(res[:, 0]), "F1_score_std": np.std(res[:, 0]),
                        "precision_mean": np.mean(res[:, 1]), "precision_std": np.std(res[:, 1]),
                        "recall_mean": np.mean(res[:, 2]), "recall_std": np.std(res[:, 2])})

    return row


def conduct_experiment(n_trials, n_samples, func, transition_period, tolerance_delay, random_seed=0):
    # fix seed for reproducibility
    np.random.seed(random_seed)
    mu_max = 1000
    div_min = 1e-8
    div_max = 1e8

    df_result = pd.DataFrame()

    print("Create Dataset")
    dataset, changepoints = create_dataset(n_samples, func, transition_period)

    print("ChangeFinder")
    row = conduct_CF(n_trials=n_trials, n_samples=n_samples, dataset=dataset,
                     changepoints=changepoints, tolerance_delay=tolerance_delay)
    df_result = pd.concat([df_result, row], axis=0)

    print("BOCPD")
    row = conduct_BOCPD(n_trials=n_trials, n_samples=n_samples, dataset=dataset,
                        changepoints=changepoints, tolerance_delay=tolerance_delay)
    df_result = pd.concat([df_result, row], axis=0)

    print("Hierarchical 0th")
    row = conduct_Hierarchical(n_trials=n_trials, n_samples=n_samples, dataset=dataset,
                               changepoints=changepoints, tolerance_delay=tolerance_delay, order=0)
    df_result = pd.concat([df_result, row], axis=0)

    print("Hierarchical 1st")
    row = conduct_Hierarchical(n_trials=n_trials, n_samples=n_samples, dataset=dataset,
                               changepoints=changepoints, tolerance_delay=tolerance_delay, order=1)
    df_result = pd.concat([df_result, row], axis=0)

    print("Hierarchical 2nd")
    row = conduct_Hierarchical(n_trials=n_trials, n_samples=n_samples, dataset=dataset,
                               changepoints=changepoints, tolerance_delay=tolerance_delay, order=2)
    df_result = pd.concat([df_result, row], axis=0)

    print("AW2S_MDL")
    row = conduct_AW2S_MDL(n_trials=n_trials, n_samples=n_samples, dataset=dataset,
                           changepoints=changepoints, tolerance_delay=tolerance_delay)
    df_result = pd.concat([df_result, row], axis=0)

    return df_result


if __name__ == '__main__':
    # parameters
    random_seed = 0
    n_trials = 5
    n_samples = 5
    tolerance_delay = 100
    transition_periods = [0, 100, 200, 300, 400]
    func_names = [(variance_changing, "variance_changing"),
                  (mean_changing, "mean_changing")]

    df_results = pd.DataFrame()

    for transition_period in transition_periods:
        for func_name in func_names:

            df_result = conduct_experiment(n_trials=n_trials, n_samples=n_samples, func=func_name[0],
                                           transition_period=transition_period, tolerance_delay=tolerance_delay,
                                           random_seed=random_seed)
            df_result["dataset"] = func_name[1]
            df_result["transition_period"] = transition_period
            df_result["n_trials"] = n_trials
            df_result["n_samples"] = n_samples
            df_result["tolerance_delay"] = tolerance_delay
            df_result["random_seed"] = random_seed

            df_result = df_result.reindex(columns=["method", "dataset", "transition_period", "F1_score_mean", "F1_score_std", "precision_mean",
                                                   "precision_std", "recall_mean", "recall_std", "n_trials", "n_samples", "tolerance_delay", "random_seed"])

            df_results = pd.concat([df_results, df_result], axis=0)
            df_results.to_csv("./results/F1_score_results.csv",
                              index=False)
