import sys
sys.path.append('./')
import optuna
import numpy as np
import pandas as pd
import multiprocessing as multi
import changefinder
import bocpd
import dmdl.sdmdl as sdmdl
import tsmdl.fw2s_mdl as fw2s_mdl
import utils.sdmdl_nml as sdmdl_nml
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from utils.utils import mean_changing, variance_changing, create_dataset, calc_AUC

# calculate the metrics


def _calc_metrics(idx_data, dataset, changepoints, tolerance_delay, retrospective):
    _retrospective = deepcopy(retrospective)
    scores = _retrospective.calc_scores(dataset[idx_data])
    AUC = calc_AUC(scores, changepoints, tolerance_delay)
    return AUC


def _objective_CF(trial, train, changepoints, tolerance_delay):  # CF
    # hyperparameters
    r = trial.suggest_uniform('r', 0.01, 0.99)
    order = trial.suggest_int('order', 1, 20)
    smooth = trial.suggest_int('smooth', 3, 20)

    retrospective = changefinder.Retrospective(r=r, order=order, smooth=smooth)
    scores = retrospective.calc_scores(train)
    AUC = calc_AUC(
        scores, changepoints, tolerance_delay)
    return -AUC


def conduct_CF(n_trials, n_samples, dataset, changepoints, tolerance_delay):  # CF
    # hyperparameter tuning
    objective_CF = partial(_objective_CF, train=dataset[0],
                           changepoints=changepoints, tolerance_delay=tolerance_delay)
    study = optuna.create_study()
    study.optimize(objective_CF, n_trials=n_trials, n_jobs=-1)
    opt_r = study.best_params['r']
    opt_order = study.best_params['order']
    opt_smooth = study.best_params['smooth']

    retrospective = changefinder.Retrospective(
        r=opt_r, order=opt_order, smooth=opt_smooth)

    # calculate metrics
    calc_metrics = partial(_calc_metrics, dataset=dataset,
                           changepoints=changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)
    p = Pool(multi.cpu_count() - 1)
    args = list(range(1, n_samples))
    res = np.array(p.map(calc_metrics, args))
    p.close()

    # result
    print("AUC:  ", np.mean(res[:]), "±", np.std(res[:]))

    row = pd.DataFrame({"method": ["ChangeFinder"], "AUC_mean": np.mean(
        res[:]), "AUC_std": np.std(res[:])})

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
    AUC = calc_AUC(scores, changepoints, tolerance_delay)

    return -AUC


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

    h = partial(bocpd.constant_hazard, opt_lam)
    lik = bocpd.StudentT(opt_alpha, opt_beta, opt_kappa, opt_mu)
    retrospective = bocpd.Retrospective(hazard_func=h, likelihood_func=lik)

    # calculate metrics
    calc_metrics = partial(_calc_metrics, dataset=dataset, changepoints=changepoints,
                           tolerance_delay=tolerance_delay, retrospective=retrospective)
    p = Pool(multi.cpu_count() - 1)
    args = list(range(1, n_samples))
    res = np.array(p.map(calc_metrics, args))
    p.close()

    # result
    print("AUC:  ", np.mean(res[:]), "±", np.std(res[:]))

    row = pd.DataFrame({"method": ["BOCPD"], "AUC_mean": np.mean(
        res[:]), "AUC_std": np.std(res[:])})

    return row



def _objective_SDMDL(trial, train, changepoints, tolerance_delay, params):  # S-MDL
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=params[
                           "mu_max"], div_min=params["div_min"], div_max=params["div_max"])
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=params[
                                  "mu_max"], div_min=params["div_min"], div_max=params["div_max"])
    window_size = trial.suggest_int('window_size', 10, 500)
    retrospective = sdmdl.Retrospective(h=window_size, encoding_func=nml_gaussian,
                                        complexity_func=complexity_gaussian, order=params["order"])

    scores = retrospective.calc_scores(train)

    AUC = calc_AUC(scores, changepoints,  tolerance_delay)
    return -AUC

def conduct_SDMDL(n_trials, n_samples, dataset, changepoints, tolerance_delay, params):  # S-MDL
    # hyperparameter tuning
    objective_SDMDL = partial(_objective_SDMDL, train=dataset[0],
                              changepoints=changepoints, tolerance_delay=tolerance_delay, params=params)
    study = optuna.create_study()
    study.optimize(objective_SDMDL, n_trials=n_trials, n_jobs=-1)
    opt_window_size = study.best_params['window_size']

    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=params["mu_max"],
                           div_min=params["div_min"], div_max=params["div_max"])
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=params["mu_max"],
                                  div_min=params["div_min"], div_max=params["div_max"])
    retrospective = sdmdl.Retrospective(h=opt_window_size, encoding_func=nml_gaussian,
                                        complexity_func=complexity_gaussian, order=params["order"])

    # calculate metrics
    calc_metrics = partial(_calc_metrics, dataset=dataset,
                           changepoints=changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)
    p = Pool(multi.cpu_count() - 1)
    args = list(range(1, n_samples))
    res = np.array(p.map(calc_metrics, args))
    p.close()

    print("AUC:  ", np.mean(res[:]), "±", np.std(res[:]))

    if params["order"] == 0:
        method_name = "SDMDL_0"
    elif params["order"] == 1:
        method_name = "SDMDL_1"
    else:
        method_name = "SDMDL_2"

    row = pd.DataFrame({"method": [method_name], "AUC_mean": np.mean(
        res[:]), "AUC_std": np.std(res[:])})

    return row


def _objective_FW2S_MDL(trial, train, changepoints, tolerance_delay, params):  # FW2S_MDL
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=params[
                           "mu_max"], div_min=params["div_min"], div_max=params["div_max"])
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=params[
                                  "mu_max"], div_min=params["div_min"], div_max=params["div_max"])
    window_size_1 = trial.suggest_int('window_size_1', 10, 500)
    window_size_2 = trial.suggest_int('window_size_2', 10, 500)

    retrospective_first = sdmdl.Retrospective(h=window_size_1, encoding_func=nml_gaussian,
                                        complexity_func=complexity_gaussian, order=0)
    retrospective_second = sdmdl.Retrospective(h=window_size_2, encoding_func=nml_gaussian,
                                               complexity_func=complexity_gaussian, order=0)
    retrospective = fw2s_mdl.Retrospective(
        retrospective_first, retrospective_second)
    scores = retrospective.calc_scores(train)

    AUC = calc_AUC(scores, changepoints,  tolerance_delay)
    return -AUC

def conduct_FW2S_MDL(n_trials, n_samples, dataset, changepoints, tolerance_delay, params):  # FW2S_MDL
    # hyperparameter tuning
    objective_FW2S_MDL = partial(_objective_FW2S_MDL, train=dataset[0],
                              changepoints=changepoints, tolerance_delay=tolerance_delay, params=params)
    study = optuna.create_study()
    study.optimize(objective_FW2S_MDL, n_trials=n_trials, n_jobs=-1)
    opt_window_size_1 = study.best_params['window_size_1']
    opt_window_size_2 = study.best_params['window_size_2']

    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=params["mu_max"],
                           div_min=params["div_min"], div_max=params["div_max"])
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=params["mu_max"],
                                  div_min=params["div_min"], div_max=params["div_max"])

    retrospective_first = sdmdl.Retrospective(h=opt_window_size_1, encoding_func=nml_gaussian,
                                        complexity_func=complexity_gaussian, order=0)
    retrospective_second = sdmdl.Retrospective(h=opt_window_size_2, encoding_func=nml_gaussian,
                                               complexity_func=complexity_gaussian, order=0)
    retrospective = fw2s_mdl.Retrospective(
        retrospective_first, retrospective_second)

    # calculate metrics
    calc_metrics = partial(_calc_metrics, dataset=dataset,
                           changepoints=changepoints, tolerance_delay=tolerance_delay, retrospective=retrospective)
    p = Pool(multi.cpu_count() - 1)
    args = list(range(1, n_samples))
    res = np.array(p.map(calc_metrics, args))
    p.close()

    print("AUC:  ", np.mean(res[:]), "±", np.std(res[:]))

    row = pd.DataFrame({"method": ["FW2S_MDL"], "AUC_mean": np.mean(
        res[:]), "AUC_std": np.std(res[:])})

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

    print("S-MDL 0th")
    params = {"mu_max": mu_max, "div_min": div_min,
              "div_max": div_max, "order": 0}
    row = conduct_SDMDL(n_trials=n_trials, n_samples=n_samples, dataset=dataset,
                        changepoints=changepoints, tolerance_delay=tolerance_delay, params=params)
    df_result = pd.concat([df_result, row], axis=0)

    print("S-MDL 1st")
    params = {"mu_max": mu_max, "div_min": div_min,
              "div_max": div_max, "order": 1}
    row = conduct_SDMDL(n_trials=n_trials, n_samples=n_samples, dataset=dataset,
                        changepoints=changepoints, tolerance_delay=tolerance_delay, params=params)
    df_result = pd.concat([df_result, row], axis=0)

    print("S-MDL 2nd")
    params = {"mu_max": mu_max, "div_min": div_min,
              "div_max": div_max, "order": 2}
    row = conduct_SDMDL(n_trials=n_trials, n_samples=n_samples, dataset=dataset,
                        changepoints=changepoints, tolerance_delay=tolerance_delay, params=params)
    df_result = pd.concat([df_result, row], axis=0)

    print("FW2S-MDL")
    params = {"mu_max": mu_max, "div_min": div_min,
              "div_max": div_max}
    row = conduct_FW2S_MDL(n_trials=n_trials, n_samples=n_samples, dataset=dataset,
                        changepoints=changepoints, tolerance_delay=tolerance_delay, params=params)
    df_result = pd.concat([df_result, row], axis=0)

    return df_result


def main():
    # parameters
    random_seed = 0
    n_trials = 100
    n_samples = 20
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

            df_result = df_result.reindex(columns=["method", "dataset", "transition_period", "AUC_mean", "AUC_std",
                                                   "n_trials", "n_samples", "tolerance_delay", "random_seed"])

            df_results = pd.concat([df_results, df_result], axis=0)
            df_results.to_csv("./results/AUC_results.csv",
                              index=False)

if __name__ == '__main__':
    main()
