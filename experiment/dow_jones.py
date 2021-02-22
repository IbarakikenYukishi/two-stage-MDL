import sys
sys.path.append('./')
from utils.utils import calc_AUC
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import utils.hsdmdl2_nml as hsdmdl2_nml
import utils.hsdmdl1_nml as hsdmdl1_nml
import utils.sdmdl_nml as sdmdl_nml
import tsmdl.aw2s_mdl as aw2s_mdl
import tsmdl.fw2s_mdl as fw2s_mdl
import dmdl.hsdmdl2 as hsdmdl2
import dmdl.hsdmdl1 as hsdmdl1
import dmdl.sdmdl as sdmdl
from functools import partial
import pandas as pd
import optuna
import changefinder
import bocpd
import datetime as dt
import boto3
from settings import AWS_S3_BUCKET_NAME

# data frame
df = pd.read_csv('./data/dow-jones-closing.csv')
# starting and ending date
start_date = dt.datetime(1972, 7, 5)
end_date = dt.datetime(1975, 6, 30)
df['Date'] = pd.to_datetime(df['Date'])

df['Rate'] = 0
for i in range(1, len(df)):
    df.iloc[i, 2] = (df.iloc[i, 1] / df.iloc[i - 1, 1]) - 1

df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
df = df.reset_index(drop=True)

# starting points of gradual changes
cp1 = dt.datetime(1973, 1, 30)
cp2 = dt.datetime(1973, 10, 19)
cp3 = dt.datetime(1974, 8, 9)


# 1) the conviction of G. Gordon Liddy and James W. McCord, Jr. in the Watergate incident on January 30, 1973
# 2) the declaration of the oil embargo by the Organization of Petroleum Exporting Countries (OPEC) against the United States on October 19, 1973
# 3) the resignation of President Nixon on August 9, 1974.
cp1_index = len(df[df['Date'] <= cp1])
cp2_index = len(df[df['Date'] <= cp2])
cp3_index = len(df[df['Date'] <= cp3])
changepoints = [cp1_index, cp2_index, cp3_index]

# parameters
tolerance_delay = 25
n_trials = 200
both = True

# BOCPD


def _objective_BOCPD(trial, train, changepoints, tolerance_delay):
    lam = trial.suggest_int('lam', 2, 100)
    alpha = trial.suggest_uniform('alpha', 1e-8, 10)
    beta = trial.suggest_uniform('beta', 1e-8, 0.001)
    kappa = trial.suggest_uniform('kappa', 1e-8, 0.001)
    mu = 0
    #mu = trial.suggest_uniform('mu', -1, 1)

    h = partial(bocpd.constant_hazard, lam)
    lik = bocpd.StudentT(alpha, beta, kappa, mu)
    retrospective = bocpd.Retrospective(hazard_func=h, likelihood_func=lik)

    scores = retrospective.calc_scores(train)
    AUC = calc_AUC(scores, changepoints, tolerance_delay, both=both)

    return -AUC


# hyperparameter tuning
objective_BOCPD = partial(_objective_BOCPD, train=np.array(df['Rate']),
                          changepoints=changepoints, tolerance_delay=tolerance_delay)
study = optuna.create_study()
study.optimize(objective_BOCPD, n_trials=n_trials)

# calculate scores
opt_lam = study.best_params['lam']
opt_alpha = study.best_params['alpha']
opt_beta = study.best_params['beta']
opt_kappa = study.best_params['kappa']
#opt_mu = study.best_params['mu']
opt_mu = 0

h = partial(bocpd.constant_hazard, opt_lam)
lik = bocpd.StudentT(opt_alpha, opt_beta, opt_kappa, opt_mu)
retrospective = bocpd.Retrospective(hazard_func=h, likelihood_func=lik)

bocpd_scores = retrospective.calc_scores(np.array(df['Rate']))

# CF


def _objective_CF(trial, train, changepoints, tolerance_delay):
    # hyperparameters
    r = trial.suggest_uniform('r', 0.01, 0.99)
    order = trial.suggest_int('order', 1, 20)
    smooth = trial.suggest_int('smooth', 3, 20)

    retrospective = changefinder.Retrospective(r=r, order=order, smooth=smooth)
    scores = retrospective.calc_scores(train)
    AUC = calc_AUC(
        scores, changepoints, tolerance_delay, both=both)
    return -AUC


# hyperparameter tuning
objective_CF = partial(_objective_CF, train=np.array(df['Rate']),
                       changepoints=changepoints, tolerance_delay=tolerance_delay)
study = optuna.create_study()
study.optimize(objective_CF, n_trials=n_trials, n_jobs=-1)
opt_r = study.best_params['r']
opt_order = study.best_params['order']
opt_smooth = study.best_params['smooth']

retrospective = changefinder.Retrospective(
    r=opt_r, order=opt_order, smooth=opt_smooth)

cf_scores = retrospective.calc_scores(np.array(df['Rate']))

# SDMDL 0th


def _objective_SDMDL(trial, train, changepoints, tolerance_delay, params):
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=params[
                           "mu_max"], div_min=params["div_min"], div_max=params["div_max"])
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=params[
                                  "mu_max"], div_min=params["div_min"], div_max=params["div_max"])
    window_size = trial.suggest_int('window_size', 5, 60)
    retrospective = sdmdl.Retrospective(h=window_size, encoding_func=nml_gaussian,
                                        complexity_func=complexity_gaussian, order=params["order"])

    scores = retrospective.calc_scores(train)
    n = len(scores)
    scores[2 * window_size - 1: n] = scores[window_size - 1: n - window_size]
    scores[0:2 * window_size] = np.nan

    AUC = calc_AUC(scores, changepoints,  tolerance_delay, both=both)
    return -AUC


mu_max = 10
div_min = 1e-2
div_max = 1e2
params = {"mu_max": mu_max, "div_min": div_min,
          "div_max": div_max, "order": 0}

# 0th D-MDL
# hyperparameter tuning
objective_SDMDL = partial(_objective_SDMDL, train=np.array(df['Rate']),
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

sdmdl_0_scores = retrospective.calc_scores(np.array(df['Rate']))
n = len(sdmdl_0_scores)
sdmdl_0_scores[2 * opt_window_size -
               1: n] = sdmdl_0_scores[opt_window_size - 1: n - opt_window_size]
sdmdl_0_scores[0:2 * opt_window_size] = np.nan

# 1st D-MDL
params = {"mu_max": mu_max, "div_min": div_min,
          "div_max": div_max, "order": 1}
# hyperparameter tuning
objective_SDMDL = partial(_objective_SDMDL, train=np.array(df['Rate']),
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

sdmdl_1_scores = retrospective.calc_scores(np.array(df['Rate']))
n = len(sdmdl_1_scores)
sdmdl_1_scores[2 * opt_window_size -
               1: n] = sdmdl_1_scores[opt_window_size - 1: n - opt_window_size]
sdmdl_1_scores[0:2 * opt_window_size] = np.nan

# 2nd D-MDL
params = {"mu_max": mu_max, "div_min": div_min,
          "div_max": div_max, "order": 2}
# hyperparameter tuning
objective_SDMDL = partial(_objective_SDMDL, train=np.array(df['Rate']),
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

sdmdl_2_scores = retrospective.calc_scores(np.array(df['Rate']))
n = len(sdmdl_2_scores)
sdmdl_2_scores[2 * opt_window_size -
               1: n] = sdmdl_2_scores[opt_window_size - 1: n - opt_window_size]
sdmdl_2_scores[0:2 * opt_window_size] = np.nan

# FW2S_MDL


def _objective_FW2S_MDL(trial, train, changepoints, tolerance_delay, params):
    nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=params[
                           "mu_max"], div_min=params["div_min"], div_max=params["div_max"])
    complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=params[
                                  "mu_max"], div_min=params["div_min"], div_max=params["div_max"])
    delta_0 = trial.suggest_uniform('delta_0', 0.001, 0.499)
    window_size_1 = trial.suggest_int('window_size_1', 5, 60)
    window_size_2 = trial.suggest_int('window_size_2', 5, 60)

    retrospective_first = sdmdl.Retrospective(h=window_size_1, encoding_func=nml_gaussian,
                                              complexity_func=complexity_gaussian, delta_0=delta_0, order=0)
    retrospective_second = sdmdl.Retrospective(h=window_size_2, encoding_func=nml_gaussian,
                                               complexity_func=complexity_gaussian, order=0)
    retrospective = fw2s_mdl.Retrospective(
        retrospective_first, retrospective_second)
    scores = retrospective.calc_scores(train)

    n = len(scores)
    scores[2 * (window_size_1 + window_size_2): n] = scores[window_size_1 +
                                                            window_size_2: n - (window_size_1 + window_size_2)]
    scores[0:2 * (window_size_1 + window_size_2)] = np.nan

    AUC = calc_AUC(scores, changepoints,  tolerance_delay, both=both)
    return -AUC

mu_max = 10
div_min = 1e-2
div_max = 1e2
params = {"mu_max": mu_max, "div_min": div_min,
          "div_max": div_max}

# hyperparameter tuning
objective_FW2S_MDL = partial(_objective_FW2S_MDL, train=np.array(df['Rate']),
                             changepoints=changepoints, tolerance_delay=tolerance_delay, params=params)
study = optuna.create_study()
study.optimize(objective_FW2S_MDL, n_trials=n_trials, n_jobs=-1)
opt_delta_0 = study.best_params['delta_0']
opt_window_size_1 = study.best_params['window_size_1']
opt_window_size_2 = study.best_params['window_size_2']

nml_gaussian = partial(sdmdl_nml.nml_gaussian, mu_max=params["mu_max"],
                       div_min=params["div_min"], div_max=params["div_max"])
complexity_gaussian = partial(sdmdl_nml.complexity_gaussian, mu_max=params["mu_max"],
                              div_min=params["div_min"], div_max=params["div_max"])

retrospective_first = sdmdl.Retrospective(h=opt_window_size_1, encoding_func=nml_gaussian,
                                          complexity_func=complexity_gaussian, delta_0=opt_delta_0, order=0)
retrospective_second = sdmdl.Retrospective(h=opt_window_size_2, encoding_func=nml_gaussian,
                                           complexity_func=complexity_gaussian, order=0)
retrospective = fw2s_mdl.Retrospective(
    retrospective_first, retrospective_second)

fw2s_mdl_scores = retrospective.calc_scores(np.array(df['Rate']))
n = len(fw2s_mdl_scores)
fw2s_mdl_scores[2 * (opt_window_size_1 + opt_window_size_2): n] = fw2s_mdl_scores[opt_window_size_1 +
                                                                                  opt_window_size_2: n - (opt_window_size_1 + opt_window_size_2)]
fw2s_mdl_scores[0:2 * (opt_window_size_1 + opt_window_size_2)] = np.nan

sdmdl_0_scores[np.isnan(sdmdl_0_scores)] = np.nanmin(sdmdl_0_scores)
sdmdl_1_scores[np.isnan(sdmdl_1_scores)] = np.nanmin(sdmdl_1_scores)
sdmdl_2_scores[np.isnan(sdmdl_2_scores)] = np.nanmin(sdmdl_2_scores)
fw2s_mdl_scores[np.isnan(fw2s_mdl_scores)] = np.nanmin(fw2s_mdl_scores)

print("SDMDL 0th: ", calc_AUC(sdmdl_0_scores,
                              changepoints,  tolerance_delay, both=both))
print("SDMDL 1st: ", calc_AUC(sdmdl_1_scores,
                              changepoints,  tolerance_delay, both=both))
print("SDMDL 2nd: ", calc_AUC(sdmdl_2_scores,
                              changepoints,  tolerance_delay, both=both))
print("FW2S-MDL: ", calc_AUC(fw2s_mdl_scores,
                             changepoints,  tolerance_delay, both=both))
print("ChangeFinder: ", calc_AUC(
    cf_scores, changepoints,  tolerance_delay, both=both))
print("BOCPD: ", calc_AUC(bocpd_scores, changepoints,  tolerance_delay, both=both))

df_result = pd.DataFrame()
row = pd.DataFrame({"method: ": ["SDMDL 0th"], "AUC: ": calc_AUC(
    sdmdl_0_scores, changepoints,  tolerance_delay, both=both)})
df_result = pd.concat([df_result, row], axis=0)
row = pd.DataFrame({"method: ": ["SDMDL 1st"], "AUC: ": calc_AUC(
    sdmdl_1_scores, changepoints,  tolerance_delay, both=both)})
df_result = pd.concat([df_result, row], axis=0)
row = pd.DataFrame({"method: ": ["SDMDL 2nd"], "AUC: ": calc_AUC(
    sdmdl_2_scores, changepoints,  tolerance_delay, both=both)})
df_result = pd.concat([df_result, row], axis=0)
row = pd.DataFrame({"method: ": ["FW2S-MDL"], "AUC: ": calc_AUC(
    fw2s_mdl_scores, changepoints,  tolerance_delay, both=both)})
df_result = pd.concat([df_result, row], axis=0)
row = pd.DataFrame({"method: ": ["ChangeFinder"], "AUC: ": calc_AUC(
    cf_scores, changepoints,  tolerance_delay, both=both)})
df_result = pd.concat([df_result, row], axis=0)
row = pd.DataFrame({"method: ": ["BOCPD"], "AUC: ": calc_AUC(
    bocpd_scores, changepoints,  tolerance_delay, both=both)})
df_result = pd.concat([df_result, row], axis=0)

df_result.to_csv("dow_jones.csv", index=False)

plt.clf()
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(20, 20))

fontsize = 18

axes[0].plot(df['Date'], df['Rate'], label='Raw Data')
axes[0].set_ylabel('Return', fontsize=fontsize)
axes[0].vlines([cp1, cp2, cp3], np.min(df['Rate']),
               np.max(df['Rate']), color="black")
axes[0].set_ylim(np.min(df['Rate']), np.max(df['Rate']))
axes[0].legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=fontsize)
axes[0].set_xlim(start_date, end_date)
axes[0].grid(True)

axes[1].plot(df['Date'], sdmdl_0_scores, label='SD-MDL 0th')
axes[1].set_ylabel('Description Length', fontsize=fontsize)
axes[1].vlines([cp1, cp2, cp3], np.nanmin(sdmdl_0_scores) - 1,
               np.nanmax(sdmdl_0_scores) + 1, color="black")
axes[1].set_ylim(np.nanmin(sdmdl_0_scores) - 1, np.nanmax(sdmdl_0_scores) + 1)
axes[1].legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=fontsize)
axes[1].set_xlim(start_date, end_date)
axes[1].grid(True)

axes[2].plot(df['Date'], sdmdl_1_scores, label='SD-MDL 1st')
axes[2].set_ylabel('Description Length', fontsize=fontsize)
axes[2].vlines([cp1, cp2, cp3], 0, np.nanmax(sdmdl_1_scores), color="black")
axes[2].set_ylim(0, np.nanmax(sdmdl_1_scores))
axes[2].legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=fontsize)
axes[2].set_xlim(start_date, end_date)
axes[2].grid(True)

axes[3].plot(df['Date'], sdmdl_2_scores, label='SD-MDL 2nd')
axes[3].set_ylabel('Description Length', fontsize=fontsize)
axes[3].vlines([cp1, cp2, cp3], 0, np.nanmax(sdmdl_1_scores), color="black")
axes[3].set_ylim(0, np.nanmax(sdmdl_1_scores))
axes[3].legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=fontsize)
axes[3].set_xlim(start_date, end_date)
axes[3].grid(True)


axes[4].plot(df['Date'], fw2s_mdl_scores, label='FW2S-MDL')
axes[4].set_ylabel('Description Length', fontsize=fontsize)
axes[4].vlines([cp1, cp2, cp3], np.nanmin(fw2s_mdl_scores) - 1,
               np.nanmax(fw2s_mdl_scores) + 1, color="black")
axes[4].set_ylim(np.nanmin(fw2s_mdl_scores) - 1,
                 np.nanmax(fw2s_mdl_scores) + 1)
axes[4].legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=fontsize)
axes[4].set_xlim(start_date, end_date)
axes[4].grid(True)

axes[5].plot(df['Date'], cf_scores, label='ChangeFinder')
axes[5].set_ylabel('Change Score', fontsize=fontsize)
axes[5].vlines([cp1, cp2, cp3], np.nanmin(cf_scores) - 1,
               np.nanmax(cf_scores) + 1, color="black")
axes[5].set_ylim(np.nanmin(cf_scores) - 1, np.nanmax(cf_scores) + 1)
axes[5].legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=fontsize)
axes[5].set_xlim(start_date, end_date)
axes[5].grid(True)

axes[6].plot(df['Date'], bocpd_scores, label='BOCPD')
axes[6].set_ylabel('Change Score', fontsize=fontsize)
axes[6].vlines([cp1, cp2, cp3], np.nanmin(bocpd_scores),
               np.nanmax(bocpd_scores), color="black")
axes[6].set_ylim(np.nanmin(bocpd_scores), np.nanmax(bocpd_scores))
axes[6].legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=fontsize)
axes[6].set_xlim(start_date, end_date)
axes[6].grid(True)

fig.savefig("dow_jones.png")

# upload to s3
s3 = boto3.resource('s3')
bucket = s3.Bucket(AWS_S3_BUCKET_NAME)

bucket.upload_file("dow_jones.png", "dow_jones/" + "dow_jones.png")
bucket.upload_file("dow_jones.csv", "dow_jones/" + "dow_jones.csv")
