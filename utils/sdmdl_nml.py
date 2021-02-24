import numpy as np
import scipy.linalg as sl
from scipy import special
import functools as fts
from utils.sir import calc_residual_error

# TODO: docstring


@fts.lru_cache(maxsize=None)
def multigamma_ln(a, d):
    """

    """
    return special.multigammaln(a, d)


@fts.lru_cache(maxsize=None)
def log_star(k):
    """
    calculate log star

    Args:
        k: log
    """
    ret = np.log(2.865)
    x = k
    while np.log(x) > 0:
        ret += np.log(x)
        x = np.log(x)
    return ret


def nml_gaussian(X, mu_max=100, div_min=0.1, div_max=5):
    """
        encode X by approximation of gaussian's NML by limiting integral domain.
    """
    n = len(X)
    Xmat = X.reshape((n, -1))
    sigma = np.std(Xmat)
    sigma = max(sigma, 1e-8)
    log_complexity = complexity_gaussian(
        n, mu_max=mu_max, div_min=div_min, div_max=div_max)

    log_pdf = n * (0.5 + np.log(sigma) + 0.5 * np.log(2 * np.pi))
    return log_pdf + log_complexity


@fts.lru_cache(maxsize=None)
def complexity_gaussian(h, mu_max=100, div_min=0.1, div_max=5):
    """
        X's approximate gaussian stochastic complexity by limiting integral domain.
    """
    return (1 / 2) * np.log(16 * mu_max * mu_max / (np.pi * div_min)) + (h / 2) * np.log(h / (2 * np.e)) - special.gammaln((h - 1) / 2)


def sir_gaussian(X, mu_max=1, div_min=1e-8, div_max=1e-2, gamma=0.1, beta_init=0.5, eps=1e-12):
    # X[0]: infectious
    # X[1]: removed
    n = X.shape[0]
    m = X.shape[1]
    error_cum = calc_residual_error(
        X[:, 0], X[:, 1], eps, gamma=gamma, beta_init=beta_init)

    return nml_gaussian(error_cum, mu_max=mu_max, div_min=div_min, div_max=div_max)


def nml_poisson(X, lmd_max):
    n = len(X)
    lmd_hat = np.mean(X)
    if lmd_hat == 0:
        neg_log = np.sum(special.gammaln(X + 1))
    else:
        neg_log = -n * lmd_hat * np.log(lmd_hat) + \
            n * lmd_hat + np.sum(special.gammaln(X + 1))
    cpl = complexity_poisson(n, lmd_max)
    return neg_log + cpl


def complexity_poisson(h, lmd_max):
    return 0.5 * np.log(h / (2 * np.pi)) + (1 + lmd_max / 2) * np.log(2) + log_star(lmd_max)


def nml_multgaussian(X, R=10e6, lambda_min=10e-6):  # ここ怪しい
    eps = 0.000001
    n = X.shape[0]
    m = X.shape[1]
    mu_hat = np.average(X, axis=0)
    res_X = X - mu_hat[np.newaxis, :]
    S = res_X.T.dot(res_X) / n
    # print(S)
    # print(S.shape)

    log_pdf = ((m * n) / 2) * np.log(2 * np.pi) + (n / 2) * np.log(np.abs(sl.det(S)) + eps) + \
        0.5 * np.trace(res_X.dot(sl.pinv(S)).dot(res_X.T)
                       )  # もっと効率的な計算方法があるはず。トレースと固有値の関係性?

    log_complexity = complexity_multgaussian(
        n, m, R=10e6, lambda_min=10e-6)
    # print(log_complexity)
    return log_pdf + log_complexity


@fts.lru_cache(maxsize=None)
def complexity_multgaussian(n, m, R=10e6, lambda_min=10e-6):  # 多分合ってる
    """
        X's approximate multi-dimensional gaussian stochastic complexity by limiting integral domain.
    """
    return (m + 1) * np.log(2) + (m / 2) * np.log(R) - ((m**2) / 2) * np.log(lambda_min) - (m + 1) * np.log(m) - special.gammaln(m / 2) + \
        (m * n / 2) * np.log(n / (2 * np.e)) - \
        special.multigammaln((n - 1) / 2, m)


def lnml_gaussian(X, sigma_given=1):
    multigammaln = multigamma_ln
    n = len(X)
    Xmat = X.reshape((n, -1))
    n, m = Xmat.shape
    if n <= 0:
        return np.nan
    nu = m  # given
    sigma = sigma_given  # given
    sum_x = np.sum(Xmat, axis=0)
    sum_xxT = sum([xi.reshape((m, -1)) @ xi.reshape((-1, m)) for xi in Xmat])
    mu = sum_x / (n + nu)
    S = (sum_xxT + nu * sigma ** 2 * np.identity(m)) / (n + nu) - mu.reshape((m, -1)) @ mu.reshape((-1, m))
    detS = sl.det(S)
    log_lnml = m / 2 * ((nu + 1) * np.log(nu) - n * np.log(np.pi) - (n + nu + 1) * np.log(n + nu)) + multigammaln(
        (n + nu) / 2, m) - multigammaln(nu / 2, m) + m * nu * np.log(sigma) - (n + nu) / 2 * np.log(detS)
    return -1 * log_lnml


def complexity_lnml_gaussian(h, m, sigma_given=1):
    multigammaln = multigamma_ln
    #n = len(X)
    #Xmat = X.reshape((n, -1))
    #n, m = Xmat.shape
    # if n <= 0:
    #    return np.nan
    n = 2 * h
    nu = m  # given
    sigma = sigma_given  # given
    #sum_x = np.sum(Xmat, axis=0)
    # sum_xxT = sum([xi.reshape((m, -1)) @ xi.reshape((-1, m)) for xi in Xmat])
    #mu = sum_x / (n + nu)
    # S = (sum_xxT + nu * sigma ** 2 * np.identity(m)) / (n + nu) - mu.reshape((m, -1)) @ mu.reshape((-1, m))
    #detS = sl.det(S)
    # log_lnml = m / 2 * ((nu + 1) * np.log(nu) - n * np.log(np.pi) - (n + nu + 1) * np.log(n + nu)) + multigammaln(
    #    (n + nu) / 2, m) - multigammaln(nu / 2, m) + m * nu * np.log(sigma) - (n + nu) / 2 * np.log(detS)
    log_C = -m * nu * np.log(sigma) + multigammaln(nu / 2, m) - multigammaln((nu + n) / 2, m) + 0.5 * m * (n + nu + 1) * np.log(
        nu + n) - 0.5 * m * (n + nu) * np.log(2) - 0.5 * m * (n + nu) - 0.5 * m * nu * np.log(np.pi) - 0.5 * m * (nu + 1) * np.log(nu)
    return log_C


def loss_regression(X):
    Xmat = np.matrix(X)
    n = Xmat.shape[0]
    if n <= 0:
        return np.nan
    W = np.ones((2, n))
    for i in range(0, n):
        W[1, i] = i + 1
    beta = sl.pinv(W.dot(W.T)).dot(W).dot(X)
    Xc = Xmat - W.T.dot(beta)
    var = Xc.T * Xc / n
    logpdf_max = n * (-1 - np.log(2 * np.pi) - np.log(var)) / 2
    capacity = np.log(n)
    return -logpdf_max + capacity


def loss_gaussian(X):
    """
        encode X by CNML
    """
    Xmat = np.matrix(X)
    n, m = Xmat.shape
    if n == 1:
        Xmat = Xmat.T
        n, m = Xmat.shape
    else:
        pass
    if n <= 0:
        return np.nan
    Xc = Xmat - np.mean(Xmat, 0)
    S = np.dot(Xc.T, Xc / n)
    detS = sl.det(S)
    logpdf_max = n * (-m - m * np.log(2 * np.pi) - np.log(detS)) / 2
    capacity = m * (m + 3) / 4 * np.log(n)  # approximation
    return -logpdf_max + capacity
