import numpy as np
import scipy.linalg as sl
from scipy import special
import functools as fts

# TODO: docstring

@fts.lru_cache(maxsize=None)
def multigamma_ln(a, d):
    return special.multigammaln(a, d)

def nml_regression(X, sum_x, sum_xxT, R=1, var_min=1e-12):
    Xmat = np.matrix(X)

    n = Xmat.shape[0]
    if n <= 0:
        return np.nan

    W = np.ones((2, n))
    W[1, :] = np.arange(1, n + 1)

    beta = sl.pinv(W.dot(W.T)).dot(W).dot(X)
    Xc = Xmat - W.T.dot(beta)
    var = float(Xc.T * Xc / n)

    eps=1e-12
    var=max(var, eps)

    return n * np.log(var)/2 + np.log(R / var_min) - special.gammaln(n / 2 - 1) + n * np.log(n * np.pi) / 2


def lnml_gaussian(X, sum_x, sum_xxT, sigma_given=1):
    """
    Calculate LNML code length of Gaussian distribution. See the paper below:
    Miyaguchi, Kohei. "Normalized Maximum Likelihood with Luckiness for Multivariate Normal Distributions." 
    arXiv preprint arXiv:1708.01861 (2017).

    parameters: 
        X: data sequence
        sum_x: mean sequence
        sum_xxT: variance sequence
        sigma_given: hyperparameter for prior distribution

    returns:
        LNML code length
    """

    multigammaln = multigamma_ln
    n, m = X.shape
    if n <= 0:
        return np.nan
    nu = m  # given
    sigma = sigma_given  # given
    mu = sum_x / (n + nu)
    S = (sum_xxT + nu * sigma ** 2 * np.matrix(np.identity(m))) / \
        (n + nu) - mu.T.dot(mu)
    detS = sl.det(S)
    log_lnml = m / 2 * ((nu + 1) * np.log(nu) - n * np.log(np.pi) - (n + nu + 1) * np.log(n + nu)) + multigammaln(
        (n + nu) / 2, m) - multigammaln(nu / 2, m) + m * nu * np.log(sigma) - (n + nu) / 2 * np.log(detS)
    return -1 * log_lnml


def complexity_lnml_gaussian(h, m, sigma_given=1):
    """
    Calculate stochastic complexity of LNML of Gaussian distribution. See the paper below:
    Miyaguchi, Kohei. "Normalized Maximum Likelihood with Luckiness for Multivariate Normal Distributions." 
    arXiv preprint arXiv:1708.01861 (2017).

    parameters: 
        h: half window size
        m: dimension of data
        sigma_given: hyperparameter for prior distribution

    returns:
        stochastics complexity
    """
    multigammaln = multigamma_ln
    n = h
    nu = m  # given
    sigma = sigma_given  # given
    log_C = -m * nu * np.log(sigma) + multigammaln(nu / 2, m) - multigammaln((nu + n) / 2, m) + 0.5 * m * (n + nu + 1) * np.log(
        nu + n) - 0.5 * m * (n + nu) * np.log(2) - 0.5 * m * (n + nu) - 0.5 * m * nu * np.log(np.pi) - 0.5 * m * (nu + 1) * np.log(nu)
    return log_C


def nml_poisson(X, sum_x, sum_xxT, lmd_max=100):
    """
    Calculate NML code length of Poisson distribution. See the paper below:
    yamanishi, Kenji, and Kohei Miyaguchi. "Detecting gradual changes from data stream using MDL-change statistics." 
    2016 IEEE International Conference on Big Data (Big Data). IEEE, 2016.

    parameters: 
        X: data sequence
        sum_x: mean sequence
        sum_xxT: variance sequence
        lmd_max: the maximum value of lambda

    returns:
        NML code length
    """
    n = len(X)
    lmd_hat = sum_x / n
    if lmd_hat == 0:
        neg_log = np.sum(special.gammaln(X + 1))
    else:
        neg_log = -n * lmd_hat * np.log(lmd_hat) + \
            n * lmd_hat + np.sum(special.gammaln(X + 1))
    cpl = complexity_poisson(n, lmd_max)
    return neg_log + cpl


def complexity_poisson(h, lmd_max=100):
    """
    Calculate stochastic complexity of Poisson distribution. See the paper below:
    yamanishi, Kenji, and Kohei Miyaguchi. "Detecting gradual changes from data stream using MDL-change statistics." 
    2016 IEEE International Conference on Big Data (Big Data). IEEE, 2016.

    parameters: 
        h: half window size
        lmd_max: the maximum value of lambda

    returns:
        stochastics complexity
    """
    return 0.5 * np.log(h / (2 * np.pi)) + (1 + lmd_max / 2) * np.log(2) + log_star(lmd_max)


def log_star(k):
    ret = np.log(2.865)
    x = k
    while np.log(x) > 0:
        ret += np.log(x)
        x = np.log(x)
    return ret
