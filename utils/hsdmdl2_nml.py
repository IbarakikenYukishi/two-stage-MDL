import numpy as np
import scipy.linalg as sl
from scipy import special
import functools as fts

# TODO: docstring

@fts.lru_cache(maxsize=None)
def multigamma_ln(a, d):
    return special.multigammaln(a, d)


def lnml_gaussian(bucket, sigma_given=0.3):

    # calculate the lnml code length of a bucket assuming gaussian family

    multigammaln = multigamma_ln
    n, m = float(bucket[1]), bucket[0][0].shape[1]
    if n <= 0:
        return np.nan
    nu = m  # given (optimal when nu = m)
    sigma = sigma_given  # given
    mu = bucket[0][0] / (n + nu)
    S = (bucket[0][1] + nu * sigma ** 2 *
         np.matrix(np.identity(m))) / (n + nu) - mu.T.dot(mu)
    detS = sl.det(S.astype(np.float64))
    log_lnml = m / 2 * ((nu + 1) * np.log(nu) - n * np.log(np.pi) - (n + nu + 1) * np.log(n + nu)) + multigammaln(
        (n + nu) / 2, m) - multigammaln(nu / 2, m) + m * nu * np.log(sigma) - (n + nu) / 2 * np.log(detS)
    return -1 * log_lnml


def nml_gaussian(bucket, mu_max=1e8, div_min=1e-8, div_max=1e8):
    n = float(bucket[1])
    if n <= 1:
        assert "The number of total datapoints should be more than 1"
    me_x = bucket[0][0] / n
    sq_x = bucket[0][1] / n
    sigma = sq_x - me_x**2

    log_complexity = app_complexity_gaussian(
        n, mu_max=mu_max, div_min=div_min, div_max=div_max)

    log_pdf = n * (0.5 + np.log(sigma + 0.0000001) + 0.5 * np.log(2 * np.pi))

    return (log_pdf + log_complexity)


def app_complexity_gaussian(h, mu_max=1e8, div_min=1e-8, div_max=1e8):
    """
        X's approximate gaussian stochastic complexity by limiting integral domain.
    """
    if h != 1:
        return (1 / 2) * np.log(16 * mu_max * mu_max / (np.pi * div_min)) + (h / 2) * np.log(h / (2 * np.e)) - special.gammaln((h - 1) / 2)
    else:
        return (1 / 2) * np.log(16 * mu_max * mu_max / (np.pi * div_min)) + (h / 2) * np.log(h / (2 * np.e))
