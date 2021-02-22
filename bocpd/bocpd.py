import numpy as np
from scipy import stats
from copy import deepcopy


class Prospective:
    """
    BOCPD (Prospective)
    """

    def __init__(self, hazard_func, likelihood_func, eps=1e-4):
        """
        Args:
            hazard_func: hazard function
            likelihood_func: likelihood function
        """
        self.__hazard_func = hazard_func
        self.__likelihood_func = likelihood_func
        self.__eps = eps
        self.__R_prev = np.ones(1)
        self.__run_length = 0

    def update(self, x):
        """
        calculate the score of the input datum

        Args:
            x: input datum

        Returns:
            float: score of the input datum
        """

        # Evaluate the predictive distribution for the new datum
        predprobs = self.__likelihood_func.pdf(x)
        self.__run_length = len(self.__R_prev)

        # Evaluate the hazard function for this interval
        H = self.__hazard_func(self.__run_length)

        # update the posterior probability of the run lengths
        R = np.zeros(self.__run_length + 1)
        R[1:self.__run_length + 1] = self.__R_prev[0:self.__run_length] * \
            predprobs * (1 - H)
        R[0] = np.sum(self.__R_prev[0:self.__run_length] * predprobs * H)

        # Renormalize for the numerical stability.
        R = R / np.sum(R)

        # Calculate the change score
        score_sequence = np.arange(0, len(R))
        score_sequence = 1.0 / (1.0 + score_sequence)
        score = score_sequence.dot(R)

        # Update the parameter and cut-off the tail of the probability of run lengths
        # the cumulative probability in the tail is less than self.__eps
        cum_prob = 0
        cut_point = 0
        for i in range(len(R)):
            cum_prob += R[len(R) - i - 1]
            if cum_prob > self.__eps:
                cut_point = len(R) - i
                break

        self.__R_prev = R[0:cut_point] / np.sum(R[0:cut_point])
        self.__likelihood_func.update_theta(x, cut_point)

        return score


class Retrospective:
    """
    BOCPD (Retrospective)
    """

    def __init__(self, hazard_func, likelihood_func, threshold=0.5):
        """
        Args:
            hazard_func: hazard function
            likelihood_func: likelihood function
            threshold: threshold for alarms.
        """
        self.__hazard_func = hazard_func
        self.__likelihood_func = likelihood_func
        self.__threshold = threshold

    def calc_scores(self, X):
        """
        calculate scores

        Args:
            X: input data

        Returns:
            ndarray: scores of the input data
        """
        likelihood_func = deepcopy(self.__likelihood_func)
        detector = Prospective(
            hazard_func=self.__hazard_func, likelihood_func=likelihood_func)
        scores = []
        for i in X:
            score = detector.update(i)
            scores.append(score)
        return np.array(scores)

    def make_alarms(self, X):
        """
        make alarms with the threshold

        Args:
            X: input data

        Returns:
            ndarray: indice of alarms
        """
        scores = self.calc_scores(X)

        # ignore warnings made by np.nan
        with np.errstate(invalid='ignore'):
            alarms = np.greater(
                scores,
                self.__threshold
            ).astype(int)

        alarms = np.where(alarms == 1)[0]

        return alarms


class StudentT():
    """
    Student's T distribution
    """

    def __init__(self, alpha, beta, kappa, mu):
        """
        Args:
            alpha: parameter of Student's T distribution
            beta: parameter of Student's T distribution
            kappa: parameter of Student's T distribution
            mu: parameter of Student's T distribution
        """
        self.__alpha_0 = self.__alpha = np.array([alpha])
        self.__beta_0 = self.__beta = np.array([beta])
        self.__kappa_0 = self.__kappa = np.array([kappa])
        self.__mu_0 = self.__mu = np.array([mu])

    def pdf(self, x):
        """
        Probability density function

        Args:
            x : observed datum

        Returns:
            float : pdf evaluated at each point x in data
        """
        return stats.t.pdf(
            x=x, df=2 * self.__alpha, loc=self.__mu,
            scale=np.sqrt(self.__beta * (self.__kappa + 1) /
                          (self.__alpha * self.__kappa))
        )

    def update_theta(self, x, cut_point):
        """
        update inner parameters

        Args:
            x : observed datum
        """
        mu_T0 = np.concatenate(
            (self.__mu_0, (self.__kappa * self.__mu + x) / (self.__kappa + 1)))
        kappa_T0 = np.concatenate((self.__kappa_0, self.__kappa + 1.))
        alpha_T0 = np.concatenate((self.__alpha_0, self.__alpha + 0.5))
        beta_T0 = np.concatenate(
            (self.__beta_0, self.__beta + (self.__kappa *
                                           (x - self.__mu)**2) / (2. * (self.__kappa + 1.)))
        )

        self.__mu = mu_T0[:cut_point]
        self.__kappa = kappa_T0[:cut_point]
        self.__alpha = alpha_T0[:cut_point]
        self.__beta = beta_T0[:cut_point]


def constant_hazard(lam, length):
    """
    constant hazard function

    Args:
        lam: lambda
        length: time length
    """
    return 1 / lam * np.ones(length)
