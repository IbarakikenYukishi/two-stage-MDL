import numpy as np
from functools import partial
from scipy import integrate
from scipy import optimize


def solve_v_ast(x_0, y_0, beta, gamma):  # v*を計算
    f = lambda v: x_0 * (1 - v) + y_0 + gamma / beta * np.log(v)
    ret = optimize.fsolve(f, 1e-8)
    return ret[0]


def h_2(xi, x_0, y_0, beta, gamma):  # h2の関数
    ret = 1 / (xi * (beta * x_0 * (1 - xi) + beta * y_0 + gamma * np.log(xi))
               ) - 1 / (xi * ((beta * x_0 - gamma) * (1 - xi) + beta * y_0))
    return ret


def integrated_h_1(v, x_0, y_0, beta, gamma):  # h1の積分の解析解
    if beta * (x_0 + y_0) == gamma:
        return (1 / (beta * y_0)) * (1 / v - 1)
    else:
        return (1 / (beta * (x_0 + y_0) - gamma)) * (np.log((beta * x_0 - gamma) * (1 - v) + beta * y_0) - np.log(beta) - np.log(y_0) - np.log(v))


def grid_h_2(a, b, x_0, y_0, beta, gamma):  # h2のある区間における積分値
    _h_2 = partial(h_2, x_0=x_0, y_0=y_0, beta=beta, gamma=gamma)
    ret = integrate.quad(_h_2, a, b)[0]
    return ret


def _prediction_SIR(y_0, z_0, gamma, beta, div=1000):  # SIRの時系列と時刻の組
    x_0 = 1 - y_0 - z_0
    v_ast = solve_v_ast(x_0, y_0, beta, gamma)
    # print(v_ast)

    if v_ast != 1:
        # divはvのv*~1を何回割るか
        v_grid = np.arange(1, v_ast, (v_ast - 1) / div)
        I_list = np.zeros(len(v_grid) - 1)
        t_list = np.zeros(len(v_grid))

        # Iの計算
        for i in range(div - 1):
            I_list[i] = grid_h_2(v_grid[i + 1], v_grid[i],
                                 x_0, y_0, beta, gamma)

        # tの計算
        t_list[0] = 0
        for i in range(1, div):  # h2分の計算
            t_list[i] = t_list[i - 1] + I_list[i - 1]
        for i in range(1, div):  # h1分の計算
            t_list[i] += integrated_h_1(v_grid[i], x_0, y_0, beta, gamma)

        # x,y,zの計算
        x = x_0 * v_grid
        y = gamma / beta * np.log(v_grid) - x_0 * v_grid + x_0 + y_0
        z = -gamma / beta * np.log(v_grid) + z_0

        return x, y, z, t_list

    else:

        t_list = [0, 1]
        x = [x_0, x_0]
        y = [y_0, y_0]
        z = [z_0, z_0]

        return x, y, z, t_list


def prediction_SIR(y_0, z_0, gamma, beta, time_length, div=1000):  # SIRの解析解を、離散的な時刻のものに変換する
    x, y, z, t_list = _prediction_SIR(y_0, z_0, gamma, beta, div)
    data_length = len(t_list)
    # print(t_list[0:100])
    # print(y)
    # print(z)

    x_disc = np.zeros(time_length)
    y_disc = np.zeros(time_length)
    z_disc = np.zeros(time_length)

    count = 0
    for i in range(time_length):
        while True:
            if count < data_length and i >= t_list[count]:
                count += 1
            else:
                break

        if count == data_length:
            x_disc[i] = x[count - 1]
            y_disc[i] = y[count - 1]
            z_disc[i] = z[count - 1]

        else:
            # linear interpolation
            if t_list[count] - t_list[count - 1] != 0:
                x_disc[i] = x[count - 1] * (t_list[count] - i) / (t_list[count] - t_list[count - 1]) + x[
                    count] * (i - t_list[count - 1]) / (t_list[count] - t_list[count - 1])
                y_disc[i] = y[count - 1] * (t_list[count] - i) / (t_list[count] - t_list[count - 1]) + y[
                    count] * (i - t_list[count - 1]) / (t_list[count] - t_list[count - 1])
                z_disc[i] = z[count - 1] * (t_list[count] - i) / (t_list[count] - t_list[count - 1]) + z[
                    count] * (i - t_list[count - 1]) / (t_list[count] - t_list[count - 1])
            else:
                x_disc[i] = x[count - 1]
                y_disc[i] = y[count - 1]
                z_disc[i] = z[count - 1]

    return x_disc, y_disc, z_disc


def _calculate_residual(c_list, gamma, y_0, z_0, beta):
    time_length = len(c_list)  # データの長さ
    log_eps = 1e-12
    _c_list = np.where(c_list < log_eps, log_eps, c_list)
    x, y, z = prediction_SIR(y_0, z_0, gamma, beta, time_length)
    c = y + z
    return _c_list - c


def calc_residual_error(infectious, removed, eps, gamma=0.1, beta_init=0.5):
    data_length = infectious.size
    cum_cases = np.array(infectious + removed)

    beta = solve_SIR_parameter(
        cum_cases, gamma, infectious[0], removed[0], beta_init=beta_init)

    _, pred_I, pred_R = prediction_SIR(
        infectious[0], removed[0], gamma, beta, data_length, div=1000)

    pred_C = pred_I + pred_R

    pred_C = np.where(pred_C < eps, eps, pred_C)
    cum_cases = np.where(cum_cases < eps, eps, cum_cases)

    residual_C = np.log(pred_C) - np.log(cum_cases)

    return residual_C


def solve_SIR_parameter(c_list, gamma, y_0, z_0, beta_init=0.5):
    objective = lambda beta: _calculate_residual(c_list, gamma, y_0, z_0, beta)
    ret = optimize.least_squares(objective, beta_init, bounds=(1e-8, 20))
    return ret.x[0]

'''
def solve_SIR_parameter(c_list, gamma, y_0, z_0, beta_init=1, alpha=1e-9, n_steps=1000, eps=1e-8, es_rounds=30):
    beta_prev = beta_init
    time_length = len(c_list)  # データの長さ
    log_eps = 1e-12
    es_count = 0
    es_eps = 1e-4
    _c_list = np.where(c_list < log_eps, log_eps, c_list)

    for i in range(n_steps):
        print(i)
        print(beta_prev)
        # 元々のパラメータでのシミュレーション
        _, y, z = prediction_SIR(y_0, z_0, gamma, beta_prev, time_length)
        c = y + z
        c = np.where(c < log_eps, log_eps, c)

        # betaをepsだけ進めた上でのシミュレーション
        _, y_beta, z_beta = prediction_SIR(
            y_0, z_0, gamma, beta_prev + eps, time_length)
        c_beta = y_beta + z_beta
        c_beta = np.where(c_beta < log_eps, log_eps, c_beta)
        grad_beta = 2 * np.sum((np.log(c) - np.log(_c_list))
                               * (np.log(c_beta) - np.log(c)) / eps)

        # gradient discent
        beta_prev -= alpha * grad_beta

        # early stopping
        if abs(alpha * grad_beta) < es_eps:
            es_count += 1
        else:
            es_count = 0
        if es_rounds < es_count:
            print("early_stopping")
            break

        if beta_prev < 1e-10:
            beta_prev = 1e-10

        if beta_prev > 10:
            beta_prev = 10

    return beta_prev
'''
