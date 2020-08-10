'''
This code estimates the regime-switching model of James D. Hamilton:

[1] Hamilton, James D. (1989), “A New Approach to the Economic Analysis of Nonstationary 
Time Series and the Business Cycle,” Econometrica 57, 357-384.

[2] Hamilton, James D. (1994), Time Series Analysis, Princeton, NJ: Princeton University Press.

This code doesn't call an external optimizer. Instead, it uses the iterative
routine specified by James D. Hamilton.

This is the original model, and it gives a good example of the use of Azure Durable
Functions. But there has been a tremendous amount of innovation in regime-switching models since 
this one. See, for example, the RISE toolbox: https://github.com/jmaih/RISE_toolbox

'''

import math
import time
import json
import logging

import numpy as np  # pylint: disable=import-error

from datetime import datetime


def main(inputs: str) -> str:

    data = json.loads(inputs)

    # record start time
    now = datetime.now()
    start_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    y = np.asarray(data.get('y'), dtype=np.float64)

    regimes = data.get('regimes')

    maxit = data.get('maxit')
    onesigma = data.get('onesigma')
    disp = data.get('disp')
    convergence = data.get('convergence')

    run_number = data.get('run_number')

    if disp:
        logs = list()

    x = list()

    for x_in in data.get('x'):
        if x_in is None:
            x.append(None)
        else:
            x.append(np.asarray(x_in, dtype=np.float64))

    out = None

    if y.ndim == 1:
        y = np.expand_dims(y, axis=-1)

    n = y.shape[0]

    vars_x = list()
    total_x = 0

    for r in range(regimes):
        if x[r] is None:
            x[r] = np.ones((n, 1))
        else:
            if x[r].ndim == 1:
                x[r] = np.expand_dims(x[r], axis=-1)
            x[r] = np.hstack((np.ones((n, 1)), x[r]))

        k = x[r].shape[1]
        total_x += k
        vars_x.append(k)

    t0 = time.perf_counter()

    eptt = np.zeros((regimes, n))
    eptplus = np.zeros((regimes, n))
    sm = np.zeros((regimes, n))
    gamma = np.zeros((regimes, n))

    sigma = np.zeros((regimes,))

    beta = list()

    for r in range(regimes):
        beta.append(np.random.rand(1, vars_x[r]) - 0.5)

    ssr = 0
    for r in range(regimes):
        err = y - np.sum(x[r] * beta[r], axis=-1, keepdims=True)
        sse = np.dot(err.T, err)[0]
        ssr += sse
        sigma[r] = np.sqrt(sse / (n - vars_x[r]))

    if onesigma:
        sigma = np.full((regimes,), np.sqrt(ssr / (n - total_x)))

    P = np.ones((regimes, regimes)) * (1. / regimes)
    epinit = np.ones((regimes,)) * (1. / regimes)

    converge = 100000000.
    n_iter = 0

    while ((converge / total_x) > convergence) and (n_iter < maxit):

        # Calculate y probabilities. Hamilton 1994 page 696.
        for r in range(regimes):
            resid = y - np.sum(x[r] * beta[r], axis=-1, keepdims=True)

            c = 1. / (math.sqrt(2. * math.pi) * sigma[r])
            numer = -1. * (resid ** 2.)
            denom = 2. * (sigma[r] ** 2.)

            g = c * np.exp(numer / denom)

            gamma[r, :] = g.T

        # Iterate on Hamilton's equations 22.4.5 and 22.4.6.
        for t in range(n):
            if t == 0:
                qq = epinit * gamma[:, 0]
            else:
                qq = eptplus[:, t - 1] * gamma[:, t]

            try:
                div = qq / np.sum(qq)
            except RuntimeWarning as rtw:
                print('Warning in divide: qq / np.sum(qq)')
                print(f'qq: {qq}')
                print(f'beta: {beta}')
                print(f'Sum qq: {np.sum(qq)}')
                print(rtw)
                return None

            eptt[:, t] = div
            eptplus[:, t] = np.matmul(P, div)

        # Calculate smoothed probabilities. Hamilton equation 22.4.14.
        sm[:, -1] = eptt[:, -1]
        for t in reversed(range(n - 1)):
            sm[:, t] = eptt[:, t] * \
                np.matmul(P.T, (sm[:, t + 1] / eptplus[:, t]))

        # Enter 1st smoothed transition vector as new starting value. [Hamilton equation 22.4.17.]
        epinit = sm[:, 0]

        # Do transition matrix. Hamilton's 22.4.16. Remember elements in transition matrix
        # counterintuitive: p(ij) is row j, col i [22.2.3]
        for i in range(regimes):
            for j in range(regimes):
                num = np.sum(sm[j, 1:n] * sm[i, :(n - 1)])
                den = np.sum(sm[i, :(n - 1)])
                P[j, i] = num / den

        # Do weighted regression.  [Hamilton 22.4.21 through 22.4.23]
        oldbeta = beta.copy()
        beta = list()
        ssr = 0.
        for r in range(regimes):
            qq = np.reshape(np.sqrt(sm[r, :]), (n, 1))
            yy = y * qq
            xx = x[r] * (qq * np.ones((1, vars_x[r])))

            reg_b, _, _, _ = np.linalg.lstsq(xx, yy, rcond=None)
            new_beta = np.reshape(reg_b, (1, vars_x[r]))

            beta.append(new_beta)
            resid = yy - np.sum(xx * new_beta, axis=-1, keepdims=True)
            err = np.dot(resid.T, resid)

            ssr += err
            sigma[r] = np.sqrt(err / (n - vars_x[r]))

        if onesigma:
            sigma = np.full((regimes,), np.sqrt(ssr / (n - total_x)))

        # Convergence figure is the sum in absolute differences between old and new beta coefficient
        # estimates.
        converge = 0.
        for r in range(regimes):
            tmp = np.max(np.abs(oldbeta[r] - beta[r]))
            if tmp > converge:
                converge = tmp

        if disp and n_iter % 10 == 0:
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_str = f'{current_time} run #: {run_number}  n_iter: {n_iter}  converge: {converge}  sigma: {sigma}'
            logs.append(log_str)

            logging.info(log_str)

        n_iter += 1  # END WHILE LOOP

    # Calculate log liklihood function. [22.4.7 and 22.4.8]
    ones = np.ones((1, regimes))
    llf = np.log(np.dot(ones, epinit * gamma[:, 0]))
    for t in range(1, n):
        llf += np.log(np.dot(ones, eptplus[:, t - 1] * gamma[:, t]))

    out = dict(iterations=n_iter, llf=llf, converge=converge, sigma=sigma.copy(), smoothed=sm.copy(), prob=eptt.copy(), pred=eptplus.copy(),
               transition_matrix=P.copy(), err_msg=None, total_x=total_x, nobs=n, beta=[b.copy().tolist() for b in beta])

    # Calculate ergodic (unconditional) vector [22.2.25-22.2.26].
    p = out['transition_matrix']
    row_ones = np.ones((1, regimes))
    A = np.vstack((np.identity(regimes) - p, row_ones))
    ee = np.zeros((regimes + 1, 1))
    ee[-1, :] = 1.
    ergodic_result = np.linalg.inv(A.T@A)@A.T@ee
    out['ergodic'] = np.squeeze(ergodic_result, axis=-1)

    # This is error checking for the above unconditional calculations. The above uses the matrix equations
    # specified by Hamilton. For the two-regime model, specified unconditional figures are available (see
    # Hamilton's page 683). So if the model has two regimes, it will calculate the mechanical undonditional
    # probabilities below and compare with the matrix results above.
    if regimes == 2:
        erg0 = (1. - p[1, 1]) / (2. - p[0, 0] - p[1, 1])
        erg1 = (1. - p[0, 0]) / (2. - p[0, 0] - p[1, 1])
        diff = abs(erg0 - out['ergodic'][0]) + abs(erg1 - out['ergodic'][1])
        if diff > 1e-6:
            erg_err = f'Warning:: Discrepancy calculating ergodic vector. ergodic: {out["ergodic"]}  erg0:{erg0}  erg1:{erg1}'
            out['err_msg'] = erg_err

    # Compute y-hat and R-squared.
    yreg = np.zeros((n, regimes))
    resid = np.zeros((n, regimes))
    for r in range(regimes):
        coeff = np.asarray(out["beta"][r], dtype=np.float64)
        yreg[:, r] = np.sum(x[r]*np.reshape(coeff,
                                            (1, vars_x[r])), axis=-1, keepdims=False)
        resid[:, r] = np.squeeze(y, axis=-1)-yreg[:, r]

    smthd = out["smoothed"]
    pred_y = np.sum(yreg*smthd.T, axis=-1, keepdims=True)
    err_y = y-pred_y

    out['sse'] = np.dot(err_y.T, err_y)
    out['sst'] = np.dot(y.T, y)-n*(np.mean(y)**2.)
    out['rsqr'] = 1.-out['sse']/out['sst']
    out['rbar'] = 1.-((n-1.)/(n-total_x))*(1.-out['rsqr'])
    out['yhat'] = np.squeeze(pred_y, axis=-1)
    out['residuals'] = np.squeeze(err_y, axis=-1)
    out['y'] = y.copy().tolist()
    out['x'] = [x_sub.copy().tolist() for x_sub in x]
    out['regimes'] = regimes
    out['onesigma'] = onesigma

    out['time_solve'] = time.perf_counter() - t0

    if disp:
        out['log_info'] = logs

    # Convert numpy arrays to lists for JSON.
    for key in out:
        if isinstance(out[key], (np.ndarray, np.generic)):
            if np.ndim(out[key]) > 1 and out[key].shape[1] == 1:
                out[key] = np.squeeze(out[key], axis=-1)
            if np.ndim(out[key]) < 1:
                out[key] = float(out[key])
            elif max(out[key].shape) == 1:
                out[key] = float(out[key][0])
            else:
                out[key] = out[key].tolist()

    # record end time
    now = datetime.now()
    end_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    out['end_time'] = end_time
    out['start_time'] = start_time
    out['run_number'] = run_number

    return json.dumps(out)
