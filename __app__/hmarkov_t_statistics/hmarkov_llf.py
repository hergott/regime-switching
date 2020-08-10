'''
This code estimates the likelihood function for the regime-switching 
model of James D. Hamilton:

[1] Hamilton, James D. (1989), “A New Approach to the Economic Analysis of Nonstationary 
Time Series and the Business Cycle,” Econometrica 57, 357-384.

[2] Hamilton, James D. (1994), Time Series Analysis, Princeton, NJ: Princeton University Press.

This code doesn't call an external optimizer. Instead, it uses the iterative
routine specified by James D. Hamilton.

This is the original model, and it gives a good example of the use of Azure Durable
Functions. But there has been a tremendous amount of innovation in regime-switching models since 
this one. See, for example, the RISE toolbox: https://github.com/jmaih/RISE_toolbox

'''

import numpy as np  # pylint: disable=import-error
import math


def hmarkov_llf(params, y, x, regime_n, regime_t, regime_init, onesigma):
    '''
    y:          an (n,1) vector for the dependent variable.
    x:          a list of items, each item is an (n,k) matrix of 
                explanatory variables. The number of variables can be different for each regime.
    regime_n:   the number of regimes
    regime_t:   this is the index of the sample to calculate the log
                likelihood function. If 'regime_t' is None, the llf for
                the entire sequence is returned.
    regime_init:
    onesigma:   a boolean whether to use the same variance for all
                regimes.

    '''
    regimes = regime_n
    T = regime_t
    epinit = regime_init

    n = y.shape[0]

    if T is None:
        full_llf = True
        T = n
    else:
        full_llf = False

    # Initialize variables.
    beta = list()
    k = list()
    counter = 0
    for r in range(regimes):
        k_ = x[r].shape[1]
        temp = params[counter: (counter+k_)]
        beta.append(np.reshape(temp, (1, k_)))
        k.append(k_)
        counter += k_

    if onesigma:
        sigma_ = params[counter:(counter+1)]
        sigma = np.full((regimes,), sigma_)
        counter += 1
    else:
        sigma = params[counter:(counter+regimes)]
        counter += regimes

    P = np.reshape(params[counter:], (regimes, regimes))

    eptt = np.zeros((regimes, n))
    eptplus = np.zeros((regimes, n))
    gamma = np.zeros((regimes, n))

    if y.ndim == 1:
        y = np.expand_dims(y, axis=-1)

    # Calculate y probabilities. Hamilton 1994 page 696.
    for r in range(regimes):

        resid = y - np.sum(x[r] * beta[r], axis=-1, keepdims=True)

        c = 1. / (math.sqrt(2. * math.pi) * sigma[r])
        numer = -1. * (resid ** 2.)
        denom = 2. * (sigma[r] ** 2.)

        g = c * np.exp(numer / denom)

        gamma[r, :] = g.T

    # Iterate on Hamilton's equations 22.4.5 and 22.4.6.
    for t in range(T):
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

    # Calculate log liklihood function. [22.4.7 and 22.4.8]
    ones = np.ones((1, regimes))
    if full_llf:
        llf = np.log(np.dot(ones, epinit*gamma[:, 0]))
        for t in range(1, T):
            llf += np.log(np.dot(ones, eptplus[:, t-1]*gamma[:, t]))

    else:
        if T == 0:
            llf = np.log(np.dot(ones, epinit*gamma[:, T]))
        else:
            llf = np.log(np.dot(ones, eptplus[:, T-1]*gamma[:, T]))

    return llf
