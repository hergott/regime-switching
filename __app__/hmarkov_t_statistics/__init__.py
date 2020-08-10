import logging
import json
import time

import numpy as np

import azure.functions as func          # pylint: disable=import-error
import azure.durable_functions as df    # pylint: disable=import-error

from .hmarkov_llf import hmarkov_llf
from .hessian import hessian


def main(inputs: str) -> str:
    res = json.loads(inputs)

    y = np.asarray(res['y'], dtype=np.float64)
    x = np.asarray(res['x'], dtype=np.float64)
    P = np.asarray(res['transition_matrix'], dtype=np.float64)
    epinit = np.asarray(res['smoothed'], dtype=np.float64)

    onesigma = res['onesigma']
    regimes = res['regimes']

    tstat_time = time.perf_counter()

    params = None

    for b in res['beta']:

        b_ = np.squeeze(np.asarray(b, dtype=np.float64))
        if np.ndim(b_) < 1:
            b_ = np.expand_dims(b_, axis=-1)

        if params is None:
            params = b_.copy()
        else:
            params = np.concatenate((params, b_))

    if onesigma:
        sig = res['sigma'][0]
    else:
        sig = res['sigma']

    sig = np.squeeze(sig)
    if np.ndim(sig) < 1:
        sig = np.expand_dims(sig, axis=-1)

    params = np.concatenate((params, sig))

    P = P.flatten()

    P = np.squeeze(P)
    if np.ndim(P) < 1:
        P = np.expand_dims(P, axis=-1)

    params = np.concatenate((params, P))

    epinit = epinit[:, 0]

    hess_info = {'y': y, 'x': x, 'regime_n': regimes,
                 'regime_t': None, 'regime_init': epinit, 'onesigma': onesigma}

    hess_eps = 1e-5
    H = hessian(params, eps=hess_eps, info=hess_info)

    Hi = np.linalg.inv(-1. * H)

    diag = np.diag(Hi)

    k = 0

    t_statistics = list()

    for xx, bb in zip(x, res['beta']):
        xvars = xx.shape[1]
        vr = diag[k:k+xvars]

        t = np.asarray(bb) / np.sqrt(vr)

        t_statistics.append(t.tolist())

        k += xvars

    t_statistics_time = time.perf_counter()-tstat_time

    out = {'t_statistics': t_statistics,
           't_statistics_time': t_statistics_time,
           'hessian': H.tolist()}

    return json.dumps(out)
