import numpy as np
from .hmarkov_llf import hmarkov_llf


def approx_fprime(x0, eps, info):

    def get_llf(x_vec):
        llf = hmarkov_llf(x_vec, info['y'], info['x'], info['regime_n'],
                          info['regime_t'], info['regime_init'], info['onesigma'])
        return llf

    n = x0.shape[0]

    x_copy = x0.copy()

    llf_base = get_llf(x_copy)

    fp = np.zeros((n, ))

    for i in range(n):
        x_orig = x_copy[i]
        x_copy[i] = x_orig+eps

        llf_adj = get_llf(x_copy)

        fp[i] = (llf_adj - llf_base)/eps

        x_copy[i] = x_orig

    return fp


def hessian(x0, eps, info, central=False):
    f1 = approx_fprime(x0, eps=eps, info=info)

    n = x0.shape[0]
    hessian = np.zeros((n, n))

    x_copy = x0.copy()

    for i in range(n):
        x_orig = x_copy[i]

        x_copy[i] = x_orig + eps
        f2 = approx_fprime(x_copy, eps=eps, info=info)

        hessian[:, i] = (f2 - f1)/eps

        x_copy[i] = x_orig

    return hessian
