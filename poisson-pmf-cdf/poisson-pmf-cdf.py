import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    pmfs = np.zeros(k + 1)

    pmfs[0] = np.exp(-lam)

    for i in range(1, k + 1):
        pmfs[i] = pmfs[i - 1] * (lam / i)

    pmf = pmfs[k]
    cdf = np.sum(pmfs)

    return pmf, cdf