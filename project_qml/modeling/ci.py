import numpy as np


def mean_ci95(x):
    x = np.array(x, dtype=float)
    m = float(x.mean())
    s = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    half = 1.96 * s / max(np.sqrt(len(x)), 1.0)
    return m, (m - half), (m + half)

def mean_ci_t(x, alpha=0.05):
    """
    Publication fix: t-based CI (small n seeds).
    """
    import math
    # from scipy.stats import t as tdist
    try:
        from scipy.stats import t as tdist
        use_scipy = True
    except Exception:
        tdist = None
        use_scipy = False

    x = np.array(x, dtype=float)
    n = int(len(x))
    m = float(x.mean())
    if n <= 1:
        return m, m, m
    s = float(x.std(ddof=1))
    se = s / max(math.sqrt(n), 1e-12)
    # tcrit = float(tdist.ppf(1.0 - alpha/2.0, df=n-1))
    if use_scipy:
        tcrit = float(tdist.ppf(1.0 - alpha/2.0, df=n-1))
    else:
        tcrit = 1.96  # normal approx
    half = tcrit * se
    return m, (m - half), (m + half)

def mean_ci_bootstrap(x, B=2000, alpha=0.05, seed=0):
    """
    Publication option: bootstrap CI of the mean.
    """
    rng = np.random.default_rng(int(seed))
    x = np.array(x, dtype=float)
    n = int(len(x))
    if n <= 1:
        m = float(x.mean()) if n == 1 else 0.0
        return m, m, m
    means = []
    for _ in range(int(B)):
        samp = rng.choice(x, size=n, replace=True)
        means.append(float(np.mean(samp)))
    means = np.array(means, dtype=float)
    lo = float(np.quantile(means, alpha/2.0))
    hi = float(np.quantile(means, 1.0 - alpha/2.0))
    m  = float(np.mean(x))
    return m, lo, hi

