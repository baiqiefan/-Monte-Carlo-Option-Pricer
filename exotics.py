"""Path-dependent ("exotic") options priced by Monte Carlo.

A European option only cares where the stock ENDS. These options care
about the whole journey, so we simulate full price paths:

  Asian call    pays max(average price along the path - K, 0)
  Barrier call  a normal call that is cancelled ("up-and-out") or only
                switched on ("up-and-in") if the stock touches a barrier

The arithmetic Asian option has no exact formula; this is where Monte
Carlo is genuinely needed.
"""
import numpy as np

from black_scholes import (EXAMPLE, call_payoff, discount_factor, norm_cdf,
                           black_scholes_call)
from monte_carlo import summarize, control_variate_adjust


def simulate_paths(S, T, r, sigma, n_steps, n_sims, rng):
    """Simulate n_sims full price paths with n_steps time steps each.

    Returns an array of shape (n_sims, n_steps + 1). Column 0 is today's
    price S; the last column is the price at expiry.

    Each step multiplies the price by exp(small drift + small random shock).
    Multiplying many exponentials is the same as adding up their exponents,
    which is why a cumulative sum (cumsum) builds the whole path at once.
    """
    dt = T / n_steps
    Z = rng.standard_normal((n_sims, n_steps))
    log_steps = (r - sigma**2/2) * dt + sigma * np.sqrt(dt) * Z
    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = S
    paths[:, 1:] = S * np.exp(np.cumsum(log_steps, axis=1))
    return paths


def geometric_asian_call_exact(S, K, T, r, sigma, n_steps):
    """Exact price of a GEOMETRIC-average Asian call (n_steps monitoring dates).

    The geometric average of lognormal prices is itself lognormal, so a
    Black-Scholes-style formula exists. We use it as an answer key and as
    a control variate for the arithmetic Asian call.
    """
    n = n_steps
    mean = np.log(S) + (r - sigma**2/2) * T * (n + 1) / (2 * n)
    var = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    d1 = (mean - np.log(K) + var) / np.sqrt(var)
    d2 = d1 - np.sqrt(var)
    return float(discount_factor(r, T)
                 * (np.exp(mean + var / 2) * norm_cdf(d1) - K * norm_cdf(d2)))


def asian_call(S, K, T, r, sigma, n_steps=252, n_sims=50_000, rng=None,
               average="arithmetic", control_variate=False):
    """Asian call: pays max(average price - K, 0).

    average         : "arithmetic" (the real product) or "geometric"
    control_variate : use the geometric Asian call (known exact price) to
                      cut the noise of the arithmetic one

    Returns (price, standard_error, (ci_low, ci_high)).
    """
    rng = np.random.default_rng() if rng is None else rng
    paths = simulate_paths(S, T, r, sigma, n_steps, n_sims, rng)
    monitored = paths[:, 1:]                    # the average excludes today's price
    df = discount_factor(r, T)

    geo_avg = np.exp(np.log(monitored).mean(axis=1))
    if average == "geometric":
        return summarize(df * call_payoff(geo_avg, K))
    if average != "arithmetic":
        raise ValueError(f"average must be 'arithmetic' or 'geometric', got {average!r}")

    samples = df * call_payoff(monitored.mean(axis=1), K)
    if control_variate:
        geo_samples = df * call_payoff(geo_avg, K)
        geo_exact = geometric_asian_call_exact(S, K, T, r, sigma, n_steps)
        samples = control_variate_adjust(samples, geo_samples, geo_exact)
    return summarize(samples)


def barrier_call(S, K, T, r, sigma, barrier, kind="up-and-out", n_steps=252,
                 n_sims=50_000, rng=None):
    """Barrier call with an upper barrier, checked at each time step.

    "up-and-out": pays like a call UNLESS the stock ever reaches the barrier
    "up-and-in" : pays like a call ONLY IF the stock ever reaches the barrier

    Returns (price, standard_error, (ci_low, ci_high)).
    """
    rng = np.random.default_rng() if rng is None else rng
    paths = simulate_paths(S, T, r, sigma, n_steps, n_sims, rng)
    hit = paths.max(axis=1) >= barrier
    vanilla = discount_factor(r, T) * call_payoff(paths[:, -1], K)

    if kind == "up-and-out":
        return summarize(np.where(hit, 0.0, vanilla))
    if kind == "up-and-in":
        return summarize(np.where(hit, vanilla, 0.0))
    raise ValueError(f"kind must be 'up-and-out' or 'up-and-in', got {kind!r}")


if __name__ == "__main__":
    p = EXAMPLE
    n_steps = 252      # one step per trading day
    seed = 42

    print("Asian call (average over 252 daily prices, 50,000 paths)")
    geo_exact = geometric_asian_call_exact(*p, n_steps)
    geo_mc, geo_se, _ = asian_call(*p, n_steps, rng=np.random.default_rng(seed),
                                   average="geometric")
    print(f"  geometric:  MC = {geo_mc:.4f} ± {geo_se:.4f}   exact = {geo_exact:.4f}")
    plain, se_plain, _ = asian_call(*p, n_steps, rng=np.random.default_rng(seed))
    cv, se_cv, _ = asian_call(*p, n_steps, rng=np.random.default_rng(seed),
                              control_variate=True)
    print(f"  arithmetic: plain MC        = {plain:.4f} ± {se_plain:.4f}")
    print(f"  arithmetic: control variate = {cv:.4f} ± {se_cv:.4f}   "
          f"(variance reduction {(se_plain / se_cv) ** 2:.0f}x)")
    print(f"  European call for comparison = {black_scholes_call(*p):.4f}")

    barrier = 130.0
    print(f"\nBarrier call (barrier = {barrier:.0f}, 50,000 paths)")
    out, se_out, _ = barrier_call(*p, barrier, "up-and-out", n_steps,
                                  rng=np.random.default_rng(seed))
    inn, se_in, _ = barrier_call(*p, barrier, "up-and-in", n_steps,
                                 rng=np.random.default_rng(seed))
    print(f"  up-and-out = {out:.4f} ± {se_out:.4f}")
    print(f"  up-and-in  = {inn:.4f} ± {se_in:.4f}")
    print(f"  out + in   = {out + inn:.4f}   European call (exact) = "
          f"{black_scholes_call(*p):.4f}")
