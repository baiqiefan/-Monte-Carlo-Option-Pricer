"""Monte Carlo pricing of European options, with variance reduction.

Three methods, all returning (price, standard_error, (ci_low, ci_high)):
  "plain"       average the discounted payoff over random futures
  "antithetic"  pair every random shock Z with its mirror image -Z
  "control"     correct each payoff using the stock price, whose mean is known
"""
import numpy as np

from black_scholes import (EXAMPLE, call_payoff, put_payoff, discount_factor,
                           black_scholes_call, black_scholes_put)

METHODS = ("plain", "antithetic", "control")


def terminal_prices(S, T, r, sigma, Z):
    """Turn standard-normal shocks Z into final stock prices (GBM).

    Formula: S_T = S * exp( (r - sigma**2/2) * T + sigma * sqrt(T) * Z )
    Z can be one number or an array; the result has the same shape.
    """
    return S * np.exp((r - sigma**2/2) * T + sigma * np.sqrt(T) * Z)


def simulate_many_prices(S, T, r, sigma, n_sims, rng):
    """Simulate n_sims random final stock prices at once (vectorized)."""
    return terminal_prices(S, T, r, sigma, rng.standard_normal(n_sims))


def summarize(samples):
    """Turn independent samples into (mean, standard_error, 95% CI)."""
    price = float(samples.mean())
    std_error = float(samples.std(ddof=1) / np.sqrt(len(samples)))
    ci = (price - 1.96 * std_error, price + 1.96 * std_error)
    return price, std_error, ci


def control_variate_adjust(Y, X, X_mean):
    """Reduce the noise in Y using a correlated quantity X whose true mean is known.

    If X came out above its known mean on a path, Y probably came out too
    high as well, so we pull Y back down by beta * (X - X_mean). The
    correction averages to zero, so the price is unchanged but less noisy.
    beta = cov(Y, X) / var(X) is the choice that removes the most noise.
    """
    cov = np.cov(Y, X)
    beta = cov[0, 1] / cov[1, 1]
    return Y - beta * (X - X_mean)


def monte_carlo_price(payoff_fn, S, K, T, r, sigma, n_sims=100_000, rng=None,
                      method="plain"):
    """Price a European option by simulation.

    payoff_fn : function (S_T, K) -> payoff, e.g. call_payoff
    rng       : a numpy Generator (np.random.default_rng(seed)) for repeatability
    method    : "plain", "antithetic" or "control"

    Returns (price, standard_error, (ci_low, ci_high)).
    """
    rng = np.random.default_rng() if rng is None else rng
    df = discount_factor(r, T)

    if method == "plain":
        S_T = simulate_many_prices(S, T, r, sigma, n_sims, rng)
        samples = df * payoff_fn(S_T, K)

    elif method == "antithetic":
        # Half as many shocks, each used twice (+Z and -Z), so the total
        # number of simulated prices is still n_sims.
        Z = rng.standard_normal(n_sims // 2)
        up = payoff_fn(terminal_prices(S, T, r, sigma, Z), K)
        down = payoff_fn(terminal_prices(S, T, r, sigma, -Z), K)
        # The two halves of a pair are linked, so the independent samples
        # are the pair averages, not the individual payoffs.
        samples = df * 0.5 * (up + down)

    elif method == "control":
        S_T = simulate_many_prices(S, T, r, sigma, n_sims, rng)
        Y = df * payoff_fn(S_T, K)
        X = df * S_T   # discounted stock price: its true mean is exactly S
        samples = control_variate_adjust(Y, X, S)

    else:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    return summarize(samples)


def monte_carlo_call(S, K, T, r, sigma, n_sims=100_000, rng=None, method="plain"):
    """Monte Carlo price of a European call."""
    return monte_carlo_price(call_payoff, S, K, T, r, sigma, n_sims, rng, method)


def monte_carlo_put(S, K, T, r, sigma, n_sims=100_000, rng=None, method="plain"):
    """Monte Carlo price of a European put."""
    return monte_carlo_price(put_payoff, S, K, T, r, sigma, n_sims, rng, method)


if __name__ == "__main__":
    p = EXAMPLE
    for name, mc_fn, bs_fn in [("call", monte_carlo_call, black_scholes_call),
                               ("put", monte_carlo_put, black_scholes_put)]:
        exact = bs_fn(*p)
        print(f"\n{name}: Black-Scholes = {exact:.4f}   (100,000 simulations)")
        se_plain = None
        for method in METHODS:
            rng = np.random.default_rng(42)
            price, se, (lo, hi) = mc_fn(*p, rng=rng, method=method)
            se_plain = se if se_plain is None else se_plain
            inside = "yes" if lo <= exact <= hi else "NO"
            print(f"  {method:<10}  {price:.4f} ± {se:.4f}   "
                  f"95% CI [{lo:.4f}, {hi:.4f}]   BS inside CI? {inside}   "
                  f"variance reduction {(se_plain / se) ** 2:.1f}x")
