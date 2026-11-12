"""Greeks (price sensitivities) by Monte Carlo finite differences.

Delta = change in price per 1.00 change in the stock price.
Vega  = change in price per 1.00 change in volatility.

Method: price the option with an input nudged up, again with it nudged
down, and divide the difference by the size of the nudge (a "central
difference"). The key trick is COMMON RANDOM NUMBERS: both prices use the
same random shocks, so the simulation noise cancels in the difference.
"""
import numpy as np

from black_scholes import (EXAMPLE, OptionParams, call_payoff, put_payoff,
                           discount_factor, bs_call_delta, bs_put_delta, bs_vega)
from monte_carlo import terminal_prices, summarize


def _finite_difference(payoff_fn, up, down, step, n_sims, rng,
                       common_random_numbers):
    """Central difference (price(up) - price(down)) / (2 * step), path by path."""
    rng = np.random.default_rng() if rng is None else rng
    Z_up = rng.standard_normal(n_sims)
    Z_down = Z_up if common_random_numbers else rng.standard_normal(n_sims)

    def discounted_payoffs(p, Z):
        S_T = terminal_prices(p.S, p.T, p.r, p.sigma, Z)
        return discount_factor(p.r, p.T) * payoff_fn(S_T, p.K)

    samples = (discounted_payoffs(up, Z_up) - discounted_payoffs(down, Z_down)) / (2 * step)
    return summarize(samples)


def mc_delta(payoff_fn, S, K, T, r, sigma, n_sims=100_000, rng=None, h=None,
             common_random_numbers=True):
    """Delta by bumping the stock price up and down by h (default 1% of S).

    Returns (delta, standard_error, (ci_low, ci_high)).
    """
    h = 0.01 * S if h is None else h
    base = OptionParams(S, K, T, r, sigma)
    return _finite_difference(payoff_fn, base._replace(S=S + h), base._replace(S=S - h),
                              h, n_sims, rng, common_random_numbers)


def mc_vega(payoff_fn, S, K, T, r, sigma, n_sims=100_000, rng=None, h=0.01,
            common_random_numbers=True):
    """Vega by bumping volatility up and down by h (default 0.01 = 1 vol point).

    Returns (vega, standard_error, (ci_low, ci_high)).
    """
    base = OptionParams(S, K, T, r, sigma)
    return _finite_difference(payoff_fn, base._replace(sigma=sigma + h),
                              base._replace(sigma=sigma - h),
                              h, n_sims, rng, common_random_numbers)


if __name__ == "__main__":
    p = EXAMPLE
    rows = [
        ("call delta", mc_delta, call_payoff, bs_call_delta(*p)),
        ("put delta", mc_delta, put_payoff, bs_put_delta(*p)),
        ("call vega", mc_vega, call_payoff, bs_vega(*p)),
        ("put vega", mc_vega, put_payoff, bs_vega(*p)),
    ]
    print("Greeks: Monte Carlo (100,000 simulations) vs exact Black-Scholes")
    for name, fn, payoff, exact in rows:
        value, se, _ = fn(payoff, *p, rng=np.random.default_rng(42))
        _, se_indep, _ = fn(payoff, *p, rng=np.random.default_rng(42),
                            common_random_numbers=False)
        print(f"  {name:<10}  MC = {value:8.4f} ± {se:.4f}   exact = {exact:8.4f}   "
              f"(± {se_indep:.2f} without common random numbers)")
