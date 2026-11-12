"""Closed-form Black-Scholes prices and Greeks for European options.

These exact formulas are the "answer key" the Monte Carlo pricer is
checked against.
"""
import math
from typing import NamedTuple

import numpy as np


class OptionParams(NamedTuple):
    """The five inputs every pricing function needs.

    A NamedTuple can be unpacked with *, so black_scholes_call(*EXAMPLE) works.
    """
    S: float      # stock price today
    K: float      # strike price
    T: float      # time until expiry, in YEARS
    r: float      # risk-free interest rate (per year)
    sigma: float  # volatility (per year)


# The example option used throughout the project
EXAMPLE = OptionParams(S=100.0, K=105.0, T=1.0, r=0.05, sigma=0.20)


def call_payoff(S_T, K):
    """Money a European call pays at expiry, given final stock price(s) S_T.

    S_T can be a single number or a numpy array of many prices.
    """
    return np.maximum(S_T - K, 0)


def put_payoff(S_T, K):
    """Money a European put pays at expiry: max(K - S_T, 0)."""
    return np.maximum(K - S_T, 0)


def discount_factor(r, T):
    """Number you multiply a future amount by to get its value today."""
    return np.exp(-r * T)


def norm_cdf(x):
    """N(x): probability a standard bell-curve number is less than x.

    Built from math.erf because scipy is blocked on this machine.
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x):
    """Height of the standard bell curve at x."""
    return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)


def d1_d2(S, K, T, r, sigma):
    """Return (d1, d2): how many 'standard wobbles' the stock is from the strike."""
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (float(d1), float(d2))


def black_scholes_call(S, K, T, r, sigma):
    """Price a European call option using the Black-Scholes formula."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return float(S * norm_cdf(d1) - K * discount_factor(r, T) * norm_cdf(d2))


def black_scholes_put(S, K, T, r, sigma):
    """Price a European put option using the Black-Scholes formula."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return float(K * discount_factor(r, T) * norm_cdf(-d2) - S * norm_cdf(-d1))


def bs_call_delta(S, K, T, r, sigma):
    """Delta of a call: how much the price moves when the stock moves by 1."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm_cdf(d1)


def bs_put_delta(S, K, T, r, sigma):
    """Delta of a put (always negative: puts gain when the stock falls)."""
    return bs_call_delta(S, K, T, r, sigma) - 1


def bs_vega(S, K, T, r, sigma):
    """Vega: how much the price moves per 1.00 change in volatility.

    The same for calls and puts.
    """
    d1, _ = d1_d2(S, K, T, r, sigma)
    return float(S * norm_pdf(d1) * np.sqrt(T))


if __name__ == "__main__":
    p = EXAMPLE
    call = black_scholes_call(*p)
    put = black_scholes_put(*p)
    print(f"call  = {call:.4f}")   # about 8.02
    print(f"put   = {put:.4f}")    # about 7.90
    # put-call parity check: should be close to 0
    parity_gap = call - put - (p.S - p.K * discount_factor(p.r, p.T))
    print(f"parity gap = {parity_gap:.2e}")
    print(f"call delta = {bs_call_delta(*p):.4f}")
    print(f"put delta  = {bs_put_delta(*p):.4f}")
    print(f"vega       = {bs_vega(*p):.4f}")
