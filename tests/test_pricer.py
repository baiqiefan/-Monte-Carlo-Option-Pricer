"""Tests for the option pricer. Run with:  python -m pytest

Every Monte Carlo test uses a fixed seed, so results are repeatable.
"Within 4 standard errors" is the tolerance for comparing a simulated
value with an exact one.
"""
import numpy as np
import pytest

from black_scholes import (EXAMPLE, call_payoff, put_payoff, discount_factor,
                           black_scholes_call, black_scholes_put,
                           bs_call_delta, bs_put_delta, bs_vega)
from exotics import (simulate_paths, geometric_asian_call_exact, asian_call,
                     barrier_call)
from greeks import mc_delta, mc_vega
from monte_carlo import METHODS, monte_carlo_call, monte_carlo_put, monte_carlo_price

P = EXAMPLE
SEED = 12345


def rng():
    return np.random.default_rng(SEED)


# ---------- Black-Scholes ----------

def test_black_scholes_known_values():
    assert black_scholes_call(*P) == pytest.approx(8.0214, abs=1e-4)
    assert black_scholes_put(*P) == pytest.approx(7.9004, abs=1e-4)


def test_put_call_parity():
    gap = (black_scholes_call(*P) - black_scholes_put(*P)
           - (P.S - P.K * discount_factor(P.r, P.T)))
    assert gap == pytest.approx(0, abs=1e-12)


def test_almost_zero_volatility_gives_intrinsic_value():
    # With no randomness the stock just grows at r, so the call is worth
    # max(S - K * exp(-rT), 0).
    price = black_scholes_call(P.S, P.K, P.T, P.r, sigma=1e-6)
    assert price == pytest.approx(P.S - P.K * discount_factor(P.r, P.T), abs=1e-6)


def test_payoffs():
    prices = np.array([90.0, 105.0, 120.0])
    assert call_payoff(prices, 105.0).tolist() == [0.0, 0.0, 15.0]
    assert put_payoff(prices, 105.0).tolist() == [15.0, 0.0, 0.0]


# ---------- Monte Carlo ----------

@pytest.mark.parametrize("method", METHODS)
def test_monte_carlo_matches_black_scholes(method):
    call, call_se, _ = monte_carlo_call(*P, rng=rng(), method=method)
    put, put_se, _ = monte_carlo_put(*P, rng=rng(), method=method)
    assert abs(call - black_scholes_call(*P)) < 4 * call_se
    assert abs(put - black_scholes_put(*P)) < 4 * put_se


def test_variance_reduction_actually_reduces_error():
    se = {m: monte_carlo_call(*P, rng=rng(), method=m)[1] for m in METHODS}
    assert se["control"] < se["antithetic"] < se["plain"]


def test_same_seed_gives_same_price():
    assert monte_carlo_call(*P, rng=rng()) == monte_carlo_call(*P, rng=rng())


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        monte_carlo_price(call_payoff, *P, method="magic")


# ---------- Greeks ----------

def test_delta_matches_black_scholes():
    call_delta, se, _ = mc_delta(call_payoff, *P, rng=rng())
    assert abs(call_delta - bs_call_delta(*P)) < 4 * se
    put_delta, se, _ = mc_delta(put_payoff, *P, rng=rng())
    assert abs(put_delta - bs_put_delta(*P)) < 4 * se


def test_vega_matches_black_scholes():
    vega, se, _ = mc_vega(call_payoff, *P, rng=rng())
    assert abs(vega - bs_vega(*P)) < 4 * se


def test_common_random_numbers_reduce_greek_noise():
    _, se_crn, _ = mc_delta(call_payoff, *P, rng=rng())
    _, se_indep, _ = mc_delta(call_payoff, *P, rng=rng(), common_random_numbers=False)
    assert se_crn < se_indep / 5


# ---------- Exotics ----------

N_STEPS = 50
N_PATHS = 20_000


def test_paths_start_at_S_and_have_right_shape():
    paths = simulate_paths(P.S, P.T, P.r, P.sigma, N_STEPS, 100, rng())
    assert paths.shape == (100, N_STEPS + 1)
    assert np.all(paths[:, 0] == P.S)
    assert np.all(paths > 0)


def test_geometric_asian_matches_exact_formula():
    price, se, _ = asian_call(*P, N_STEPS, N_PATHS, rng(), average="geometric")
    assert abs(price - geometric_asian_call_exact(*P, N_STEPS)) < 4 * se


def test_asian_ordering():
    # geometric average <= arithmetic average on every path, and averaging
    # smooths the price so an Asian call is cheaper than a European call.
    geo, _, _ = asian_call(*P, N_STEPS, N_PATHS, rng(), average="geometric")
    arith, _, _ = asian_call(*P, N_STEPS, N_PATHS, rng())
    assert geo < arith < black_scholes_call(*P)


def test_asian_control_variate_agrees_and_is_less_noisy():
    plain, se_plain, _ = asian_call(*P, N_STEPS, N_PATHS, rng())
    cv, se_cv, _ = asian_call(*P, N_STEPS, N_PATHS, rng(), control_variate=True)
    assert abs(plain - cv) < 4 * se_plain
    assert se_cv < se_plain / 5


def test_barrier_in_out_parity():
    # On every path exactly one of (knock-out, knock-in) pays the call payoff,
    # so their prices must add up to the European call on the same paths.
    out, _, _ = barrier_call(*P, 130.0, "up-and-out", N_STEPS, N_PATHS, rng())
    inn, _, _ = barrier_call(*P, 130.0, "up-and-in", N_STEPS, N_PATHS, rng())
    paths = simulate_paths(P.S, P.T, P.r, P.sigma, N_STEPS, N_PATHS, rng())
    vanilla = (discount_factor(P.r, P.T) * call_payoff(paths[:, -1], P.K)).mean()
    assert out + inn == pytest.approx(vanilla, abs=1e-9)


def test_unreachable_barrier_is_a_european_call():
    out, se, _ = barrier_call(*P, 1e9, "up-and-out", N_STEPS, N_PATHS, rng())
    assert abs(out - black_scholes_call(*P)) < 4 * se
