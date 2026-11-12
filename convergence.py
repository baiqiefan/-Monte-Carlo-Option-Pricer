"""Convergence plot: does the Monte Carlo price settle on the Black-Scholes price?

Runs the Monte Carlo call pricer at increasing simulation counts and plots
the estimate with its 95% confidence band against the exact answer, plus
the standard error of each variance-reduction method.
Saves the figure as convergence.png.
"""
import matplotlib.pyplot as plt
import numpy as np

from black_scholes import EXAMPLE, black_scholes_call
from monte_carlo import monte_carlo_call

p = EXAMPLE
rng = np.random.default_rng(0)
exact = black_scholes_call(*p)

# Simulation counts spaced evenly on a log scale: 100 ... 1,000,000
sim_counts = np.logspace(2, 6, num=25).astype(int)

prices, ci_lows, ci_highs = [], [], []
std_errors = {"plain": [], "antithetic": [], "control": []}
for n in sim_counts:
    price, se, (lo, hi) = monte_carlo_call(*p, n_sims=n, rng=rng)
    prices.append(price)
    ci_lows.append(lo)
    ci_highs.append(hi)
    std_errors["plain"].append(se)
    for method in ("antithetic", "control"):
        std_errors[method].append(monte_carlo_call(*p, n_sims=n, rng=rng, method=method)[1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: price estimate converging on the exact value
ax1.fill_between(sim_counts, ci_lows, ci_highs, alpha=0.25, label="95% confidence band")
ax1.plot(sim_counts, prices, "o-", markersize=4, label="Monte Carlo estimate")
ax1.axhline(exact, color="black", linestyle="--", label=f"Black-Scholes = {exact:.4f}")
ax1.set_xscale("log")
ax1.set_xlabel("Number of simulations")
ax1.set_ylabel("Call price")
ax1.set_title("Monte Carlo price converges to Black-Scholes")
ax1.legend()

# Right: standard error shrinks like 1/sqrt(N) -> a straight line on log-log axes
ax2.loglog(sim_counts, std_errors["plain"], "o-", markersize=4, label="Plain Monte Carlo")
ax2.loglog(sim_counts, std_errors["antithetic"], "s-", markersize=4, label="Antithetic variates")
ax2.loglog(sim_counts, std_errors["control"], "^-", markersize=4, label="Control variate")
reference = std_errors["plain"][0] * np.sqrt(sim_counts[0] / sim_counts)
ax2.loglog(sim_counts, reference, "--", color="black", label=r"$1/\sqrt{N}$ reference")
ax2.set_xlabel("Number of simulations")
ax2.set_ylabel("Standard error")
ax2.set_title(r"Error shrinks as $1/\sqrt{N}$")
ax2.legend()

fig.tight_layout()
fig.savefig("convergence.png", dpi=150)
print("Saved convergence.png")
print(f"N = {sim_counts[-1]:,}: MC = {prices[-1]:.4f} ± {std_errors['plain'][-1]:.4f}, "
      f"BS = {exact:.4f}")
plain = np.array(std_errors["plain"])
for method in ("antithetic", "control"):
    ratio = np.mean((plain / np.array(std_errors[method])) ** 2)
    print(f"{method} variance reduction (avg over all N): {ratio:.2f}x")
