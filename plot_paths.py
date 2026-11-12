"""Plot a handful of simulated price paths with the strike and barrier.

Paths that touch the barrier (and so knock out an up-and-out call) are
drawn in grey. Saves the figure as paths.png.
"""
import matplotlib.pyplot as plt
import numpy as np

from black_scholes import EXAMPLE
from exotics import simulate_paths

p = EXAMPLE
barrier = 130.0
n_steps = 252

paths = simulate_paths(p.S, p.T, p.r, p.sigma, n_steps, n_sims=40,
                       rng=np.random.default_rng(7))
times = np.linspace(0, p.T, n_steps + 1)
hit = paths.max(axis=1) >= barrier

fig, ax = plt.subplots(figsize=(9, 4.5))
for path, knocked_out in zip(paths, hit):
    ax.plot(times, path, color="grey" if knocked_out else "tab:blue",
            alpha=0.45 if knocked_out else 0.7, linewidth=1)
ax.axhline(barrier, color="tab:red", linestyle="--", label=f"Barrier = {barrier:.0f}")
ax.axhline(p.K, color="black", linestyle=":", label=f"Strike = {p.K:.0f}")
ax.plot([], [], color="tab:blue", label="Path stays below the barrier")
ax.plot([], [], color="grey", label="Path touches the barrier (knocked out)")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Stock price")
ax.set_title("40 simulated stock price paths (Geometric Brownian Motion)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("paths.png", dpi=150)
print(f"Saved paths.png ({hit.sum()} of {len(hit)} paths touched the barrier)")
