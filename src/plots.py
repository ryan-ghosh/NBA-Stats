"""
plots.py
========
Final figures for the writeup. Loads posterior samples from results/.

Outputs (PNG, in results/figures/):
    01_posterior_densities.png   — μ, τ², τ_s², σ², β₁, β₂
    02_shrinkage.png             — raw mean error vs posterior θ̂_j (the wow factor)
    03_caterpillar.png           — team-level forest plot of θ_j with 95% CI
    04_rho_density.png           — posterior of intraclass correlation ρ
    05_beta_intervals.png        — β₁ and β₂ posterior densities + intervals
    06_prior_sensitivity.png     — primary vs tight on τ² and θ_j
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def load(npz: Path) -> dict:
    with np.load(npz, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def _flat(x):
    return x.reshape(-1) if x.ndim <= 2 else x.reshape(-1, x.shape[-1])


def fig_posterior_densities(s: dict, df_obs: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6))
    spec = [
        ("mu",     r"$\mu$ (league mean error)", axes[0, 0], None),
        ("tau2",   r"$\tau^2$ (between-team var)", axes[0, 1], None),
        ("tau_s2", r"$\tau_s^2$ (between-season var)", axes[0, 2], None),
        ("sigma2", r"$\sigma^2$ (within-team var)", axes[1, 0], None),
        ("beta1",  r"$\beta_1$ (rest differential effect)", axes[1, 1], None),
        ("beta2",  r"$\beta_2$ (home-favorite effect)", axes[1, 2], None),
    ]
    for k, label, ax, _ in spec:
        x = _flat(s[k])
        ax.hist(x, bins=60, color="#4c72b0", density=True,
                edgecolor="white", linewidth=0.4)
        m = np.mean(x); lo, hi = np.quantile(x, [0.025, 0.975])
        ax.axvline(m, color="black", lw=1)
        ax.axvline(lo, color="black", lw=0.6, ls="--")
        ax.axvline(hi, color="black", lw=0.6, ls="--")
        ax.set_title(f"{label}\nmean={m:.3f}, 95% CI=({lo:.3f}, {hi:.3f})", fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_shrinkage(s: dict, df: pd.DataFrame, out: Path) -> None:
    teams = list(s["teams"])
    th = s["theta"].reshape(-1, len(teams))     # all chain×iter samples
    th_mean = th.mean(axis=0)
    raw = (df.groupby("fav_team")["error"].mean()
             .reindex(teams).to_numpy())
    n = (df.groupby("fav_team").size()
           .reindex(teams).to_numpy())

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sizes = (n / n.max()) * 220 + 30
    ax.scatter(raw, th_mean, s=sizes, alpha=0.55,
               color="#4c72b0", edgecolor="white", linewidth=0.4)
    # 45-degree line
    lo = min(raw.min(), th_mean.min()) - 0.2
    hi = max(raw.max(), th_mean.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color="#888", lw=0.8, ls="--",
            label="45° (no shrinkage)")
    # Horizontal line at posterior mean of mu
    mu_post = float(np.mean(s["mu"]))
    ax.axhline(mu_post, color="#c44e52", lw=0.8, ls=":", label=f"posterior mean $\\mu$ = {mu_post:.2f}")
    # Label teams
    for i, t in enumerate(teams):
        # Short label (last word of team name)
        short = t.split()[-1]
        ax.annotate(short, (raw[i], th_mean[i]),
                    fontsize=7, alpha=0.85,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Raw mean spread error per team (points, no model)")
    ax.set_ylabel(r"Posterior mean $\hat{\theta}_j$ (points, after partial pooling)")
    ax.set_title("Shrinkage of team-level spread bias\n"
                 "Point size = games as favorite")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_caterpillar(s: dict, out: Path) -> None:
    teams = list(s["teams"])
    th = s["theta"].reshape(-1, len(teams))
    m = th.mean(axis=0)
    lo, hi = np.quantile(th, [0.025, 0.975], axis=0)
    order = np.argsort(m)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    y = np.arange(len(teams))
    color_above = "#4c72b0"; color_below = "#dd8452"
    for k, i in enumerate(order):
        col = color_above if m[i] >= 0 else color_below
        ax.plot([lo[i], hi[i]], [k, k], color=col, lw=1.5, alpha=0.85)
        ax.plot(m[i], k, "o", color=col, ms=5)
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.axvline(float(np.mean(s["mu"])), color="#c44e52", lw=0.7, ls=":",
               label=f"posterior mean $\\mu$")
    ax.set_yticks(y); ax.set_yticklabels([teams[i] for i in order], fontsize=8)
    ax.set_xlabel(r"Posterior $\theta_j$ (points), 95% credible interval")
    ax.set_title("Team-level systematic spread bias\n(after rest, home, and season adjustments)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_rho(s: dict, out: Path) -> None:
    rho = (s["tau2"] / (s["tau2"] + s["tau_s2"] + s["sigma2"])).reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rho * 100, bins=60, color="#4c72b0", density=True,
            edgecolor="white", linewidth=0.4)
    m = rho.mean(); lo, hi = np.quantile(rho, [0.025, 0.975])
    ax.axvline(m * 100, color="black", lw=1)
    ax.axvline(lo * 100, color="black", lw=0.6, ls="--")
    ax.axvline(hi * 100, color="black", lw=0.6, ls="--")
    ax.set_xlabel(r"Intraclass correlation $\rho$ (%)")
    ax.set_ylabel("Posterior density")
    ax.set_title(rf"Posterior of $\rho$ — share of variance attributable to team heterogeneity"
                 f"\nposterior mean = {m*100:.2f}%, 95% CI = ({lo*100:.2f}%, {hi*100:.2f}%)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_betas(s: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k, lab in zip(axes, ["beta1", "beta2"],
                          [r"$\beta_1$ — rest differential (per day)",
                           r"$\beta_2$ — home-favorite indicator"]):
        x = _flat(s[k])
        ax.hist(x, bins=60, color="#4c72b0", density=True,
                edgecolor="white", linewidth=0.4)
        m = x.mean(); lo, hi = np.quantile(x, [0.025, 0.975])
        ax.axvline(0, color="#c44e52", lw=0.8, ls="--")
        ax.axvline(m, color="black", lw=1)
        ax.axvline(lo, color="black", lw=0.6, ls="--")
        ax.axvline(hi, color="black", lw=0.6, ls="--")
        ax.set_title(f"{lab}\nposterior mean = {m:+.3f}, 95% CI = ({lo:+.3f}, {hi:+.3f})",
                     fontsize=10)
        ax.set_xlabel("Effect on spread error (points)")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_prior_sensitivity(prim: dict, tight: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # tau^2
    ax = axes[0]
    for s, lab, col in [(prim, "primary", "#4c72b0"),
                        (tight, "tight (η₀=50, τ₀²=0.1)", "#c44e52")]:
        x = _flat(s["tau2"])
        ax.hist(x, bins=80, density=True, alpha=0.55,
                color=col, label=f"{lab}: mean={x.mean():.3f}",
                edgecolor="white", linewidth=0.3)
    ax.set_xlim(0, 1.5)
    ax.set_xlabel(r"$\tau^2$ (points$^2$)")
    ax.set_ylabel("Posterior density")
    ax.set_title(r"Prior sensitivity for $\tau^2$")
    ax.legend(frameon=False, fontsize=9)

    # team thetas: per-team posterior mean under each prior
    ax = axes[1]
    teams = list(prim["teams"])
    th_p = prim["theta"].reshape(-1, len(teams)).mean(axis=0)
    th_t = tight["theta"].reshape(-1, len(teams)).mean(axis=0)
    order = np.argsort(th_p)
    ax.plot(th_p[order], np.arange(len(teams)), "o", color="#4c72b0",
            ms=5, label="primary")
    ax.plot(th_t[order], np.arange(len(teams)), "s", color="#c44e52",
            ms=4, label="tight")
    for i, j in enumerate(order):
        ax.plot([th_t[j], th_p[j]], [i, i], color="#888", lw=0.5, alpha=0.6)
    ax.set_yticks(np.arange(len(teams)))
    ax.set_yticklabels([teams[j] for j in order], fontsize=7)
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel(r"Posterior mean $\hat{\theta}_j$ (points)")
    ax.set_title("Effect of prior on team-level shrinkage")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    here = Path(__file__).resolve().parents[1]
    fig_dir = here / "results" / "figures"; fig_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(here / "data" / "nba_spreads.csv")
    prim = load(here / "results" / "posterior_primary.npz")
    tight = load(here / "results" / "posterior_tight.npz")

    fig_posterior_densities(prim, df, fig_dir / "01_posterior_densities.png")
    fig_shrinkage(prim, df, fig_dir / "02_shrinkage.png")
    fig_caterpillar(prim, fig_dir / "03_caterpillar.png")
    fig_rho(prim, fig_dir / "04_rho_density.png")
    fig_betas(prim, fig_dir / "05_beta_intervals.png")
    fig_prior_sensitivity(prim, tight, fig_dir / "06_prior_sensitivity.png")
    print("Figures saved to", fig_dir)


if __name__ == "__main__":
    main()
