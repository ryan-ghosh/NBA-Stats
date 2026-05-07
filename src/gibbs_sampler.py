"""
gibbs_sampler.py
================
Pure-NumPy Gibbs sampler for the two-level multilevel model

    X_i = theta_{j[i]} + beta_1 R_i + beta_2 H_i + gamma_{s[i]} + eps_i,
    eps_i ~ N(0, sigma^2),
    theta_j ~ N(mu, tau^2),  mu ~ N(mu0, gamma0^2),
    gamma_s ~ N(0, tau_s^2),
    beta ~ N(0, sigma_beta^2 I),
    sigma^2  ~ IG(nu0/2, nu0*sigma0^2/2),
    tau^2    ~ IG(eta0/2, eta0*tau0^2/2),
    tau_s^2  ~ IG(eta_s0/2, eta_s0*tau_s0^2/2).

Full conditionals are derived in writeup/full_conditionals.md.

Run:
    python src/gibbs_sampler.py --prior primary
    python src/gibbs_sampler.py --prior tight
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd


# ----- Hyperparameter sets ---------------------------------------------------

@dataclass
class Hyper:
    name: str
    mu0: float = 0.0
    gamma0_sq: float = 4.0          # prior var for mu
    sigma_beta_sq: float = 1.0      # prior var for beta_1, beta_2
    nu0: float = 1.0
    sigma0_sq: float = 144.0        # weak prior for sigma^2 anchored at empirical
    eta0: float = 1.0
    tau0_sq: float = 1.0            # team-level
    eta_s0: float = 1.0
    tau_s0_sq: float = 1.0          # season-level


PRIMARY = Hyper(name="primary")
# Tight prior: strongly shrinks team-level variance toward 0.1 — pushes the
# posterior toward complete-pooling. Pick this as the contrast prior since the
# substantive question is whether the data overcomes a strong "all teams
# equal" prior.
TIGHT = Hyper(name="tight", eta0=50.0, tau0_sq=0.1)


# ----- Sampler ---------------------------------------------------------------

def run_chain(X, team_idx, season_idx, R, H, n_teams, n_seasons,
              hyper: Hyper, n_iter: int, burn: int, seed: int,
              dispersed: bool = True) -> dict:
    """Run one Gibbs chain. Returns dict of stored samples (post burn-in)."""
    rng = np.random.default_rng(seed)
    N = len(X)
    J, S = n_teams, n_seasons

    # Pre-allocate storage
    keep = n_iter - burn
    out = {
        "mu":      np.empty(keep),
        "tau2":    np.empty(keep),
        "tau_s2":  np.empty(keep),
        "sigma2":  np.empty(keep),
        "beta1":   np.empty(keep),
        "beta2":   np.empty(keep),
        "theta":   np.empty((keep, J)),
        "gamma":   np.empty((keep, S)),
    }

    # Pre-compute per-group counts and sufficient stats helpers
    # (means/sums computed each iteration via np.add.at)
    n_j = np.bincount(team_idx, minlength=J)
    n_s = np.bincount(season_idx, minlength=S)

    # Design matrix Z = [R, H], used in the joint beta update
    Z = np.column_stack([R, H])
    ZtZ = Z.T @ Z   # 2x2

    # Initialization (dispersed for multi-chain R-hat)
    if dispersed:
        mu = rng.normal(0, 5)
        tau2 = float(rng.uniform(0.1, 5.0))
        tau_s2 = float(rng.uniform(0.1, 5.0))
        sigma2 = float(rng.uniform(80, 200))
        beta1 = rng.normal(0, 1)
        beta2 = rng.normal(0, 2)
        theta = rng.normal(mu, np.sqrt(tau2), size=J)
        gamma = rng.normal(0, np.sqrt(tau_s2), size=S)
    else:
        mu, tau2, tau_s2, sigma2 = 0.0, 1.0, 1.0, 144.0
        beta1, beta2 = 0.0, 0.0
        theta = np.zeros(J)
        gamma = np.zeros(S)

    h = hyper
    inv_sigma_beta_sq = 1.0 / h.sigma_beta_sq
    eye2 = np.eye(2)

    for t in range(n_iter):
        # ----- Update mu | rest (Eq. A.2) ------------------------------------
        v_mu = 1.0 / (1.0 / h.gamma0_sq + J / tau2)
        m_mu = v_mu * (h.mu0 / h.gamma0_sq + theta.sum() / tau2)
        mu = float(rng.normal(m_mu, np.sqrt(v_mu)))

        # ----- Update tau^2 | rest (Eq. A.3) ---------------------------------
        a_tau = (h.eta0 + J) / 2.0
        b_tau = (h.eta0 * h.tau0_sq + np.sum((theta - mu) ** 2)) / 2.0
        tau2 = float(b_tau / rng.gamma(a_tau))   # Inv-Gamma(a,b) sample

        # ----- Update theta_j | rest (Eq. A.1) -------------------------------
        # partial residual r_i = X_i - beta1*R_i - beta2*H_i - gamma[s[i]]
        r = X - beta1 * R - beta2 * H - gamma[season_idx]
        sum_r_per_team = np.zeros(J)
        np.add.at(sum_r_per_team, team_idx, r)
        v_th = 1.0 / (1.0 / tau2 + n_j / sigma2)
        m_th = v_th * (mu / tau2 + sum_r_per_team / sigma2)
        theta = rng.normal(m_th, np.sqrt(v_th))

        # ----- Update beta = (beta1, beta2) jointly (Eq. A.7) ----------------
        y_resid = X - theta[team_idx] - gamma[season_idx]
        Zty = Z.T @ y_resid
        V_beta_inv = ZtZ / sigma2 + inv_sigma_beta_sq * eye2
        V_beta = np.linalg.inv(V_beta_inv)
        m_beta = V_beta @ (Zty / sigma2)
        # MV normal sample via Cholesky
        L = np.linalg.cholesky(V_beta)
        z = rng.standard_normal(2)
        beta = m_beta + L @ z
        beta1, beta2 = float(beta[0]), float(beta[1])

        # ----- Update gamma_s | rest (Eq. A.5) -------------------------------
        q = X - theta[team_idx] - beta1 * R - beta2 * H
        sum_q_per_season = np.zeros(S)
        np.add.at(sum_q_per_season, season_idx, q)
        v_g = 1.0 / (1.0 / tau_s2 + n_s / sigma2)
        m_g = v_g * (sum_q_per_season / sigma2)
        gamma = rng.normal(m_g, np.sqrt(v_g))

        # ----- Update tau_s^2 | rest (Eq. A.6) -------------------------------
        a_ts = (h.eta_s0 + S) / 2.0
        b_ts = (h.eta_s0 * h.tau_s0_sq + np.sum(gamma ** 2)) / 2.0
        tau_s2 = float(b_ts / rng.gamma(a_ts))

        # ----- Update sigma^2 | rest (Eq. A.4) -------------------------------
        resid = X - theta[team_idx] - beta1 * R - beta2 * H - gamma[season_idx]
        a_sig = (h.nu0 + N) / 2.0
        b_sig = (h.nu0 * h.sigma0_sq + np.sum(resid ** 2)) / 2.0
        sigma2 = float(b_sig / rng.gamma(a_sig))

        if t >= burn:
            k = t - burn
            out["mu"][k] = mu
            out["tau2"][k] = tau2
            out["tau_s2"][k] = tau_s2
            out["sigma2"][k] = sigma2
            out["beta1"][k] = beta1
            out["beta2"][k] = beta2
            out["theta"][k] = theta
            out["gamma"][k] = gamma

    return out


def run(prior_name: str, n_chains: int = 4, n_iter: int = 12000,
        burn: int = 2000, seed_base: int = 20260507) -> Path:
    here = Path(__file__).resolve().parents[1]
    df = pd.read_csv(here / "data" / "nba_spreads.csv")

    # Index encodings
    teams = sorted(df["fav_team"].unique())
    seasons = sorted(df["season"].unique())
    team_to_id = {t: i for i, t in enumerate(teams)}
    season_to_id = {s: i for i, s in enumerate(seasons)}

    X = df["error"].to_numpy(dtype=np.float64)
    team_idx = df["fav_team"].map(team_to_id).to_numpy(dtype=np.int64)
    season_idx = df["season"].map(season_to_id).to_numpy(dtype=np.int64)
    R = df["rest_diff"].to_numpy(dtype=np.float64)   # already centered
    H = df["home"].to_numpy(dtype=np.float64)        # 0/1

    hyper = TIGHT if prior_name == "tight" else PRIMARY

    print(f"\n=== Running {n_chains} chains × {n_iter} iters "
          f"(burn={burn}) under prior '{prior_name}' ===")
    print(f"N={len(X)}, J={len(teams)}, S={len(seasons)}")

    chains = []
    t0 = time.time()
    for c in range(n_chains):
        seed = seed_base + 1000 * c
        out = run_chain(X, team_idx, season_idx, R, H,
                        len(teams), len(seasons),
                        hyper, n_iter, burn, seed=seed,
                        dispersed=True)
        elapsed = time.time() - t0
        print(f"  chain {c}: seed={seed}, "
              f"posterior mean tau2={out['tau2'].mean():.3f}, "
              f"sigma2={out['sigma2'].mean():.2f}, "
              f"mu={out['mu'].mean():+.3f}  "
              f"({elapsed:5.1f}s elapsed)")
        chains.append(out)

    # Stack chains: shape (n_chains, n_keep, ...)
    stacked = {k: np.stack([c[k] for c in chains], axis=0)
               for k in chains[0]}
    stacked["teams"] = np.array(teams)
    stacked["seasons"] = np.array(seasons)
    stacked["prior_name"] = np.array(prior_name)

    out_path = here / "results" / f"posterior_{prior_name}.npz"
    np.savez_compressed(out_path, **stacked)
    print(f"Saved {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prior", choices=["primary", "tight"], default="primary")
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--iters", type=int, default=12000)
    p.add_argument("--burn", type=int, default=2000)
    args = p.parse_args()
    run(args.prior, n_chains=args.chains, n_iter=args.iters, burn=args.burn)
