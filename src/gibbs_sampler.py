from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd



@dataclass
class Hyper:
    name: str
    mu0: float = 0.0
    gamma0_sq: float = 4.0
    sigma_beta_sq: float = 1.0
    nu0: float = 1.0
    sigma0_sq: float = 144.0
    eta0: float = 1.0
    tau0_sq: float = 1.0
    eta_s0: float = 1.0
    tau_s0_sq: float = 1.0


PRIMARY = Hyper(name="primary")
TIGHT = Hyper(name="tight", eta0=50.0, tau0_sq=0.1)



def run_chain(X, team_idx, season_idx, R, H, n_teams, n_seasons,
              hyper: Hyper, n_iter: int, burn: int, seed: int,
              dispersed: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    N = len(X)
    J, S = n_teams, n_seasons


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



    n_j = np.bincount(team_idx, minlength=J)
    n_s = np.bincount(season_idx, minlength=S)


    Z = np.column_stack([R, H])
    ZtZ = Z.T @ Z

    #initialization
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

        #mu
        v_mu = 1.0 / (1.0 / h.gamma0_sq + J / tau2)
        m_mu = v_mu * (h.mu0 / h.gamma0_sq + theta.sum() / tau2)
        mu = float(rng.normal(m_mu, np.sqrt(v_mu)))


        #tau
        a_tau = (h.eta0 + J) / 2.0
        b_tau = (h.eta0 * h.tau0_sq + np.sum((theta - mu) ** 2)) / 2.0
        tau2 = float(b_tau / rng.gamma(a_tau))


        #theta
        r = X - beta1 * R - beta2 * H - gamma[season_idx]
        sum_r_per_team = np.zeros(J)
        np.add.at(sum_r_per_team, team_idx, r)
        v_th = 1.0 / (1.0 / tau2 + n_j / sigma2)
        m_th = v_th * (mu / tau2 + sum_r_per_team / sigma2)
        theta = rng.normal(m_th, np.sqrt(v_th))


        #beta
        y_resid = X - theta[team_idx] - gamma[season_idx]
        Zty = Z.T @ y_resid
        V_beta_inv = ZtZ / sigma2 + inv_sigma_beta_sq * eye2
        V_beta = np.linalg.inv(V_beta_inv)
        m_beta = V_beta @ (Zty / sigma2)
        L = np.linalg.cholesky(V_beta)
        z = rng.standard_normal(2)
        beta = m_beta + L @ z
        beta1, beta2 = float(beta[0]), float(beta[1])


        #gamma
        q = X - theta[team_idx] - beta1 * R - beta2 * H
        sum_q_per_season = np.zeros(S)
        np.add.at(sum_q_per_season, season_idx, q)
        v_g = 1.0 / (1.0 / tau_s2 + n_s / sigma2)
        m_g = v_g * (sum_q_per_season / sigma2)
        gamma = rng.normal(m_g, np.sqrt(v_g))


        #tau_s^2
        a_ts = (h.eta_s0 + S) / 2.0
        b_ts = (h.eta_s0 * h.tau_s0_sq + np.sum(gamma ** 2)) / 2.0
        tau_s2 = float(b_ts / rng.gamma(a_ts))


        #sigma^2
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


    teams = sorted(df["fav_team"].unique())
    seasons = sorted(df["season"].unique())
    team_to_id = {t: i for i, t in enumerate(teams)}
    season_to_id = {s: i for i, s in enumerate(seasons)}


    X = df["error"].to_numpy(dtype=np.float64)
    team_idx = df["fav_team"].map(team_to_id).to_numpy(dtype=np.int64)
    season_idx = df["season"].map(season_to_id).to_numpy(dtype=np.int64)
    R = df["rest_diff"].to_numpy(dtype=np.float64)
    H = df["home"].to_numpy(dtype=np.float64)


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
