"""
wow_betting.py
==============
"Wow factor": posterior-predictive betting-strategy simulation.

For each game in the dataset, draw S samples of the predicted spread error
under the fitted model:

    X*_i = theta_{j[i]} + beta_1 R_i + beta_2 H_i + gamma_{s[i]} + eps,
    eps ~ N(0, sigma^2).

We compute Pr(X* > t | data) for two cover thresholds t in {0, +1}.
A simple strategy: bet on the favorite to cover when P(X*>0) >= cutoff,
bet the underdog when P(X*>0) <= 1-cutoff, otherwise no bet. Settle each
realized bet at standard -110 odds (win 100/110, lose 1).

Outputs:
    results/figures/07_wow_betting.png
    results/wow_betting.txt    — text summary used in the writeup
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


def main():
    here = Path(__file__).resolve().parents[1]
    df = pd.read_csv(here / "data" / "nba_spreads.csv")
    with np.load(here / "results" / "posterior_primary.npz", allow_pickle=True) as f:
        s = {k: f[k] for k in f.files}

    teams = list(s["teams"])
    seasons = list(s["seasons"])
    team_to_id = {t: i for i, t in enumerate(teams)}
    season_to_id = {ss: i for i, ss in enumerate(seasons)}

    # Stack all kept samples: (M, T, ...) -> (M*T, ...)
    th = s["theta"].reshape(-1, len(teams))
    gm = s["gamma"].reshape(-1, len(seasons))
    b1 = s["beta1"].reshape(-1)
    b2 = s["beta2"].reshape(-1)
    sg2 = s["sigma2"].reshape(-1)

    rng = np.random.default_rng(42)
    # Subsample posterior draws for speed (still huge — 2000 draws)
    n_draws = 2000
    idx = rng.choice(len(b1), size=n_draws, replace=False)

    th = th[idx]; gm = gm[idx]
    b1 = b1[idx]; b2 = b2[idx]; sg2 = sg2[idx]

    j = df["fav_team"].map(team_to_id).to_numpy()
    sidx = df["season"].map(season_to_id).to_numpy()
    R = df["rest_diff"].to_numpy()
    H = df["home"].to_numpy()
    X = df["error"].to_numpy()
    N = len(df)

    # Posterior expected error per game, vectorised across draws
    # mean[d, i] = theta[d, j[i]] + beta1[d]*R[i] + beta2[d]*H[i] + gamma[d, s[i]]
    mean = th[:, j] + b1[:, None] * R + b2[:, None] * H + gm[:, sidx]
    sd = np.sqrt(sg2)[:, None]

    # P(X*>0 | data) per game, averaged over posterior draws
    from scipy.stats import norm
    p_cover = norm.sf(0, loc=mean, scale=sd).mean(axis=0)   # length N
    p_lt0 = 1 - p_cover

    # Bet outcomes at -110: stake 1 unit, profit +0.909 if win, -1 if loss.
    # Favorite "covers" when X > 0; underdog covers when X < 0.
    fav_won_cover = (X > 0).astype(int)
    dog_won_cover = (X < 0).astype(int)
    push = (X == 0).astype(int)

    # Strategy: vary cutoff; bet fav if p_cover>=cutoff, dog if p_cover<=1-cutoff.
    cutoffs = np.linspace(0.5, 0.7, 21)
    payouts_fav = np.where(fav_won_cover == 1, 0.909, -1.0)
    payouts_dog = np.where(dog_won_cover == 1, 0.909, -1.0)
    payouts_fav[push == 1] = 0
    payouts_dog[push == 1] = 0

    rows = []
    for c in cutoffs:
        bet_fav = p_cover >= c
        bet_dog = p_cover <= 1 - c
        n_bets = int(bet_fav.sum() + bet_dog.sum())
        profit = payouts_fav[bet_fav].sum() + payouts_dog[bet_dog].sum()
        roi = profit / n_bets * 100 if n_bets else 0.0
        # Hit rate
        wins = int(fav_won_cover[bet_fav].sum() + dog_won_cover[bet_dog].sum())
        rows.append((c, n_bets, wins, roi))

    print(f"{'cutoff':>7s}  {'bets':>6s}  {'wins':>5s}  {'win%':>6s}  {'ROI(%)':>7s}")
    for c, nb, w, r in rows:
        wr = w / nb * 100 if nb else 0
        print(f"{c:7.3f}  {nb:6d}  {w:5d}  {wr:6.2f}  {r:7.2f}")

    # ---- Figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: ROI vs cutoff (+ random-betting baseline)
    ax = axes[0]
    cs = np.array([r[0] for r in rows])
    rois = np.array([r[3] for r in rows])
    nbets = np.array([r[1] for r in rows])
    ax.plot(cs, rois, "-o", color="#4c72b0", ms=4, label="model strategy")
    ax.axhline(-4.55, color="#c44e52", ls="--", lw=1,
               label="break-even at -110 (-4.55%)")
    ax2 = ax.twinx()
    ax2.bar(cs, nbets, width=0.005, color="#cccccc", alpha=0.55,
            label="bets placed")
    ax.set_xlabel("Cutoff on Pr(favorite covers | data)")
    ax.set_ylabel("ROI per bet (%)")
    ax2.set_ylabel("Bets placed (gray bars)")
    ax.set_title("Posterior-predictive betting strategy")
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    # Right: Distribution of P(fav covers) across all games + actual hit rate
    ax = axes[1]
    bins = np.linspace(0.3, 0.7, 25)
    centers = (bins[:-1] + bins[1:]) / 2
    digitized = np.digitize(p_cover, bins) - 1
    digitized = np.clip(digitized, 0, len(centers) - 1)
    actual = np.array([fav_won_cover[digitized == k].mean()
                       if (digitized == k).sum() > 30 else np.nan
                       for k in range(len(centers))])
    counts = np.array([(digitized == k).sum() for k in range(len(centers))])
    ax.bar(centers, counts, width=0.014, color="#cccccc", alpha=0.5,
           label="games per bin")
    ax2 = ax.twinx()
    ax2.plot([0.3, 0.7], [0.3, 0.7], color="#888", lw=0.7, ls="--")
    ax2.plot(centers, actual, "o", color="#4c72b0", ms=5,
             label="actual cover rate")
    ax.set_xlabel(r"Posterior $\Pr(\text{favorite covers}\mid\text{data})$")
    ax.set_ylabel("Game count (gray)")
    ax2.set_ylabel("Actual cover rate")
    ax2.set_ylim(0.3, 0.7)
    ax.set_title("Calibration: predicted vs realized cover rate")
    ax2.legend(frameon=False, loc="upper left", fontsize=8)

    fig.tight_layout()
    out = here / "results" / "figures" / "07_wow_betting.png"
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)

    # Save text summary for the writeup
    summary_path = here / "results" / "wow_betting.txt"
    best = max(rows, key=lambda r: (r[3] if r[1] >= 100 else -1e9))
    with open(summary_path, "w") as fh:
        fh.write(f"Best ROI cutoff: {best[0]:.3f} -> "
                 f"{best[1]} bets, {best[2]} wins, "
                 f"{best[2]/best[1]*100:.2f}% hit rate, "
                 f"{best[3]:+.2f}% ROI\n")
        fh.write(f"\ncutoff  bets  wins  win%  ROI(%)\n")
        for c, nb, w, r in rows:
            wr = w / nb * 100 if nb else 0
            fh.write(f"{c:.3f}  {nb}  {w}  {wr:.2f}  {r:+.2f}\n")
    print("Saved", summary_path)


if __name__ == "__main__":
    main()
