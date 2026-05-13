from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})



def plot_error_hist(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    err = df["error"].values
    ax.hist(err, bins=60, density=True, color="#4c72b0", alpha=0.85,
            edgecolor="white", linewidth=0.4)
    xs = np.linspace(err.min(), err.max(), 400)


    ax.plot(xs, stats.norm.pdf(xs, err.mean(), err.std()),
            color="#c44e52", lw=2, label=f"N({err.mean():.2f}, {err.std():.2f}²)")
    ax.set_xlabel("Spread error  X = actual margin − |spread|  (favorite-perspective, points)")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of spread errors  (n={len(err):,})")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_team_means(df: pd.DataFrame, out: Path) -> None:
    g = df.groupby("fav_team")["error"].agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    g = g.sort_values("mean")

    
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    y = np.arange(len(g))
    ax.errorbar(g["mean"], y, xerr=2 * g["se"], fmt="o", color="#4c72b0",
                ecolor="#cccccc", capsize=2, ms=5)
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=8)
    ax.set_xlabel("Mean spread error as favorite (points), ±2 SE")
    ax.set_title("Raw mean spread error by team (no shrinkage)")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_season_means(df: pd.DataFrame, out: Path) -> None:
    g = df.groupby("season")["error"].agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(np.arange(len(g)), g["mean"], yerr=2 * g["se"],
                fmt="o-", color="#4c72b0", capsize=3)
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xticks(np.arange(len(g))); ax.set_xticklabels(g.index, rotation=45, ha="right")
    ax.set_xlabel("Season"); ax.set_ylabel("Mean spread error (pts), ±2 SE")
    ax.set_title("Mean spread error by season")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_home_rest(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    #home vs road favorite
    ax = axes[0]
    g = df.groupby("home")["error"].agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    ax.bar([0, 1], g["mean"], yerr=2 * g["se"], capsize=4,
           color=["#dd8452", "#4c72b0"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Road favorite", "Home favorite"])
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_ylabel("Mean spread error (pts), ±2 SE")
    ax.set_title("Mean spread error by favorite location")


    ax = axes[1]
    df2 = df.copy()
    df2["rdb"] = pd.cut(df2["rest_diff_raw"],
                       bins=[-10, -2, -1, 0, 1, 2, 10],
                       labels=["≤-2", "-1", "0", "+1", "+2", "≥+3"])
    g = df2.groupby("rdb")["error"].agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    ax.errorbar(np.arange(len(g)), g["mean"], yerr=2 * g["se"],
                fmt="o", capsize=4, color="#4c72b0")
    ax.set_xticks(np.arange(len(g))); ax.set_xticklabels(g.index)
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xlabel("Rest differential (fav rest − dog rest, days)")
    ax.set_ylabel("Mean spread error (pts), ±2 SE")
    ax.set_title("Mean spread error by rest differential")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_sample_sizes(df: pd.DataFrame, out: Path) -> None:
    g = df["fav_team"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.barh(np.arange(len(g)), g.values, color="#4c72b0")
    ax.set_yticks(np.arange(len(g))); ax.set_yticklabels(g.index, fontsize=8)
    ax.set_xlabel("Games as favorite (n)")
    ax.set_title("Sample size per team (favorite role only)")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    df = pd.read_csv(here / "data" / "nba_spreads.csv")
    out_dir = here / "eda"
    out_dir.mkdir(exist_ok=True)
    plot_error_hist(df, out_dir / "01_error_hist.png")
    plot_team_means(df, out_dir / "02_team_means.png")
    plot_season_means(df, out_dir / "03_season_means.png")
    plot_home_rest(df, out_dir / "04_home_and_rest.png")
    plot_sample_sizes(df, out_dir / "05_sample_sizes.png")
    print("EDA plots saved to", out_dir)


    print(f"\nGrand mean error: {df['error'].mean():+.3f}  SD: {df['error'].std():.3f}")
    print(f"Shapiro p-value (sample of 5000): {stats.shapiro(df['error'].sample(min(5000, len(df)), random_state=0)).pvalue:.3g}")
    print(f"Skew: {stats.skew(df['error']):.3f}, Kurt(excess): {stats.kurtosis(df['error']):.3f}")


if __name__ == "__main__":
    main()
