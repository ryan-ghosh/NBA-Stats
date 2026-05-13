from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def split_chains(x: np.ndarray) -> np.ndarray:
    M, T = x.shape[0], x.shape[1]
    half = T // 2

    a = x[:, :half]
    b = x[:, half:2 * half]

    return np.concatenate([a, b], axis=0)


def rhat(x: np.ndarray) -> float | np.ndarray:

    s = split_chains(x)
    M, T = s.shape[0], s.shape[1]

    chain_mean = s.mean(axis=1)
    grand_mean = chain_mean.mean(axis=0)

    B = T * np.var(chain_mean, axis=0, ddof=1)
    W = np.mean(np.var(s, axis=1, ddof=1), axis=0)
    var_hat = (1 - 1/T)*W + B/T


    return np.sqrt(var_hat / W)


def ess(x: np.ndarray, max_lag: int | None = None) -> float | np.ndarray:

    if x.ndim == 2:
        return _ess_1d(x.reshape(-1))
    
    K = x.shape[-1]

    return np.array([_ess_1d(x[..., k].reshape(-1)) for k in range(K)])


def _ess_1d(z: np.ndarray, max_lag: int | None = None) -> float:

    n = len(z)
    z = z - z.mean()
    var = np.var(z, ddof=1)

    if var == 0:
        return float(n)
    
    f = np.fft.fft(z, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf = acf / acf[0]

    #Geyer sequence: sum pairs while positive and monotonically decreasing
    rho_pair = acf[:-1:2] + acf[1::2]
    cum_min = np.minimum.accumulate(rho_pair)
    keep = cum_min > 0

    if not keep.any():
        return float(n)
    
    last = int(np.argmax(~keep)) if (~keep).any() else len(rho_pair)

    if last == 0:
        return float(n)
    
    tau = -1 + 2 * rho_pair[:last].sum()

    if tau < 1:
        tau = 1.0

    return float(n / tau)


def trace_plot(samples: dict, params: list[tuple[str, str]],
               out_path: Path) -> None:
    
    n = len(params)
    fig, axes = plt.subplots(n, 2, figsize=(10, 1.8 * n))

    if n == 1:
        axes = axes.reshape(1, 2)


    for i, (k, label) in enumerate(params):

        x = samples[k]
        if x.ndim == 3:
            x = x[..., 0]

        ax = axes[i, 0]
        for c in range(x.shape[0]):
            ax.plot(x[c], lw=0.5, alpha=0.7)

        ax.set_ylabel(label, fontsize=9)
        if i == 0: ax.set_title("Trace", fontsize=10)


        ax = axes[i, 1]
        z = x.reshape(-1) - x.mean()
        T = len(z)

        f = np.fft.fft(z, n=2 * T)
        acf = np.fft.ifft(f * np.conj(f))[:T].real
        acf = acf / acf[0]
        L = min(80, T)


        ax.bar(np.arange(L), acf[:L], color="#4c72b0", width=0.9)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylim(-0.2, 1.05)

        if i == n - 1:
            ax.set_xlabel("Lag", fontsize=9)

        if i == 0:
            ax.set_title("Autocorrelation (combined chains)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def summarize(npz_path: Path, out_dir: Path, label: str) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as f:
        s = {k: f[k] for k in f.files}

    teams = s["teams"]; seasons = s["seasons"]

    n_chains, n_keep = s["mu"].shape


    rows = []

    def addrow(label, x):
        flat = x.reshape(-1)
        m = flat.mean(); sd = flat.std(ddof=1)
        lo, hi = np.quantile(flat, [0.025, 0.975])
        rows.append((label, m, sd, lo, hi, float(rhat(x)), float(ess(x))))


    addrow("mu", s["mu"])
    addrow("tau^2", s["tau2"])
    addrow("tau_s^2", s["tau_s2"])
    addrow("sigma^2", s["sigma2"])
    addrow("beta1 (rest)", s["beta1"])
    addrow("beta2 (home)", s["beta2"])
    rho_chain = s["tau2"] / (s["tau2"] + s["tau_s2"] + s["sigma2"])
    addrow("rho (ICC)", rho_chain)

    #theta summary
    th = s["theta"]
    rh = rhat(th); es = ess(th)
    th_means = th.reshape(-1, th.shape[-1]).mean(axis=0)

    #gamma summary
    gm = s["gamma"]
    rh_g = rhat(gm); es_g = ess(gm)



    print(f"\n=== {label} prior ({n_chains} chains × {n_keep} samples each) ===")
    print(f"{'param':14s}  {'mean':>9s}  {'sd':>7s}  {'2.5%':>7s}  {'97.5%':>8s}  {'R-hat':>6s}  {'ESS':>7s}")

    for r in rows:
        print(f"{r[0]:14s}  {r[1]:+9.4f}  {r[2]:7.4f}  {r[3]:+7.4f}  {r[4]:+8.4f}  "
              f"{r[5]:6.3f}  {r[6]:7.0f}")
        
    print(f"\nTeam intercepts theta_j: max R-hat = {rh.max():.3f}, "
          f"min ESS = {es.min():.0f}")
    
    print(f"Season intercepts gamma_s: max R-hat = {rh_g.max():.3f}, "
          f"min ESS = {es_g.min():.0f}")


    md = [f"## Posterior summary — {label} prior\n",
          f"Sampler: {n_chains} chains × {n_keep} kept iterations after burn-in.\n",
          "| Parameter | Mean | SD | 2.5% | 97.5% | R-hat | ESS |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    
    for r in rows:
        md.append(f"| {r[0]} | {r[1]:+.4f} | {r[2]:.4f} | {r[3]:+.4f} | "
                  f"{r[4]:+.4f} | {r[5]:.3f} | {r[6]:.0f} |")
        
    md.append(f"\n*Team intercepts $\\theta_j$ (J=30): max R-hat = {rh.max():.3f}, "
              f"min ESS = {es.min():.0f}.*")
    
    md.append(f"*Season intercepts $\\gamma_s$ (S=10): max R-hat = {rh_g.max():.3f}, "
              f"min ESS = {es_g.min():.0f}.*")
    
    out_md = out_dir / f"summary_{label}.md"
    out_md.write_text("\n".join(md))
    



    trace_plot(s, [("mu", r"$\mu$"),
                   ("tau2", r"$\tau^2$"),
                   ("tau_s2", r"$\tau_s^2$"),
                   ("sigma2", r"$\sigma^2$"),
                   ("beta1", r"$\beta_1$ (rest)"),
                   ("beta2", r"$\beta_2$ (home)")],
               out_dir / f"trace_{label}.png")


    fig, axes = plt.subplots(2, 2, figsize=(10, 4.5))

    for c in range(th.shape[0]):
        axes[0, 0].plot(th[c, :, 0], lw=0.4, alpha=0.7)

    axes[0, 0].set_title(rf"trace $\theta_{{{teams[0]}}}$")

    for c in range(gm.shape[0]):
        axes[0, 1].plot(gm[c, :, 0], lw=0.4, alpha=0.7)

    axes[0, 1].set_title(rf"trace $\gamma_{{{seasons[0]}}}$")


    for ax_idx, (vec, name) in enumerate([(th[..., 0], "theta_0"),
                                          (gm[..., 0], "gamma_0")]):
        z = vec.reshape(-1); z = z - z.mean()
        f = np.fft.fft(z, n=2 * len(z))
        acf = np.fft.ifft(f * np.conj(f))[:len(z)].real / np.dot(z, z)
        L = 80
        axes[1, ax_idx].bar(np.arange(L), acf[:L], color="#4c72b0", width=0.9)
        axes[1, ax_idx].axhline(0, color="black", lw=0.5)
        axes[1, ax_idx].set_xlabel("Lag")
        axes[1, ax_idx].set_title(f"ACF {name}")

    fig.tight_layout()
    fig.savefig(out_dir / f"trace_groups_{label}.png")
    plt.close(fig)
    

    return {"rows": rows, "theta_rhat_max": rh.max(),
            "theta_ess_min": es.min(),
            "gamma_rhat_max": rh_g.max(),
            "gamma_ess_min": es_g.min()}


def main():
    here = Path(__file__).resolve().parents[1]
    out_dir = here / "results" / "diagnostics"
    summarize(here / "results" / "posterior_primary.npz", out_dir, "primary")
    summarize(here / "results" / "posterior_tight.npz", out_dir, "tight")


if __name__ == "__main__":
    main()
