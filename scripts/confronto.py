"""
confronto_finale_budget.py
============================
Confronto finale a parità di BUDGET COMPLETO tra 3 strategie:
  1. Random Search    → 100 sim totali per replica
  2. GA Puro          →  92 sim totali per replica
  3. GA + Surrogato   → 100 sim totali (50 train + 50 validazione)

Soglia crash: dist == 0.0 (solo collisioni reali, no near-miss).

Input atteso:
  Random:         dataset_random_r1.csv, r2.csv, r3.csv
  GA Puro:        risultati_solo_GA/replicaN/tutte_le_simulazioni.csv
  GA+Surr train:  dataset_pipeline_rN_50.csv  (50 training)
  GA+Surr val:    ga_results/pipeline_rN_50/validazione_risultati.csv (50 val)
  Surrogato CV:   models/pipeline_rN_50/confronto_modelli.csv
  FeatImp:        models/pipeline_rN_50/feature_importance.csv

Output (in confronto/):
  01_crash_3strategie.png          — n. crash medio per strategia
  02_distance_boxplot.png          — boxplot distanze sull'intero budget
  03_per_replica_crash.png         — variabilità inter-replica
  04_surrogate_accuracy.png        — pred vs reale per ogni replica
  05_feature_importance.png        — importance dei modelli
  06_surrogate_cv_metrics.png      — MAE/R² dei surrogati
  07_evolution_ga_puro.png         — convergenza GA puro
  report_finale.txt
  metriche_aggregate.csv

Uso:
    python3 confronto_finale_budget.py --output confronto/
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE PERCORSI
# ════════════════════════════════════════════════════════════════════════════

RANDOM_CSVS = [
    "dataset_random_r1.csv",
    "dataset_random_r2.csv",
    "dataset_random_r3.csv",
]

GA_PURO_CSVS = [
    "risultati_solo_GA/replica1/tutte_le_simulazioni.csv",
    "risultati_solo_GA/replica2/tutte_le_simulazioni.csv",
    "risultati_solo_GA/replica3/tutte_le_simulazioni.csv",
]

# Dataset di training del surrogato (50 sim) — fa parte del budget!
TRAIN_CSVS = [
    "dataset_pipeline_r1_50.csv",
    "dataset_pipeline_r2_50.csv",
    "dataset_pipeline_r3_50.csv",
]

# Validazione GA su surrogato (50 sim)
VAL_CSVS = [
    "ga_results/pipeline_r1_50/validazione_risultati.csv",
    "ga_results/pipeline_r2_50/validazione_risultati.csv",
    "ga_results/pipeline_r3_50/validazione_risultati.csv",
]

MODEL_CV_CSVS = [
    "models/pipeline_r1_50/confronto_modelli.csv",
    "models/pipeline_r2_50/confronto_modelli.csv",
    "models/pipeline_r3_50/confronto_modelli.csv",
]

FEAT_IMP_CSVS = [
    "models/pipeline_r1_50/feature_importance.csv",
    "models/pipeline_r2_50/feature_importance.csv",
    "models/pipeline_r3_50/feature_importance.csv",
]

# Soglia STRICT: solo collisioni reali, no near-miss
CRASH_THRESHOLD = 0.0    # == 0.0 esatto

COLORS = {
    "random":  "#1f77b4",
    "ga_puro": "#2ca02c",
    "ga_surr": "#d62728",
    "gb":      "#2ca02c",
    "rf":      "#ff7f0e",
    "nn":      "#9467bd",
}

# ════════════════════════════════════════════════════════════════════════════
# CARICAMENTO DATI
# ════════════════════════════════════════════════════════════════════════════

def is_crash(arr):
    """Crash = distanza esattamente 0.0 (collisione fisica)."""
    return arr == CRASH_THRESHOLD


def load_random_replica(csv_path):
    """Random Search: 100 sim."""
    df = pd.read_csv(csv_path)
    validi = df.dropna(subset=["distanza_minima_m"])
    dist = validi["distanza_minima_m"].values

    n_crash = int(is_crash(dist).sum())
    n_total = len(df)
    n_val   = len(validi)

    return {
        "n_total":  n_total,
        "n_valid":  n_val,
        "n_crash":  n_crash,
        "crash_rate": n_crash / n_val if n_val > 0 else 0.0,
        "dist_mean":   float(dist.mean()) if n_val > 0 else np.nan,
        "dist_median": float(np.median(dist)) if n_val > 0 else np.nan,
        "dist_min":    float(dist.min()) if n_val > 0 else np.nan,
        "dist_all":    dist.tolist(),
    }


def load_ga_puro_replica(csv_path):
    """GA Puro: 92 sim totali."""
    df = pd.read_csv(csv_path)
    validi = df.dropna(subset=["distanza_minima_m"])
    dist = validi["distanza_minima_m"].values

    n_crash = int(is_crash(dist).sum())
    n_total = len(df)
    n_val   = len(validi)

    # History per generazione (per plot evoluzione)
    gen_history = []
    if "generation" in df.columns:
        df_g = validi.copy()
        df_g["generation"] = pd.to_numeric(df_g["generation"], errors="coerce")
        df_g = df_g.dropna(subset=["generation"])
        df_g["generation"] = df_g["generation"].astype(int)
        for gen, sub in df_g.groupby("generation"):
            d = sub["distanza_minima_m"].values
            gen_history.append({
                "generation": int(gen),
                "best":  float(d.min()),
                "mean":  float(d.mean()),
                "crash": int(is_crash(d).sum()),
                "n":     len(d),
            })
        gen_history.sort(key=lambda x: x["generation"])

    return {
        "n_total":  n_total,
        "n_valid":  n_val,
        "n_crash":  n_crash,
        "crash_rate": n_crash / n_val if n_val > 0 else 0.0,
        "dist_mean":   float(dist.mean()) if n_val > 0 else np.nan,
        "dist_median": float(np.median(dist)) if n_val > 0 else np.nan,
        "dist_min":    float(dist.min()) if n_val > 0 else np.nan,
        "dist_all":    dist.tolist(),
        "gen_history": gen_history,
    }


def load_ga_surr_replica(train_csv, val_csv):
    """
    GA + Surrogato: il budget completo è training (50) + validazione (50).
    """
    # Training set (50 random samples)
    df_train = pd.read_csv(train_csv)
    train_valid = df_train.dropna(subset=["distanza_minima_m"])
    dist_train = train_valid["distanza_minima_m"].values

    # Validation set (50 candidati GA)
    df_val = pd.read_csv(val_csv)
    val_valid = df_val.dropna(subset=["distanza_reale_m"])
    dist_val = val_valid["distanza_reale_m"].values
    pred_val = (val_valid["pred_dist_m"].values
                if "pred_dist_m" in val_valid.columns else None)

    # Combine
    dist_all = np.concatenate([dist_train, dist_val])

    n_crash_train = int(is_crash(dist_train).sum())
    n_crash_val   = int(is_crash(dist_val).sum())
    n_crash_total = n_crash_train + n_crash_val

    n_total = len(df_train) + len(df_val)
    n_val_tot = len(train_valid) + len(val_valid)

    # MAE surrogato (solo sui candidati di validazione)
    if pred_val is not None and len(pred_val) == len(dist_val):
        mae = float(np.abs(pred_val - dist_val).mean())
        rmse = float(np.sqrt(((pred_val - dist_val) ** 2).mean()))
    else:
        mae = rmse = np.nan

    return {
        "n_total":  n_total,
        "n_valid":  n_val_tot,
        "n_crash":  n_crash_total,
        "n_crash_train": n_crash_train,
        "n_crash_val":   n_crash_val,
        "crash_rate":    n_crash_total / n_val_tot if n_val_tot > 0 else 0.0,
        "dist_mean":     float(dist_all.mean()) if n_val_tot > 0 else np.nan,
        "dist_median":   float(np.median(dist_all)) if n_val_tot > 0 else np.nan,
        "dist_min":      float(dist_all.min()) if n_val_tot > 0 else np.nan,
        "dist_all":      dist_all.tolist(),
        "dist_train":    dist_train.tolist(),
        "dist_val":      dist_val.tolist(),
        "pred_val":      pred_val.tolist() if pred_val is not None else [],
        "mae_val":       mae,
        "rmse_val":      rmse,
    }


def load_cv_replica(csv_path):
    """Metriche CV del surrogato (3 modelli)."""
    try:
        df = pd.read_csv(csv_path)
        gb = df[df["model"] == "GradientBoosting"].iloc[0]
        rf = df[df["model"] == "RandomForest"].iloc[0]
        nn = df[df["model"] == "NeuralNetwork"].iloc[0]
        return {
            "gb_mae":  float(gb["mae_mean"]),
            "gb_rmse": float(gb["rmse_mean"]),
            "gb_r2":   float(gb["r2_mean"]),
            "rf_mae":  float(rf["mae_mean"]),
            "rf_r2":   float(rf["r2_mean"]),
            "nn_mae":  float(nn["mae_mean"]),
            "nn_r2":   float(nn["r2_mean"]),
        }
    except Exception as e:
        print(f"   ⚠️  Impossibile caricare CV {csv_path}: {e}")
        return {k: np.nan for k in
                ["gb_mae","gb_rmse","gb_r2","rf_mae","rf_r2","nn_mae","nn_r2"]}


def load_feature_importance_avg(csv_paths):
    dfs = []
    for p in csv_paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all.groupby(["model","feature"])["importance"].mean().reset_index()


def aggregate(values):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return np.nan, np.nan
    return float(arr.mean()), float(arr.std())


# ════════════════════════════════════════════════════════════════════════════
# GRAFICI
# ════════════════════════════════════════════════════════════════════════════

def plot_crash_3strategie(rand_reps, ga_puro_reps, ga_surr_reps, output_dir):
    """Bar chart: n. crash medio (assoluto) e crash rate (%) per strategia."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    labels = ["Random Search\n(100 sim)", "GA Puro\n(92 sim)",
              "GA + Surrogato\n(50+50 sim)"]
    colors_list = [COLORS["random"], COLORS["ga_puro"], COLORS["ga_surr"]]

    # ── Panel sinistra: numero assoluto di crash ─────────────────────────
    ax = axes[0]
    n_crash_r = [r["n_crash"] for r in rand_reps]
    n_crash_p = [r["n_crash"] for r in ga_puro_reps]
    n_crash_s = [r["n_crash"] for r in ga_surr_reps]

    m_r, s_r = aggregate(n_crash_r)
    m_p, s_p = aggregate(n_crash_p)
    m_s, s_s = aggregate(n_crash_s)
    means = [m_r, m_p, m_s]
    stds  = [s_r, s_p, s_s]

    bars = ax.bar(labels, means, color=colors_list, edgecolor="black",
                  width=0.55, yerr=stds, capsize=8,
                  error_kw=dict(linewidth=2))
    for bar, val, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.5,
                f"{val:.1f}", ha="center", fontsize=13, fontweight="bold")

    ax.set_ylabel("Numero di crash (dist = 0.0m)", fontsize=11)
    ax.set_title("Crash assoluti per strategia\n(media ± std su 3 repliche)",
                 fontsize=12, fontweight="bold")
    ymax = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, ymax * 1.35 + 2)
    ax.grid(True, axis="y", alpha=0.3)

    # ── Panel destra: crash rate (%) ─────────────────────────────────────
    ax = axes[1]
    cr_r = [r["crash_rate"] * 100 for r in rand_reps]
    cr_p = [r["crash_rate"] * 100 for r in ga_puro_reps]
    cr_s = [r["crash_rate"] * 100 for r in ga_surr_reps]

    m_r2, s_r2 = aggregate(cr_r)
    m_p2, s_p2 = aggregate(cr_p)
    m_s2, s_s2 = aggregate(cr_s)
    means2 = [m_r2, m_p2, m_s2]
    stds2  = [s_r2, s_p2, s_s2]

    bars2 = ax.bar(labels, means2, color=colors_list, edgecolor="black",
                   width=0.55, yerr=stds2, capsize=8,
                   error_kw=dict(linewidth=2))
    for bar, val, std in zip(bars2, means2, stds2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 1,
                f"{val:.1f}%", ha="center", fontsize=13, fontweight="bold")

    ax.set_ylabel("Crash rate — % di sim con dist = 0.0m", fontsize=11)
    ax.set_title("Crash rate per strategia\n(% sul budget totale)",
                 fontsize=12, fontweight="bold")
    ymax2 = max(m + s for m, s in zip(means2, stds2))
    ax.set_ylim(0, ymax2 * 1.35 + 5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.suptitle("Crash assoluti (dist = 0.0m) — confronto a parità di budget",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_dir / "01_crash_3strategie.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 {out.name}")


def plot_distance_boxplot(rand_reps, ga_puro_reps, ga_surr_reps, output_dir):
    """Boxplot delle distanze reali per le 3 strategie, intero budget."""
    fig, ax = plt.subplots(figsize=(11, 7))

    data = [
        np.concatenate([r["dist_all"] for r in rand_reps]),
        np.concatenate([r["dist_all"] for r in ga_puro_reps]),
        np.concatenate([r["dist_all"] for r in ga_surr_reps]),
    ]
    colors_list = [COLORS["random"], COLORS["ga_puro"], COLORS["ga_surr"]]
    labels = ["Random Search\n(100 sim × 3)",
              "GA Puro\n(92 sim × 3)",
              "GA + Surrogato\n(100 sim × 3)"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2.5),
                    flierprops=dict(marker="o", markersize=4, alpha=0.4))
    for box, c in zip(bp["boxes"], colors_list):
        box.set_facecolor(c)
        box.set_alpha(0.65)

    for i, (d, c) in enumerate(zip(data, colors_list), start=1):
        xs = np.random.normal(i, 0.05, size=len(d))
        ax.scatter(xs, d, color=c, alpha=0.3, s=14, zorder=3)
        m = np.mean(d)
        med = np.median(d)
        ax.text(i, m + 0.15, f"μ={m:.2f}m\nmed={med:.2f}m",
                ha="center", fontsize=9, color="darkred", fontweight="bold")

    ax.axhline(y=CRASH_THRESHOLD, color="red", linestyle="--",
               linewidth=1.5, alpha=0.8,
               label=f"Soglia crash (dist = {CRASH_THRESHOLD}m)")
    ax.set_ylabel("Distanza minima reale (m)", fontsize=11)
    ax.set_title("Distribuzione delle distanze sull'intero budget\n"
                 "(3 strategie × 3 repliche)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "02_distance_boxplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 {out.name}")


def plot_per_replica_crash(rand_reps, ga_puro_reps, ga_surr_reps, output_dir):
    """Numero di crash per ogni singola replica delle 3 strategie."""
    fig, ax = plt.subplots(figsize=(12, 6.5))

    x = np.arange(3)
    width = 0.27
    labels_rep = ["Replica 1\n(seed 42)", "Replica 2\n(seed 123)",
                  "Replica 3\n(seed 456)"]

    n_r = [r["n_crash"] for r in rand_reps]
    n_p = [r["n_crash"] for r in ga_puro_reps]
    n_s = [r["n_crash"] for r in ga_surr_reps]

    bars_r = ax.bar(x - width, n_r, width, label="Random (100 sim)",
                    color=COLORS["random"], edgecolor="black", alpha=0.85)
    bars_p = ax.bar(x,         n_p, width, label="GA Puro (92 sim)",
                    color=COLORS["ga_puro"], edgecolor="black", alpha=0.85)
    bars_s = ax.bar(x + width, n_s, width, label="GA + Surrogato (100 sim)",
                    color=COLORS["ga_surr"], edgecolor="black", alpha=0.85)

    for bars in [bars_r, bars_p, bars_s]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{int(h)}", ha="center", fontsize=11, fontweight="bold")

    for vals, color, lab in [
        (n_r, COLORS["random"],  "Media Random"),
        (n_p, COLORS["ga_puro"], "Media GA Puro"),
        (n_s, COLORS["ga_surr"], "Media GA+Surr"),
    ]:
        m = np.mean(vals)
        ax.axhline(y=m, color=color, linestyle="--", alpha=0.5,
                   linewidth=1.5, label=f"{lab}: {m:.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_rep, fontsize=11)
    ax.set_ylabel("Numero di crash (dist = 0.0m)", fontsize=11)
    ax.set_title("Crash per singola replica (variabilità inter-replica)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ymax = max(max(n_r), max(n_p), max(n_s))
    ax.set_ylim(0, ymax * 1.35 + 3)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "03_per_replica_crash.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 {out.name}")


def plot_surrogate_accuracy(ga_surr_reps, output_dir):
    """Predetto vs reale per il surrogato (3 repliche)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors_rep = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    seeds = [42, 123, 456]

    for i, (rep, ax) in enumerate(zip(ga_surr_reps, axes)):
        pred = np.array(rep["pred_val"])
        real = np.array(rep["dist_val"])
        if len(pred) == 0 or len(real) == 0:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        n = min(len(pred), len(real))
        pred, real = pred[:n], real[:n]

        ax.scatter(pred, real, color=colors_rep[i], alpha=0.7,
                   s=60, edgecolor="black", linewidth=0.5)
        lo = min(pred.min(), real.min(), 0)
        hi = max(pred.max(), real.max(), 1)
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1.5,
                label="Predizione perfetta")

        mae = np.abs(pred - real).mean()
        r2 = (1 - np.sum((real - pred) ** 2) /
              np.sum((real - real.mean()) ** 2)) if np.var(real) > 0 else 0
        ax.text(0.05, 0.95, f"MAE = {mae:.3f}m\nN = {n}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"))
        ax.set_xlabel("Distanza predetta (m)", fontsize=10)
        ax.set_title(f"Replica {i+1} (seed {seeds[i]})",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        if i == 0:
            ax.set_ylabel("Distanza reale (m)", fontsize=10)

    plt.suptitle("Accuratezza del surrogato GradientBoosting in validazione",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "04_surrogate_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 {out.name}")


def plot_feature_importance(feat_imp_csvs, output_dir):
    df_imp = load_feature_importance_avg(feat_imp_csvs)
    if df_imp.empty:
        print("   ⚠️  Feature importance non disponibile")
        return
    models = ["GradientBoosting", "RandomForest", "NeuralNetwork"]
    colors_model = [COLORS["gb"], COLORS["rf"], COLORS["nn"]]
    feat_order = ["obs_y", "obs_x", "obs_l", "obs_w", "obs_h", "obs_r"]
    feat_labels = ["obs_y\n(lat.)", "obs_x\n(long.)", "obs_l\n(lung.)",
                   "obs_w\n(larg.)", "obs_h\n(alt.)", "obs_r\n(rot.)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, model, color in zip(axes, models, colors_model):
        df_m = df_imp[df_imp["model"] == model].copy()
        df_m = df_m.set_index("feature").reindex(feat_order).reset_index()
        importances = df_m["importance"].fillna(0).values
        bars = ax.barh(feat_labels, importances, color=color,
                       edgecolor="black", alpha=0.8)
        for bar, val in zip(bars, importances):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)
        ax.set_xlabel("Importanza relativa", fontsize=10)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(importances) * 1.2 if max(importances) > 0 else 1)
        ax.grid(True, axis="x", alpha=0.3)
    plt.suptitle("Feature importance media su 3 repliche",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "05_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 {out.name}")


def plot_surrogate_cv_metrics(cv_reps, output_dir):
    models = ["GradientBoosting", "RandomForest", "NeuralNetwork"]
    keys_mae = ["gb_mae", "rf_mae", "nn_mae"]
    keys_r2  = ["gb_r2",  "rf_r2",  "nn_r2"]
    colors_model = [COLORS["gb"], COLORS["rf"], COLORS["nn"]]
    mae_means, mae_stds, r2_means, r2_stds = [], [], [], []
    for k_mae, k_r2 in zip(keys_mae, keys_r2):
        m, s = aggregate([r[k_mae] for r in cv_reps])
        mae_means.append(m); mae_stds.append(s)
        m, s = aggregate([r[k_r2] for r in cv_reps])
        r2_means.append(m); r2_stds.append(s)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(models))

    ax = axes[0]
    bars = ax.bar(x, mae_means, color=colors_model, edgecolor="black",
                  alpha=0.8, yerr=mae_stds, capsize=7,
                  error_kw=dict(linewidth=2))
    for bar, m in zip(bars, mae_means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(mae_stds) + 0.005,
                f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["GB", "RF", "NN"], fontsize=12)
    ax.set_ylabel("MAE (m)", fontsize=11)
    ax.set_title("MAE cross-validation\n(media ± std su 3 repliche)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(mae_means) * 1.4 if max(mae_means) > 0 else 1)

    ax = axes[1]
    bars = ax.bar(x, r2_means, color=colors_model, edgecolor="black",
                  alpha=0.8, yerr=r2_stds, capsize=7,
                  error_kw=dict(linewidth=2))
    for bar, m in zip(bars, r2_means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(r2_stds) + 0.01,
                f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["GB", "RF", "NN"], fontsize=12)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("R² cross-validation\n(media ± std su 3 repliche)",
                 fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Qualità dei modelli surrogati (5-fold CV su 3 repliche)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "06_surrogate_cv_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 {out.name}")


def plot_ga_puro_evolution(ga_puro_reps, output_dir):
    """Convergenza del GA puro: best fitness e # crash per generazione."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_rep = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    seeds = [42, 123, 456]

    ax = axes[0]
    for i, rep in enumerate(ga_puro_reps):
        hist = rep.get("gen_history", [])
        if not hist:
            continue
        gens = [h["generation"] for h in hist]
        bests = [h["best"] for h in hist]
        ax.plot(gens, bests, "-o", color=colors_rep[i], linewidth=2,
                label=f"Replica {i+1} (seed {seeds[i]})")
    ax.axhline(y=CRASH_THRESHOLD, color="red", linestyle="--", alpha=0.7,
               label=f"Soglia crash ({CRASH_THRESHOLD}m)")
    ax.set_xlabel("Generazione")
    ax.set_ylabel("Best fitness — distanza minima (m)")
    ax.set_title("Convergenza GA puro: best per generazione",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    width = 0.27
    for i, rep in enumerate(ga_puro_reps):
        hist = rep.get("gen_history", [])
        if not hist:
            continue
        gens = np.array([h["generation"] for h in hist])
        crash = [h["crash"] for h in hist]
        ax.bar(gens + (i - 1) * width, crash, width,
               color=colors_rep[i], edgecolor="black", alpha=0.8,
               label=f"Replica {i+1}")
    ax.set_xlabel("Generazione")
    ax.set_ylabel("# crash (dist = 0.0m)")
    ax.set_title("Crash trovati per generazione",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Evoluzione del GA Puro su simulatore reale",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "07_evolution_ga_puro.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# REPORT TESTUALE
# ════════════════════════════════════════════════════════════════════════════

def genera_report(rand_reps, ga_puro_reps, ga_surr_reps, cv_reps, output_dir):
    def fmt(m, s, unit=""):
        if np.isnan(m):
            return "—"
        return f"{m:.3f}{unit} ± {s:.3f}"

    # Aggregazione metriche
    r_n_crash    = aggregate([r["n_crash"] for r in rand_reps])
    p_n_crash    = aggregate([r["n_crash"] for r in ga_puro_reps])
    s_n_crash    = aggregate([r["n_crash"] for r in ga_surr_reps])

    r_cr_rate    = aggregate([r["crash_rate"] * 100 for r in rand_reps])
    p_cr_rate    = aggregate([r["crash_rate"] * 100 for r in ga_puro_reps])
    s_cr_rate    = aggregate([r["crash_rate"] * 100 for r in ga_surr_reps])

    r_mean       = aggregate([r["dist_mean"]   for r in rand_reps])
    p_mean       = aggregate([r["dist_mean"]   for r in ga_puro_reps])
    s_mean       = aggregate([r["dist_mean"]   for r in ga_surr_reps])

    r_med        = aggregate([r["dist_median"] for r in rand_reps])
    p_med        = aggregate([r["dist_median"] for r in ga_puro_reps])
    s_med        = aggregate([r["dist_median"] for r in ga_surr_reps])

    r_min        = aggregate([r["dist_min"]    for r in rand_reps])
    p_min        = aggregate([r["dist_min"]    for r in ga_puro_reps])
    s_min        = aggregate([r["dist_min"]    for r in ga_surr_reps])

    s_mae        = aggregate([r["mae_val"]     for r in ga_surr_reps])
    s_rmse       = aggregate([r["rmse_val"]    for r in ga_surr_reps])

    gb_mae_cv    = aggregate([r["gb_mae"] for r in cv_reps])
    gb_r2_cv     = aggregate([r["gb_r2"]  for r in cv_reps])

    lines = []
    lines.append("═" * 80)
    lines.append("   CONFRONTO FINALE — 3 STRATEGIE A PARITÀ DI BUDGET")
    lines.append("   DOE accoppiato: 3 repliche (seed 42, 123, 456)")
    lines.append(f"   Soglia crash: distanza == {CRASH_THRESHOLD}m (solo collisioni reali)")
    lines.append("═" * 80)
    lines.append("")
    lines.append(f"{'METRICA':<40} {'Random':<14} {'GA Puro':<14} {'GA+Surr':<14}")
    lines.append("─" * 80)
    lines.append(f"{'Budget simulazioni reali':<40} {'100':<14} {'92':<14} "
                 f"{'100 (50+50)':<14}")
    lines.append("")

    lines.append(">> CRASH ASSOLUTI (dist = 0.0m)")
    lines.append("─" * 80)
    lines.append(f"{'Numero medio di crash':<40} "
                 f"{fmt(*r_n_crash):<14} {fmt(*p_n_crash):<14} "
                 f"{fmt(*s_n_crash):<14}")
    lines.append(f"{'Crash rate (%)':<40} "
                 f"{fmt(*r_cr_rate, '%'):<14} {fmt(*p_cr_rate, '%'):<14} "
                 f"{fmt(*s_cr_rate, '%'):<14}")
    lines.append("")

    lines.append(">> DISTANZA MINIMA REALE (m) — sull'intero budget")
    lines.append("─" * 80)
    lines.append(f"{'Media':<40} "
                 f"{fmt(*r_mean, 'm'):<14} {fmt(*p_mean, 'm'):<14} "
                 f"{fmt(*s_mean, 'm'):<14}")
    lines.append(f"{'Mediana':<40} "
                 f"{fmt(*r_med, 'm'):<14} {fmt(*p_med, 'm'):<14} "
                 f"{fmt(*s_med, 'm'):<14}")
    lines.append(f"{'Minima trovata':<40} "
                 f"{fmt(*r_min, 'm'):<14} {fmt(*p_min, 'm'):<14} "
                 f"{fmt(*s_min, 'm'):<14}")
    lines.append("")

    lines.append(">> QUALITÀ SURROGATO GRADIENTBOOSTING")
    lines.append("─" * 80)
    lines.append(f"{'MAE cross-validation':<40} {'—':<14} {'—':<14} "
                 f"{fmt(*gb_mae_cv, 'm'):<14}")
    lines.append(f"{'R² cross-validation':<40} {'—':<14} {'—':<14} "
                 f"{fmt(*gb_r2_cv):<14}")
    lines.append(f"{'MAE validazione (pred vs reale)':<40} {'—':<14} {'—':<14} "
                 f"{fmt(*s_mae, 'm'):<14}")
    lines.append(f"{'RMSE validazione':<40} {'—':<14} {'—':<14} "
                 f"{fmt(*s_rmse, 'm'):<14}")
    lines.append("")

    lines.append("═" * 80)
    lines.append("")
    lines.append(">> CRASH PER SINGOLA REPLICA")
    lines.append("─" * 80)
    lines.append(f"{'Replica':<14} {'Random':<14} {'GA Puro':<14} "
                 f"{'GA+Surr':<25} {'MAE surr'}")
    lines.append(f"{'':14} {'(/100)':<14} {'(/92)':<14} "
                 f"{'(train+val=100)':<25}")
    lines.append("─" * 80)
    seeds = [42, 123, 456]
    for i, (r, p, s, sd) in enumerate(zip(rand_reps, ga_puro_reps,
                                            ga_surr_reps, seeds)):
        gas_str = f"{s['n_crash']:>2}  ({s['n_crash_train']} train + " \
                  f"{s['n_crash_val']} val)"
        lines.append(
            f"{'Rep '+str(i+1)+' (s='+str(sd)+')':<14} "
            f"{r['n_crash']:>5}          "
            f"{p['n_crash']:>5}          "
            f"{gas_str:<25} "
            f"{s['mae_val']:>6.3f}m"
        )
    lines.append("═" * 80)
    lines.append("")
    lines.append(">> INTERPRETAZIONE — crash assoluti")
    lines.append("─" * 80)
    if not np.isnan(r_n_crash[0]) and r_n_crash[0] > 0:
        gain_p = (p_n_crash[0] / r_n_crash[0] - 1) * 100
        gain_s = (s_n_crash[0] / r_n_crash[0] - 1) * 100
        lines.append(f"  GA Puro vs Random:        {gain_p:+.1f}%")
        lines.append(f"  GA + Surrogato vs Random: {gain_s:+.1f}%")
    if not np.isnan(p_n_crash[0]) and p_n_crash[0] > 0:
        gain_sp = (s_n_crash[0] / p_n_crash[0] - 1) * 100
        lines.append(f"  GA + Surrogato vs GA Puro: {gain_sp:+.1f}%")
    lines.append("═" * 80)

    report = "\n".join(lines)
    print("\n" + report)
    out = output_dir / "report_finale.txt"
    with open(out, "w") as f:
        f.write(report)
    print(f"\n   📄 Report → {out.name}")
    return report


def salva_metriche_csv(rand_reps, ga_puro_reps, ga_surr_reps, cv_reps, output_dir):
    rows = []
    seeds = [42, 123, 456]
    for i, (r, p, s, c, sd) in enumerate(zip(rand_reps, ga_puro_reps,
                                               ga_surr_reps, cv_reps, seeds)):
        rows.append({
            "replica": i + 1, "seed": sd,
            "rs_n_total":    r["n_total"],
            "rs_n_crash":    r["n_crash"],
            "rs_crash_rate": round(r["crash_rate"], 4),
            "rs_dist_mean":  round(r["dist_mean"],  3),
            "rs_dist_min":   round(r["dist_min"],   3),
            "gap_n_total":    p["n_total"],
            "gap_n_crash":    p["n_crash"],
            "gap_crash_rate": round(p["crash_rate"], 4),
            "gap_dist_mean":  round(p["dist_mean"],  3),
            "gap_dist_min":   round(p["dist_min"],   3),
            "gas_n_total":       s["n_total"],
            "gas_n_crash_train": s["n_crash_train"],
            "gas_n_crash_val":   s["n_crash_val"],
            "gas_n_crash":       s["n_crash"],
            "gas_crash_rate":    round(s["crash_rate"], 4),
            "gas_dist_mean":     round(s["dist_mean"],  3),
            "gas_dist_min":      round(s["dist_min"],   3),
            "gas_mae_surr":      round(s["mae_val"],    3),
            "gas_rmse_surr":     round(s["rmse_val"],   3),
            "surr_gb_mae_cv":    round(c["gb_mae"], 4),
            "surr_gb_r2_cv":     round(c["gb_r2"],  4),
        })
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ["replica", "seed"]]
    mean_row = {"replica": "media", "seed": "—"}
    std_row  = {"replica": "std",   "seed": "—"}
    for col in num_cols:
        vals = df[col].values.astype(float)
        mean_row[col] = round(float(vals.mean()), 4)
        std_row[col]  = round(float(vals.std()),  4)
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    out = output_dir / "metriche_aggregate.csv"
    df.to_csv(out, index=False)
    print(f"   📄 Metriche CSV → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="confronto/",
                        help="Cartella output (default: confronto/)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 70)
    print("📊 CONFRONTO FINALE — 3 STRATEGIE A PARITÀ DI BUDGET")
    print(f"   Soglia crash: dist == {CRASH_THRESHOLD}m (solo collisioni reali)")
    print("═" * 70)

    print("\n📂 Caricamento dati...")
    rand_reps, ga_puro_reps, ga_surr_reps, cv_reps = [], [], [], []
    for i in range(3):
        print(f"   Replica {i+1}:")
        print(f"      random:  {Path(RANDOM_CSVS[i]).name}")
        print(f"      ga_puro: {Path(GA_PURO_CSVS[i]).parent.name}/")
        print(f"      train:   {Path(TRAIN_CSVS[i]).name}")
        print(f"      val:     {Path(VAL_CSVS[i]).parent.name}/")
        rand_reps.append(load_random_replica(RANDOM_CSVS[i]))
        ga_puro_reps.append(load_ga_puro_replica(GA_PURO_CSVS[i]))
        ga_surr_reps.append(load_ga_surr_replica(TRAIN_CSVS[i], VAL_CSVS[i]))
        cv_reps.append(load_cv_replica(MODEL_CV_CSVS[i]))

    print("\n📊 Generazione grafici...")
    plot_crash_3strategie(rand_reps, ga_puro_reps, ga_surr_reps, output_dir)
    plot_distance_boxplot(rand_reps, ga_puro_reps, ga_surr_reps, output_dir)
    plot_per_replica_crash(rand_reps, ga_puro_reps, ga_surr_reps, output_dir)
    plot_surrogate_accuracy(ga_surr_reps, output_dir)
    plot_feature_importance(FEAT_IMP_CSVS, output_dir)
    plot_surrogate_cv_metrics(cv_reps, output_dir)
    plot_ga_puro_evolution(ga_puro_reps, output_dir)

    print("\n📄 Report e metriche aggregate...")
    genera_report(rand_reps, ga_puro_reps, ga_surr_reps, cv_reps, output_dir)
    salva_metriche_csv(rand_reps, ga_puro_reps, ga_surr_reps, cv_reps, output_dir)

    print(f"\n{'═'*70}")
    print(f"✅ CONFRONTO COMPLETATO — output in: {output_dir}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()