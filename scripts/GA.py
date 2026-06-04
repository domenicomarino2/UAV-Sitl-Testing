"""
algoritmo_genetico_puro.py
============================
Algoritmo Genetico (GA) che valuta DIRETTAMENTE sul simulatore Aerialist
ogni individuo, SENZA usare un modello surrogato.

Differenze rispetto a algoritmo_genetico.py:
  - evaluate_population() lancia simulazioni reali via Docker (~2.5 min/sim)
  - Budget fissato a numero di simulazioni reali (non al numero di generazioni)
  - Popolazione ridotta (20) e generazioni ridotte (5) per stare nel budget
  - Salva incrementalmente ogni valutazione nel CSV (resilienza a crash)

Budget = POP_SIZE × N_GEN = 20 × 5 = 100 simulazioni reali

Pipeline:
  1. Inizializza popolazione random (20 individui) → 20 sim reali (gen 0)
  2. Per ogni generazione successiva:
       a) Seleziona, ricombina, muta → genera 20 nuovi individui
       b) Valuta i nuovi 20 sul simulatore reale → 20 sim reali
  3. Totale: 20 + 4×20 = 100 simulazioni reali
  4. Salva tutti i 100 individui valutati + top-30 per la validazione finale

Uso:
    python3 algoritmo_genetico_puro.py --output ga_puro_results/replica1/ --seed 42
    python3 algoritmo_genetico_puro.py --output ga_puro_results/replica2/ --seed 123
    python3 algoritmo_genetico_puro.py --output ga_puro_results/replica3/ --seed 456
"""

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyulog import ULog

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ════════════════════════════════════════════════════════════════════════════

# Feature dell'ostacolo
FEATURE_NAMES = ["obs_l", "obs_w", "obs_h", "obs_x", "obs_y", "obs_r"]

# Bounds IDENTICI a quelli del generatore dataset random
BOUNDS = {
    "obs_l": (3.0,  6.0),
    "obs_w": (3.0,  6.0),
    "obs_h": (3.0,  5.0),
    "obs_x": (-12.0, -5.0),
    "obs_y": (-2.0,  6.0),
    "obs_r": (0.0, 180.0),
}

# Parametri GA — ridotti per rispettare budget di 100 simulazioni reali
POP_SIZE    = 20      # 20 individui per generazione
N_GEN       = 5       # 5 generazioni (1 iniziale + 4 evolutive)
TOTAL_SIMS  = POP_SIZE * N_GEN   # = 100 simulazioni reali

TOURN_SIZE  = 3       # torneo ridotto (con pop=20 un torneo di 5 sarebbe troppo selettivo)
CX_PROB     = 0.85
MUT_PROB    = 0.30    # mutazione più alta per compensare poche generazioni
MUT_SIGMA   = 0.20    # sigma maggiore per esplorare più aggressivamente
ELITE_N     = 2      # 2 élite (10% della popolazione)
TOP_K       = 30      # top-K candidati finali per validazione comparativa
COLLISION_THRESHOLD = 0.5

# Aerialist / Docker
TEMPLATE_FILE = "samples/tests/mission1.yaml"
DOCKER_IMAGE  = "skhatiri/aerialist:latest"
RESULTS_DIR   = Path("results/")
TIMEOUT_S     = 240

# Setup directory bounds
BOUNDS_LOW  = np.array([BOUNDS[f][0] for f in FEATURE_NAMES])
BOUNDS_HIGH = np.array([BOUNDS[f][1] for f in FEATURE_NAMES])
N_GENES     = len(FEATURE_NAMES)

# ════════════════════════════════════════════════════════════════════════════
# OPERATORI GENETICI (identici a algoritmo_genetico.py)
# ════════════════════════════════════════════════════════════════════════════

def random_individual(rng):
    return rng.uniform(BOUNDS_LOW, BOUNDS_HIGH)

def clip_individual(ind):
    return np.clip(ind, BOUNDS_LOW, BOUNDS_HIGH)

def tournament_selection(population, fitness, rng):
    idx = rng.integers(0, len(population), size=TOURN_SIZE)
    best = idx[np.argmin(fitness[idx])]
    return population[best].copy()

def crossover_blx(p1, p2, rng, alpha=0.3):
    d = np.abs(p1 - p2)
    lo = np.minimum(p1, p2) - alpha * d
    hi = np.maximum(p1, p2) + alpha * d
    c1 = rng.uniform(lo, hi)
    c2 = rng.uniform(lo, hi)
    return clip_individual(c1), clip_individual(c2)

def mutate_gaussian(ind, rng):
    mutant = ind.copy()
    sigma = (BOUNDS_HIGH - BOUNDS_LOW) * MUT_SIGMA
    gene_mut_prob = 1.0 / N_GENES
    mask = rng.random(N_GENES) < gene_mut_prob
    noise = rng.normal(0, sigma)
    mutant[mask] += noise[mask]
    return clip_individual(mutant)

# ════════════════════════════════════════════════════════════════════════════
# VALUTAZIONE SUL SIMULATORE REALE
# ════════════════════════════════════════════════════════════════════════════

def crea_yaml_individuo(individual, run_id, output_dir):
    """Crea il YAML di missione per l'individuo."""
    obs_params = {name: float(val) for name, val in zip(FEATURE_NAMES, individual)}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(TEMPLATE_FILE, "r") as f:
        missione = yaml.safe_load(f)

    missione["simulation"]["obstacles"] = [{
        "size": {
            "l": obs_params["obs_l"],
            "w": obs_params["obs_w"],
            "h": obs_params["obs_h"],
        },
        "position": {
            "x": obs_params["obs_x"],
            "y": obs_params["obs_y"],
            "z": 0.0,
            "r": obs_params["obs_r"],
        }
    }]
    if "wind" in missione.get("simulation", {}):
        del missione["simulation"]["wind"]

    output_path = os.path.join(output_dir, f"gen_{run_id}.yaml")
    with open(output_path, "w") as f:
        yaml.dump(missione, f, default_flow_style=False, sort_keys=False)
    return output_path


def simula_e_calcola_distanza(individual, run_id, output_dir):
    """
    Simula un individuo sul simulatore reale e ritorna la distanza minima.
    Usa la stessa logica Docker di genera_dataset_ostacolo.py.

    Restituisce: dict con dist, durata, errore
    """
    result = {"dist": None, "durata": None, "errore": None, "landato": None}

    # 1. Crea YAML
    yaml_path = crea_yaml_individuo(individual, run_id, output_dir)
    yaml_abs = str(Path(yaml_path).resolve())
    container_yaml = f"samples/tests/{Path(output_dir).name}/gen_{run_id}.yaml"
    results_abs = str(RESULTS_DIR.resolve())

    # 2. Conta ulg prima
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ulgs_prima = set(glob.glob(f"{RESULTS_DIR}/*.ulg"))

    # 3. Lancia simulazione Aerialist
    container_name = f"ga_pure_{run_id}"
    comando = [
        "docker", "run", "-it", "--rm",
        "--name", container_name,
        "-v", f"{yaml_abs}:/src/aerialist/{container_yaml}",
        "-v", f"{results_abs}:/src/aerialist/results",
        DOCKER_IMAGE,
        "python3", "aerialist", "exec",
        "--test", container_yaml,
        "--simulator", "ros",
        "--robot", "px4_ros",
        "--headless",
    ]

    t0 = time.time()
    try:
        subprocess.run(comando, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", container_name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result["errore"] = "sim_timeout"
        result["durata"] = round(time.time() - t0, 1)
        return result

    result["durata"] = round(time.time() - t0, 1)
    time.sleep(3)

    # 4. Trova nuovo .ulg
    ulgs_dopo = set(glob.glob(f"{RESULTS_DIR}/*.ulg"))
    nuovi = list(ulgs_dopo - ulgs_prima)
    if not nuovi:
        result["errore"] = "no_log_saved"
        return result

    ulg_path = max(nuovi, key=os.path.getmtime)

    # 5. Calcola distanza minima usando Aerialist nativo (container separato)
    ulg_rel = os.path.relpath(ulg_path, os.getcwd())
    json_out_rel = f"results/_ga_pure_{run_id}.json"
    obs_dict = {n: float(v) for n, v in zip(FEATURE_NAMES, individual)}

    inner_script = f'''
import json
try:
    from aerialist.px4.trajectory import Trajectory
    from aerialist.px4.obstacle import Obstacle

    size = Obstacle.Size(l={obs_dict["obs_l"]}, w={obs_dict["obs_w"]}, h={obs_dict["obs_h"]})
    pos  = Obstacle.Position(x={obs_dict["obs_x"]}, y={obs_dict["obs_y"]},
                              z=0.0, r={obs_dict["obs_r"]})
    obstacle = Obstacle(size, pos)
    traj = Trajectory.extract_from_log("/workspace/{ulg_rel}")
    dist = traj.min_distance_to_obstacles([obstacle])
    json.dump({{"dist": float(dist), "err": None}},
              open("/workspace/{json_out_rel}", "w"))
except Exception as e:
    json.dump({{"dist": None, "err": str(e)}},
              open("/workspace/{json_out_rel}", "w"))
'''
    calc_container = f"calc_ga_pure_{run_id}"
    cmd = [
        "docker", "run", "-i", "--rm",
        "--name", calc_container,
        "-e", "PYTHONPATH=/src/aerialist",
        "-v", f"{os.getcwd()}:/workspace",
        DOCKER_IMAGE,
        "python3", "-c", inner_script,
    ]
    try:
        subprocess.run(cmd, timeout=120,
                       stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", calc_container],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result["errore"] = "calc_timeout"
        return result

    json_path = Path(os.getcwd()) / json_out_rel
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            if data.get("dist") is not None:
                result["dist"] = round(float(data["dist"]), 3)
            else:
                result["errore"] = f"aerialist:{data.get('err')}"
        except Exception as e:
            result["errore"] = f"json_parse:{e}"

        try:
            json_path.unlink()
        except PermissionError:
            subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{os.getcwd()}:/workspace",
                 DOCKER_IMAGE, "rm", "-f", f"/workspace/{json_out_rel}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            )
        except Exception:
            pass
    else:
        result["errore"] = "no_json_output"

    # Landed state
    try:
        log = ULog(ulg_path)
        land = log.get_dataset("vehicle_land_detected").data
        result["landato"] = bool(land["landed"][-1])
    except Exception:
        result["landato"] = None

    return result

# ════════════════════════════════════════════════════════════════════════════
# CSV PROGRESSIVO
# ════════════════════════════════════════════════════════════════════════════

CSV_COLUMNS = [
    "sim_idx", "generation", "individual_idx",
    "obs_l", "obs_w", "obs_h", "obs_x", "obs_y", "obs_r",
    "distanza_minima_m", "landato", "durata_s", "errore",
]

def init_csv(csv_path):
    if not Path(csv_path).exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False)

def append_csv(csv_path, row):
    df_row = pd.DataFrame([{c: row.get(c) for c in CSV_COLUMNS}])
    df_row.to_csv(csv_path, mode="a", header=False, index=False)

# ════════════════════════════════════════════════════════════════════════════
# GA PRINCIPALE
# ════════════════════════════════════════════════════════════════════════════

def run_ga_puro(seed, output_dir):
    """
    Esegue il GA puro sul simulatore reale.
    Salva incrementalmente nel CSV (resilienza a crash).
    """
    rng = np.random.default_rng(seed)
    yaml_dir = output_dir / "yaml"
    csv_path = output_dir / "tutte_le_simulazioni.csv"
    init_csv(csv_path)

    sim_counter = 0
    all_individuals = []
    all_fitness = []
    history = []

    # ── Generazione 0: popolazione iniziale random ─────────────────────────
    print(f"\n{'='*60}")
    print(f"🧬 GENERAZIONE 0 — Popolazione iniziale random")
    print(f"{'='*60}")

    population = np.array([random_individual(rng) for _ in range(POP_SIZE)])
    fitness = np.zeros(POP_SIZE)

    for i, ind in enumerate(population):
        sim_counter += 1
        run_id = f"g0_i{i:02d}_sim{sim_counter:03d}"

        print(f"\n📦 Sim {sim_counter}/{TOTAL_SIMS}  |  Gen 0  Ind {i}")
        print(f"   l={ind[0]:.2f} w={ind[1]:.2f} h={ind[2]:.2f} "
              f"x={ind[3]:.2f} y={ind[4]:.2f} r={ind[5]:.2f}")

        res = simula_e_calcola_distanza(ind, run_id, str(yaml_dir))
        dist = res["dist"] if res["dist"] is not None else 999.0
        fitness[i] = dist

        print(f"   → dist={res['dist']}  durata={res['durata']}s  errore={res['errore']}")

        append_csv(csv_path, {
            "sim_idx": sim_counter, "generation": 0, "individual_idx": i,
            **{f: float(v) for f, v in zip(FEATURE_NAMES, ind)},
            "distanza_minima_m": res["dist"],
            "landato": res["landato"],
            "durata_s": res["durata"],
            "errore": res["errore"],
        })

        all_individuals.append(ind.copy())
        all_fitness.append(dist)

    history.append({
        "gen": 0,
        "best": float(np.min(fitness)),
        "mean": float(np.mean(fitness)),
        "min_below_threshold": int((fitness <= COLLISION_THRESHOLD).sum()),
    })

    # ── Generazioni evolutive ──────────────────────────────────────────────
    for gen in range(1, N_GEN):
        print(f"\n{'='*60}")
        print(f"🧬 GENERAZIONE {gen}")
        print(f"   best={history[-1]['best']:.3f}m  "
              f"crit={history[-1]['min_below_threshold']}/{POP_SIZE}")
        print(f"{'='*60}")

        # Elitismo
        elite_idx = np.argsort(fitness)[:ELITE_N]
        elites = population[elite_idx].copy()
        elite_fitness = fitness[elite_idx].copy()

        # Genera nuovi figli
        new_pop = list(elites)
        new_fitness = list(elite_fitness)

        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(population, fitness, rng)
            p2 = tournament_selection(population, fitness, rng)

            if rng.random() < CX_PROB:
                c1, c2 = crossover_blx(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if rng.random() < MUT_PROB:
                c1 = mutate_gaussian(c1, rng)
            if rng.random() < MUT_PROB:
                c2 = mutate_gaussian(c2, rng)

            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)
                new_fitness.append(None)
            new_fitness.append(None)

        new_pop = np.array(new_pop[:POP_SIZE])

        # Valuta i NUOVI individui (skip élite già valutate)
        for i in range(ELITE_N, POP_SIZE):
            sim_counter += 1
            run_id = f"g{gen}_i{i:02d}_sim{sim_counter:03d}"
            ind = new_pop[i]

            print(f"\n📦 Sim {sim_counter}/{TOTAL_SIMS}  |  Gen {gen}  Ind {i}")
            print(f"   l={ind[0]:.2f} w={ind[1]:.2f} h={ind[2]:.2f} "
                  f"x={ind[3]:.2f} y={ind[4]:.2f} r={ind[5]:.2f}")

            res = simula_e_calcola_distanza(ind, run_id, str(yaml_dir))
            dist = res["dist"] if res["dist"] is not None else 999.0
            new_fitness[i] = dist

            print(f"   → dist={res['dist']}  durata={res['durata']}s  errore={res['errore']}")

            append_csv(csv_path, {
                "sim_idx": sim_counter, "generation": gen, "individual_idx": i,
                **{f: float(v) for f, v in zip(FEATURE_NAMES, ind)},
                "distanza_minima_m": res["dist"],
                "landato": res["landato"],
                "durata_s": res["durata"],
                "errore": res["errore"],
            })

            all_individuals.append(ind.copy())
            all_fitness.append(dist)

        population = new_pop
        fitness = np.array(new_fitness, dtype=float)

        history.append({
            "gen": gen,
            "best": float(np.min(fitness)),
            "mean": float(np.mean(fitness)),
            "min_below_threshold": int((fitness <= COLLISION_THRESHOLD).sum()),
        })

    return np.array(all_individuals), np.array(all_fitness), history, sim_counter

# ════════════════════════════════════════════════════════════════════════════
# OUTPUT FINALE
# ════════════════════════════════════════════════════════════════════════════

def save_results(individuals, fitness, history, sim_counter, output_dir, seed):
    """Salva i top-K candidati, il grafico di convergenza e il riepilogo."""

    # Top-K per distanza minima (i più critici)
    order = np.argsort(fitness)
    top_k_idx = order[:TOP_K]

    rows = []
    for rank, idx in enumerate(top_k_idx, start=1):
        ind = individuals[idx]
        row = {f: round(float(v), 4) for f, v in zip(FEATURE_NAMES, ind)}
        row["rank"] = rank
        row["dist_reale_m"] = round(float(fitness[idx]), 4)
        row["obs_z"] = 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[["rank", "dist_reale_m"] + FEATURE_NAMES + ["obs_z"]]
    top_path = output_dir / f"top{TOP_K}_candidati.csv"
    df.to_csv(top_path, index=False)
    print(f"\n   ✅ Top-{TOP_K} candidati → {top_path}")
    print(df.head(10).to_string(index=False))

    # Grafico convergenza per generazione
    gens = [h["gen"] for h in history]
    bests = [h["best"] for h in history]
    means = [h["mean"] for h in history]
    crits = [h["min_below_threshold"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(gens, bests, "b-o", linewidth=2, label="Best fitness")
    ax1.plot(gens, means, "r--s", linewidth=1, alpha=0.7, label="Mean fitness")
    ax1.axhline(y=COLLISION_THRESHOLD, color="green", linestyle=":",
                label=f"Soglia critica ({COLLISION_THRESHOLD}m)")
    ax1.set_xlabel("Generazione")
    ax1.set_ylabel("Distanza minima reale (m)")
    ax1.set_title("Convergenza GA puro")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(gens, crits, color="orange", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Generazione")
    ax2.set_ylabel("# scenari critici (dist ≤ 0.5m)")
    ax2.set_title(f"Scenari critici per generazione (su {POP_SIZE} individui)")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "convergenza_ga_puro.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"   📈 Convergenza → {plot_path}")

    # Metadati
    n_crit_totali = int((fitness <= COLLISION_THRESHOLD).sum())
    hit_rate = n_crit_totali / len(fitness) if len(fitness) > 0 else 0
    meta = {
        "tecnica":          "GA_puro_simulatore_reale",
        "seed":             seed,
        "pop_size":         POP_SIZE,
        "n_gen":            N_GEN,
        "total_sims":       TOTAL_SIMS,
        "sims_eseguite":    sim_counter,
        "best_dist_reale":  round(float(fitness.min()), 4),
        "mean_dist_reale":  round(float(fitness.mean()), 4),
        "scenari_critici":  n_crit_totali,
        "hit_rate":         round(hit_rate, 4),
        "history":          history,
        "bounds":           {f: list(v) for f, v in BOUNDS.items()},
    }
    with open(output_dir / "ga_puro_summary.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   ⚙️  Summary → {output_dir}/ga_puro_summary.json")

    # Riepilogo testuale
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║   GA PURO SU SIMULATORE REALE — RIEPILOGO
╠══════════════════════════════════════════════════════════════╣
║  Budget:                  {TOTAL_SIMS} simulazioni reali
║  Eseguite effettivamente: {sim_counter}
║  Seed:                    {seed}
║  Pop × Gen:               {POP_SIZE} × {N_GEN}
╠══════════════════════════════════════════════════════════════╣
║  RISULTATI:
║    Best distanza:         {fitness.min():.3f} m
║    Mean distanza:         {fitness.mean():.3f} m
║    Scenari critici:       {n_crit_totali} / {len(fitness)}
║    Hit rate:              {hit_rate*100:.1f}%
╚══════════════════════════════════════════════════════════════╝
"""
    print(summary)
    with open(output_dir / "ga_puro_summary.txt", "w") as f:
        f.write(summary)

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GA puro su simulatore Aerialist reale (no surrogato)"
    )
    parser.add_argument("--output", required=True,
                        help="Cartella output (es: ga_puro_results/replica1/)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🧬 ALGORITMO GENETICO PURO — Simulatore reale")
    print("=" * 60)
    print(f"   Output:       {output_dir}")
    print(f"   Seed:         {args.seed}")
    print(f"   Pop × Gen:    {POP_SIZE} × {N_GEN} = {TOTAL_SIMS} sim reali")
    print(f"   Tempo stim.:  ~{TOTAL_SIMS * 2.5 / 60:.1f}h")
    print(f"   Bounds:       {BOUNDS}")

    t0 = time.time()
    individuals, fitness, history, sim_counter = run_ga_puro(args.seed, output_dir)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"🎉 GA PURO COMPLETATO")
    print(f"   Durata totale: {elapsed/3600:.1f}h")
    print(f"   Sim eseguite:  {sim_counter}")
    print(f"{'='*60}")

    save_results(individuals, fitness, history, sim_counter, output_dir, args.seed)


if __name__ == "__main__":
    main()