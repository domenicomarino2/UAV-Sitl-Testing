"""
algoritmo_genetico.py
======================
Algoritmo Genetico (GA) che usa un modello surrogato per trovare
le configurazioni di ostacolo più critiche (distanza minima → 0).

Il GA è CONDIVISO tra Ramo A e Ramo B: cambia solo il modello surrogato
caricato. Questo garantisce che il confronto misuri la qualità del dataset
di training, non differenze algoritmiche.

Pipeline completa:
  1. Carica il surrogato (pkl) — GB del Ramo A o del Ramo B
  2. Esegue il GA usando il surrogato come funzione di fitness (costo ~0)
  3. Salva i top-K candidati in un CSV per la validazione sul simulatore reale
  4. Produce il grafico della convergenza

Uso:
    # Ramo A
    python3 algoritmo_genetico.py --surrogate models/aerialist/gradientboosting.pkl \
                                   --output ga_results/ramo_a/

    # Ramo B
    python3 algoritmo_genetico.py --surrogate models/surrealist/gradientboosting.pkl \
                                   --output ga_results/ramo_b/
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ════════════════════════════════════════════════════════════════════════════

# Feature nell'ordine esatto usato durante il training
FEATURE_NAMES = ["obs_l", "obs_w", "obs_h", "obs_x", "obs_y", "obs_r"]

# Bounds dello spazio di ricerca — IDENTICI per entrambi i rami
# (basati sullo spazio del Ramo A per non favorire nessuno)
BOUNDS = {
    "obs_l": (3.0, 6.0),   # da OBSTACLE_SPACE del generatore dataset
    "obs_w": (3.0, 6.0),
    "obs_h": (3.0, 5.0),   
    "obs_x": (-12.0, -5.0),  
    "obs_y": (-2.0, 6.0),  
    "obs_r": (0.0, 180.0),
}

# Parametri GA
POP_SIZE    = 200    # individui per generazione
N_GEN       = 300    # numero generazioni
TOURN_SIZE  = 5      # dimensione torneo per la selezione
CX_PROB     = 0.85   # probabilità crossover
MUT_PROB    = 0.15   # probabilità mutazione per individuo
MUT_SIGMA   = 0.15   # deviazione standard relativa della mutazione gaussiana
ELITE_N     = 5      # numero di élite da preservare ogni generazione
TOP_K       = 50     # candidati da restituire per la validazione reale
RANDOM_SEED = 456

# ════════════════════════════════════════════════════════════════════════════
# RAPPRESENTAZIONE INDIVIDUO
# ════════════════════════════════════════════════════════════════════════════

BOUNDS_LOW  = np.array([BOUNDS[f][0] for f in FEATURE_NAMES])
BOUNDS_HIGH = np.array([BOUNDS[f][1] for f in FEATURE_NAMES])
N_GENES     = len(FEATURE_NAMES)


def random_individual(rng: np.random.Generator) -> np.ndarray:
    """Individuo casuale uniforme nello spazio di ricerca."""
    return rng.uniform(BOUNDS_LOW, BOUNDS_HIGH)


def clip(individual: np.ndarray) -> np.ndarray:
    """Porta l'individuo dentro i bounds dopo mutazione."""
    return np.clip(individual, BOUNDS_LOW, BOUNDS_HIGH)

# ════════════════════════════════════════════════════════════════════════════
# FITNESS
# ════════════════════════════════════════════════════════════════════════════

def evaluate_population(population: np.ndarray, surrogate) -> np.ndarray:
    """
    Valuta tutta la popolazione in un unico batch sul surrogato.
    Il GA MINIMIZZA la distanza → scenario più critico = fitness più bassa.
    Clamp a 0: distanze negative sono fisicamente impossibili e indicano
    extrapolazione del modello fuori dal dominio di training.
    """
    preds = surrogate.predict(population)
    return np.clip(preds, 0.0, None)

# ════════════════════════════════════════════════════════════════════════════
# OPERATORI GENETICI
# ════════════════════════════════════════════════════════════════════════════

def tournament_selection(population: np.ndarray,
                          fitness: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Selezione per torneo: sceglie TOURN_SIZE individui a caso,
    restituisce il migliore (minore fitness = più critico).
    """
    idx = rng.integers(0, len(population), size=TOURN_SIZE)
    best = idx[np.argmin(fitness[idx])]
    return population[best].copy()


def crossover_blx(parent1: np.ndarray,
                   parent2: np.ndarray,
                   rng: np.random.Generator,
                   alpha: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """
    BLX-alpha crossover: campiona figli da un intervallo esteso
    rispetto all'intervallo dei genitori. Più esplorativi di SBX
    su spazi continui piccoli.
    """
    d = np.abs(parent1 - parent2)
    lo = np.minimum(parent1, parent2) - alpha * d
    hi = np.maximum(parent1, parent2) + alpha * d
    child1 = rng.uniform(lo, hi)
    child2 = rng.uniform(lo, hi)
    return clip(child1), clip(child2)


def mutate_gaussian(individual: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Mutazione gaussiana adattiva: il sigma è proporzionale all'ampiezza
    del bound di ciascuna feature, moltiplicato per MUT_SIGMA.
    Ogni gene muta con probabilità 1/N_GENES (una feature per volta
    in media), mantenendo alta diversità senza distruggere tutto.
    """
    mutant = individual.copy()
    sigma = (BOUNDS_HIGH - BOUNDS_LOW) * MUT_SIGMA
    gene_mut_prob = 1.0 / N_GENES     # muta in media 1 gene per individuo
    mask = rng.random(N_GENES) < gene_mut_prob
    noise = rng.normal(0, sigma)
    mutant[mask] += noise[mask]
    return clip(mutant)

# ════════════════════════════════════════════════════════════════════════════
# ALGORITMO GENETICO PRINCIPALE
# ════════════════════════════════════════════════════════════════════════════

def run_ga(surrogate, seed: int = RANDOM_SEED) -> tuple[np.ndarray, list[float]]:
    """
    Esegue il GA e restituisce:
      - population finale ordinata per fitness crescente
      - storia del best fitness per generazione (per il plot di convergenza)
    """
    rng = np.random.default_rng(seed)

    # Inizializzazione
    population = np.array([random_individual(rng) for _ in range(POP_SIZE)])
    fitness    = evaluate_population(population, surrogate)

    best_history = [float(np.min(fitness))]
    mean_history = [float(np.mean(fitness))]

    print(f"   Gen 000 | best={best_history[0]:.4f}  "
          f"mean={mean_history[0]:.4f}  "
          f"pop={POP_SIZE}")

    for gen in range(1, N_GEN + 1):
        # ── Elitismo: preserva i migliori ELITE_N ──────────────────────────
        elite_idx = np.argsort(fitness)[:ELITE_N]
        elites    = population[elite_idx].copy()

        # ── Genera nuova popolazione via selezione + crossover + mutazione ──
        new_pop = list(elites)  # élite passano direttamente

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

        population = np.array(new_pop[:POP_SIZE])
        fitness    = evaluate_population(population, surrogate)

        best_history.append(float(np.min(fitness)))
        mean_history.append(float(np.mean(fitness)))

        if gen % 50 == 0 or gen == N_GEN:
            print(f"   Gen {gen:03d} | best={best_history[-1]:.4f}  "
                  f"mean={mean_history[-1]:.4f}")

    # Ordina per fitness crescente e restituisce
    order      = np.argsort(fitness)
    population = population[order]
    fitness    = fitness[order]

    return population, fitness, best_history, mean_history

# ════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def save_candidates(population: np.ndarray,
                     fitness: np.ndarray,
                     output_dir: Path,
                     top_k: int = TOP_K) -> pd.DataFrame:
    """
    Salva i top-K candidati in un CSV pronto per la validazione sul simulatore.

    Usa una selezione diversificata (greedy):
    - prende il migliore in assoluto
    - poi aggiunge sempre il prossimo candidato che è più lontano
      da tutti quelli già selezionati (distanza euclidea normalizzata)
    Questo evita che tutti i candidati siano copie quasi identiche
    dello stesso minimo locale.
    """
    # Normalizza feature per il calcolo della distanza (0-1 per range)
    ranges = BOUNDS_HIGH - BOUNDS_LOW
    pop_norm = (population - BOUNDS_LOW) / ranges

    selected_idx = [0]   # inizia dal migliore in assoluto

    while len(selected_idx) < top_k and len(selected_idx) < len(population):
        # Distanza minima di ogni candidato dai già selezionati
        min_dists = []
        for i in range(len(population)):
            if i in selected_idx:
                min_dists.append(-1)
                continue
            dists = [
                np.linalg.norm(pop_norm[i] - pop_norm[j])
                for j in selected_idx
            ]
            min_dists.append(min(dists))

        # Aggiunge il candidato più lontano dai selezionati
        next_idx = int(np.argmax(min_dists))
        selected_idx.append(next_idx)

    rows = []
    for rank, idx in enumerate(selected_idx, start=1):
        ind       = population[idx]
        pred_dist = fitness[idx]
        row = {f: round(float(v), 4) for f, v in zip(FEATURE_NAMES, ind)}
        row["rank"]        = rank
        row["pred_dist_m"] = round(float(pred_dist), 4)
        row["obs_z"]       = 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["rank", "pred_dist_m"] + FEATURE_NAMES + ["obs_z"]
    df = df[cols]

    out_path = output_dir / f"candidati_top{top_k}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n   ✅ Top-{top_k} candidati (diversificati) → {out_path}")
    print(df.head(10).to_string(index=False))
    return df


def save_convergence_plot(best_history: list[float],
                           mean_history: list[float],
                           output_dir: Path,
                           label: str = "") -> None:
    """
    Salva il grafico di convergenza del GA (best e mean per generazione).
    """
    plt.figure(figsize=(10, 5))
    gens = list(range(len(best_history)))

    plt.plot(gens, best_history, "b-",  linewidth=2, label="Best fitness")
    plt.plot(gens, mean_history, "r--", linewidth=1, alpha=0.7,
             label="Mean fitness")

    plt.xlabel("Generazione")
    plt.ylabel("Distanza minima predetta (m)")
    plt.title(f"Convergenza GA — {label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / "convergenza_ga.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   📈 Grafico convergenza → {out_path}")


def save_metadata(best_history: list[float],
                   elapsed_s: float,
                   surrogate_path: str,
                   output_dir: Path) -> None:
    meta = {
        "surrogate":         surrogate_path,
        "pop_size":          POP_SIZE,
        "n_gen":             N_GEN,
        "top_k":             TOP_K,
        "best_pred_dist_m":  round(best_history[-1], 4),
        "worst_pred_dist_m": round(best_history[0],  4),
        "elapsed_s":         round(elapsed_s, 1),
        "random_seed":       RANDOM_SEED,
        "bounds":            {f: list(v) for f, v in BOUNDS.items()},
    }
    out_path = output_dir / "ga_config.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   ⚙️  Config salvata → {out_path}")

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GA surrogate-guided per scenari UAV critici"
    )
    parser.add_argument(
        "--surrogate", required=True,
        help="Path al modello surrogato (.pkl), es: models/aerialist/gradientboosting.pkl"
    )
    parser.add_argument(
        "--output", required=True,
        help="Cartella dove salvare i risultati, es: ga_results/ramo_a/"
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Numero di candidati da restituire (default: {TOP_K})"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Label per i plot (estratta dal path del surrogato)
    label = Path(args.surrogate).parent.parent.name.upper()

    print("=" * 60)
    print(f"🧬 ALGORITMO GENETICO — Surrogate-Guided UAV Testing")
    print("=" * 60)
    print(f"   Surrogato:  {args.surrogate}")
    print(f"   Output:     {output_dir}")
    print(f"   Popolazione: {POP_SIZE}  |  Generazioni: {N_GEN}")
    print(f"   Top-K candidati: {args.top_k}")

    # Carica surrogato
    print(f"\n📦 Caricamento surrogato...")
    surrogate = joblib.load(args.surrogate)
    print(f"   ✓ Caricato: {type(surrogate).__name__}")

    # Esegui GA
    print(f"\n🔄 Avvio GA (seed={RANDOM_SEED})...")
    t0 = time.time()
    population, fitness, best_history, mean_history = run_ga(surrogate)
    elapsed = time.time() - t0
    print(f"\n   ⏱  Completato in {elapsed:.1f}s")
    print(f"   🏆 Miglior distanza predetta: {best_history[-1]:.4f}m")

    # Salva output
    print(f"\n💾 Salvataggio risultati...")
    save_candidates(population, fitness, output_dir, top_k=args.top_k)
    save_convergence_plot(best_history, mean_history, output_dir, label=label)
    save_metadata(best_history, elapsed, args.surrogate, output_dir)

    print(f"\n{'='*60}")
    print(f"🎉 GA COMPLETATO")
    print(f"   Risultati in: {output_dir}")
    print(f"   Prossimo step: validare i candidati su Aerialist con:")
    print(f"   → python3 valida_candidati.py --input {output_dir}/candidati_top{args.top_k}.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()