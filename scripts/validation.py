"""
valida_candidati.py
====================
Valida sul simulatore Aerialist reale i candidati trovati dal GA.

Per ogni candidato nel CSV:
  1. Crea un YAML di missione con i parametri dell'ostacolo
  2. Lancia Aerialist via Docker (stesso meccanismo di genera_dataset_ostacolo.py)
  3. Calcola la distanza minima reale con le funzioni native di Aerialist
  4. Confronta distanza predetta dal surrogato vs distanza reale misurata

Output:
  - <output_dir>/validazione_risultati.csv   (predetto vs reale per ogni candidato)
  - <output_dir>/validazione_summary.txt     (statistiche aggregate)

Uso:
    cd ~/Aerialist

    # Valida i candidati del Ramo A
    python3 valida_candidati.py \
        --input  ga_results/ramo_a/candidati_top30.csv \
        --output ga_results/ramo_a/ \
        --ramo   A

    # Valida i candidati del Ramo B
    python3 valida_candidati.py \
        --input  ga_results/ramo_b/candidati_top30.csv \
        --output ga_results/ramo_b/ \
        --ramo   B

Prerequisiti:
    - Docker con immagine skhatiri/aerialist:latest
    - File .env nella directory corrente (~/Aerialist/)
    - Lanciare dalla cartella ~/Aerialist/ (come gli altri script)
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
from pyulog import ULog

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ════════════════════════════════════════════════════════════════════════════

TEMPLATE_FILE = "samples/tests/mission1.yaml"
OUTPUT_DIR_YAML = "samples/tests/validation/"
RESULTS_DIR   = "results/"
DOCKER_IMAGE  = "skhatiri/aerialist:latest"
TIMEOUT_S     = 240    # 4 minuti per simulazione
COLLISION_THRESHOLD = 0.0   # soglia per contare una collisione

# ════════════════════════════════════════════════════════════════════════════
# CREAZIONE YAML
# ════════════════════════════════════════════════════════════════════════════

def crea_yaml(candidato: dict, run_id: str) -> str:
    """Crea un YAML di missione a partire dai parametri del candidato GA."""
    Path(OUTPUT_DIR_YAML).mkdir(parents=True, exist_ok=True)

    with open(TEMPLATE_FILE, "r") as f:
        missione = yaml.safe_load(f)

    missione["simulation"]["obstacles"] = [{
        "size": {
            "l": float(candidato["obs_l"]),
            "w": float(candidato["obs_w"]),
            "h": float(candidato["obs_h"]),
        },
        "position": {
            "x": float(candidato["obs_x"]),
            "y": float(candidato["obs_y"]),
            "z": float(candidato.get("obs_z", 0.0)),
            "r": float(candidato["obs_r"]),
        }
    }]

    if "wind" in missione.get("simulation", {}):
        del missione["simulation"]["wind"]

    output_path = os.path.join(OUTPUT_DIR_YAML, f"val_{run_id}.yaml")
    with open(output_path, "w") as f:
        yaml.dump(missione, f, default_flow_style=False, sort_keys=False)

    return output_path

# ════════════════════════════════════════════════════════════════════════════
# SIMULAZIONE
# ════════════════════════════════════════════════════════════════════════════

def avvia_simulazione(yaml_path: str, run_id: str) -> dict:
    """Lancia Aerialist via Docker. Stessa logica di genera_dataset_ostacolo.py."""
    yaml_abs      = str(Path(yaml_path).resolve())
    results_abs   = str(Path(RESULTS_DIR).resolve())
    container_yaml = f"samples/tests/validation/val_{run_id}.yaml"
    container_name = f"val_{run_id}"

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
    timed_out = False
    success   = False

    try:
        proc = subprocess.run(comando, timeout=TIMEOUT_S)
        success = (proc.returncode == 0)
    except subprocess.TimeoutExpired:
        timed_out = True
        subprocess.run(["docker", "rm", "-f", container_name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    durata = round(time.time() - t0, 1)
    return {"success": success, "timed_out": timed_out, "durata_s": durata}

# ════════════════════════════════════════════════════════════════════════════
# CALCOLO DISTANZA REALE
# ════════════════════════════════════════════════════════════════════════════

def calcola_distanza_reale(ulg_path: str, candidato: dict,
                            run_id: str) -> dict:
    """
    Calcola la distanza minima reale dal .ulg usando Aerialist nativo
    nel container. Stesso meccanismo di genera_dataset_ostacolo.py.
    """
    result = {"distanza_reale_m": None, "landato": None, "errore": None}

    if not Path(ulg_path).exists():
        result["errore"] = "ulg_non_trovato"
        return result

    ulg_rel      = os.path.relpath(ulg_path, os.getcwd())
    json_out_rel = f"results/_val_{run_id}.json"

    inner_script = f'''
import json
try:
    from aerialist.px4.trajectory import Trajectory
    from aerialist.px4.obstacle import Obstacle

    size = Obstacle.Size(
        l={candidato["obs_l"]}, w={candidato["obs_w"]}, h={candidato["obs_h"]}
    )
    pos = Obstacle.Position(
        x={candidato["obs_x"]}, y={candidato["obs_y"]},
        z=0.0, r={candidato["obs_r"]}
    )
    obstacle = Obstacle(size, pos)
    traj = Trajectory.extract_from_log("/src/aerialist/{ulg_rel}")
    dist = traj.min_distance_to_obstacles([obstacle])
    json.dump({{"dist": float(dist), "err": None}},
              open("/src/aerialist/{json_out_rel}", "w"))
except Exception as e:
    json.dump({{"dist": None, "err": str(e)}},
              open("/src/aerialist/{json_out_rel}", "w"))
'''

    calc_container = f"calc_val_{run_id}"
    cmd = [
        "docker", "run", "-i", "--rm",
        "--name", calc_container,
        "-v", f"{os.getcwd()}:/src/aerialist",
        DOCKER_IMAGE,
        "python3", "-c", inner_script,
    ]

    try:
        subprocess.run(cmd, timeout=120, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
                result["distanza_reale_m"] = round(float(data["dist"]), 3)
            else:
                result["errore"] = f"aerialist:{data.get('err')}"
        except Exception as e:
            result["errore"] = f"json_parse:{e}"

        try:
            json_path.unlink()
        except PermissionError:
            subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{os.getcwd()}:/src/aerialist",
                 DOCKER_IMAGE, "rm", "-f", f"/src/aerialist/{json_out_rel}"],
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
# SUMMARY STATISTICHE
# ════════════════════════════════════════════════════════════════════════════

def genera_summary(df: pd.DataFrame, ramo: str, output_dir: Path) -> None:
    """Calcola e salva le statistiche aggregate della validazione."""

    validi = df.dropna(subset=["distanza_reale_m"])
    n_tot   = len(df)
    n_val   = len(validi)
    n_err   = n_tot - n_val

    if n_val == 0:
        print("   ⚠️  Nessun risultato valido, impossibile calcolare statistiche.")
        return

    # Metriche principali
    hit_rate = (validi["distanza_reale_m"] <= COLLISION_THRESHOLD).sum() / n_val
    dist_media  = validi["distanza_reale_m"].mean()
    dist_min    = validi["distanza_reale_m"].min()
    dist_max    = validi["distanza_reale_m"].max()

    # Accuratezza surrogato (solo dove predetto non è NaN)
    if "pred_dist_m" in validi.columns:
        errore_abs = (validi["pred_dist_m"] - validi["distanza_reale_m"]).abs()
        mae_surr   = errore_abs.mean()
        rmse_surr  = np.sqrt((errore_abs ** 2).mean())
    else:
        mae_surr = rmse_surr = float("nan")

    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║   VALIDAZIONE RAMO {ramo} — RISULTATI FINALI
╠══════════════════════════════════════════════════════════════╣
║  Candidati validati:      {n_val:3d} / {n_tot}
║  Errori simulazione:      {n_err:3d}
╠══════════════════════════════════════════════════════════════╣
║  SCENARI CRITICI (dist ≤ {COLLISION_THRESHOLD}m):
║    Hit rate reale:        {hit_rate*100:.1f}%  ({int(hit_rate*n_val)}/{n_val})
║
║  DISTANZA MINIMA REALE:
║    Media:                 {dist_media:.3f} m
║    Minima:                {dist_min:.3f} m
║    Massima:               {dist_max:.3f} m
╠══════════════════════════════════════════════════════════════╣
║  ACCURATEZZA SURROGATO (predetto vs reale):
║    MAE:                   {mae_surr:.3f} m
║    RMSE:                  {rmse_surr:.3f} m
╚══════════════════════════════════════════════════════════════╝
"""
    print(summary)

    out_path = output_dir / "validazione_summary.txt"
    with open(out_path, "w") as f:
        f.write(summary)
    print(f"   📄 Summary salvato → {out_path}")

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Valida candidati GA su simulatore Aerialist reale"
    )
    parser.add_argument("--input",  required=True,
                        help="CSV candidati dal GA (es: ga_results/ramo_a/candidati_top30.csv)")
    parser.add_argument("--output", required=True,
                        help="Cartella output (es: ga_results/ramo_a/)")
    parser.add_argument("--ramo",   default="?",
                        help="Etichetta del ramo (A o B) per i report")
    parser.add_argument("--skip-to", type=int, default=1,
                        help="Salta ai candidati a partire dal rank N (riprendere run interrotti)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    risultati_csv = output_dir / "validazione_risultati.csv"

    # Carica candidati
    df_candidati = pd.read_csv(args.input)
    n_candidati  = len(df_candidati)

    print("=" * 60)
    print(f"🔬 VALIDAZIONE CANDIDATI GA — RAMO {args.ramo}")
    print("=" * 60)
    print(f"   Input:       {args.input}")
    print(f"   Output:      {output_dir}")
    print(f"   Candidati:   {n_candidati}")
    print(f"   Timeout sim: {TIMEOUT_S}s")
    stima = n_candidati * 2.5 / 60
    print(f"   Stima tempo: ~{stima:.0f} minuti")

    # Inizializza CSV risultati
    cols_out = ["rank", "pred_dist_m",
                "obs_l", "obs_w", "obs_h", "obs_x", "obs_y", "obs_r", "obs_z",
                "distanza_reale_m", "landato",
                "collisione_reale", "errore_surrogato_m",
                "timeout", "durata_s", "errore"]

    if not risultati_csv.exists() or args.skip_to == 1:
        pd.DataFrame(columns=cols_out).to_csv(risultati_csv, index=False)

    # Loop sui candidati
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    n_ok = 0
    n_err = 0

    for _, row in df_candidati.iterrows():
        rank = int(row["rank"])
        if rank < args.skip_to:
            continue

        pred_dist = float(row["pred_dist_m"])
        run_id    = f"ramo{args.ramo}_rank{rank:03d}"

        print(f"\n{'─'*55}")
        print(f"📦 Rank {rank:2d}/{n_candidati}  |  pred={pred_dist:.4f}m")
        print(f"   l={row['obs_l']:.2f} w={row['obs_w']:.2f} h={row['obs_h']:.2f} "
              f"x={row['obs_x']:.2f} y={row['obs_y']:.2f} r={row['obs_r']:.2f}")

        # 1. Crea YAML
        yaml_path = crea_yaml(row.to_dict(), run_id)

        # 2. Conta ulg prima
        ulgs_prima = set(glob.glob(f"{RESULTS_DIR}*.ulg"))

        # 3. Simula
        run = avvia_simulazione(yaml_path, run_id)
        print(f"   Docker: success={run['success']} "
              f"timeout={run['timed_out']} durata={run['durata_s']}s")

        time.sleep(3)

        # 4. Trova nuovo ulg
        ulgs_dopo = set(glob.glob(f"{RESULTS_DIR}*.ulg"))
        nuovi_ulg = list(ulgs_dopo - ulgs_prima)

        if not nuovi_ulg:
            print(f"   🚨 Nessun log salvato.")
            riga = {c: None for c in cols_out}
            riga.update({
                "rank": rank, "pred_dist_m": pred_dist,
                "obs_l": row["obs_l"], "obs_w": row["obs_w"],
                "obs_h": row["obs_h"], "obs_x": row["obs_x"],
                "obs_y": row["obs_y"], "obs_r": row["obs_r"],
                "obs_z": row.get("obs_z", 0.0),
                "timeout": run["timed_out"],
                "durata_s": run["durata_s"],
                "errore": "no_log_saved",
            })
            pd.DataFrame([{c: riga.get(c) for c in cols_out}]).to_csv(
                risultati_csv, mode="a", header=False, index=False)
            n_err += 1
            continue

        ulg_path = max(nuovi_ulg, key=os.path.getmtime)
        print(f"   Log: {ulg_path}")

        # 5. Calcola distanza reale
        yaml_rel = os.path.relpath(yaml_path, os.getcwd())
        metriche = calcola_distanza_reale(ulg_path, row.to_dict(), run_id)

        dist_reale = metriche["distanza_reale_m"]
        landato    = metriche["landato"]
        collisione = (dist_reale is not None and
                      dist_reale <= COLLISION_THRESHOLD)
        err_surr   = (round(abs(pred_dist - dist_reale), 3)
                      if dist_reale is not None else None)

        print(f"   ✅ dist_reale={dist_reale}m  pred={pred_dist}m  "
              f"err={err_surr}m  collisione={collisione}")

        # 6. Salva riga
        riga = {
            "rank":             rank,
            "pred_dist_m":      pred_dist,
            "obs_l":            row["obs_l"],
            "obs_w":            row["obs_w"],
            "obs_h":            row["obs_h"],
            "obs_x":            row["obs_x"],
            "obs_y":            row["obs_y"],
            "obs_r":            row["obs_r"],
            "obs_z":            row.get("obs_z", 0.0),
            "distanza_reale_m": dist_reale,
            "landato":          landato,
            "collisione_reale": collisione,
            "errore_surrogato_m": err_surr,
            "timeout":          run["timed_out"],
            "durata_s":         run["durata_s"],
            "errore":           metriche["errore"],
        }
        pd.DataFrame([{c: riga.get(c) for c in cols_out}]).to_csv(
            risultati_csv, mode="a", header=False, index=False)

        if dist_reale is not None:
            n_ok += 1
        else:
            n_err += 1

    # Summary finale
    print(f"\n{'='*60}")
    print(f"🎉 VALIDAZIONE COMPLETATA — RAMO {args.ramo}")
    print(f"   Validi: {n_ok}  |  Errori: {n_err}")
    print(f"{'='*60}")

    df_ris = pd.read_csv(risultati_csv)
    genera_summary(df_ris, args.ramo, output_dir)


if __name__ == "__main__":
    main()