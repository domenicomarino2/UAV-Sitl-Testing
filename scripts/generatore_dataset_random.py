"""
genera_dataset_ostacolo.py
===========================
Genera N missioni UAV variando solo la posizione e la forma dell'ostacolo.
Per ogni missione:
  1. Crea un YAML a partire dal template base (vento commentato)
  2. Lancia Aerialist via Docker (timeout 4 min)
  3. Legge il log .ulg e calcola distanza minima drone-ostacolo
  4. Aggiorna il CSV subito dopo ogni missione

INPUT  (7 colonne): l, w, h, x, y, z (fisso=0), r
OUTPUT (2 colonne): distanza_minima_m, landato

Uso:
    python genera_dataset_ostacolo.py

Prerequisiti:
    pip install pyulog pyyaml pandas numpy
    Docker con immagine skhatiri/aerialist:latest disponibile
    File .env nella directory corrente (come nel setup Aerialist)
"""

import os
import glob
import time
import random
import subprocess
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from pyulog import ULog

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ════════════════════════════════════════════════════════════════════════════

TEMPLATE_FILE = "samples/tests/mission1.yaml"   # YAML base (vento commentato)
OUTPUT_DIR    = "samples/tests/generated_pipeline_r3_50/"         # dove salvare i YAML generati
RESULTS_DIR   = "results/"                         # dove Aerialist scrive i .ulg
DATASET_CSV   = "dataset_pipeline_r3_50.csv"             # CSV aggiornato a ogni run

N_MISSIONI  =  50     # numero totale di scenari da generare
TIMEOUT_S   = 240      # timeout per simulazione (4 minuti)
RANDOM_SEED = 456       # riproducibilità: pipeline e random (42,123,456)

# Soglia collisione: distanza minima sotto cui consideriamo crash
COLLISION_THRESHOLD = 0.0   # metri

# Raggio drone (per il calcolo della distanza bordo-drone)
RAGGIO_DRONE = 0.12  # metri

# ── Spazio di ricerca dell'ostacolo ─────────────────────────────────────────
OBSTACLE_SPACE = {
    "l": (3.0, 6.0),      # max 6m: ostacolo realistico 
    "w": (3.0, 6.0),      # idem
    "h": (3.0, 5.0),      # drone vola a 3m → oltre 5m non aggiunge informazione
    "x": (-12.0, -5.0),   # zona centrale: lontano da decollo (0) e destinazione (-15)
    "y": (-2.0, 6.0),     # zona laterale ragionevole rispetto alla traiettoria
    "r": (0.0, 180.0),    
}

# ════════════════════════════════════════════════════════════════════════════
# COLONNE DEL CSV
# ════════════════════════════════════════════════════════════════════════════

CSV_COLUMNS = [
    "test_id",
    # 7 input ostacolo
    "obs_l", "obs_w", "obs_h",
    "obs_x", "obs_y", "obs_z", "obs_r",
    # 2 output
    "distanza_minima_m",
    "landato",
    # metadati
    "timeout", "durata_s", "errore",
]

# ════════════════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════════════════

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
random.seed(RANDOM_SEED)

# Inizializza CSV con header se non esiste
if not Path(DATASET_CSV).exists():
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(DATASET_CSV, index=False)
    print(f"📄 CSV inizializzato: {DATASET_CSV}")
else:
    print(f"📄 CSV esistente trovato, aggiunta in append: {DATASET_CSV}")

# ════════════════════════════════════════════════════════════════════════════
# GENERAZIONE PARAMETRI
# ════════════════════════════════════════════════════════════════════════════

def genera_parametri() -> dict:
    """
    Campiona casualmente i parametri dell'ostacolo nello spazio di ricerca.
    z è sempre 0 (ostacolo a terra).
    """
    return {
        "l": round(random.uniform(*OBSTACLE_SPACE["l"]), 2),
        "w": round(random.uniform(*OBSTACLE_SPACE["w"]), 2),
        "h": round(random.uniform(*OBSTACLE_SPACE["h"]), 2),
        "x": round(random.uniform(*OBSTACLE_SPACE["x"]), 2),
        "y": round(random.uniform(*OBSTACLE_SPACE["y"]), 2),
        "z": 0.0,
        "r": round(random.uniform(*OBSTACLE_SPACE["r"]), 1),
    }

# ════════════════════════════════════════════════════════════════════════════
# CREAZIONE YAML
# ════════════════════════════════════════════════════════════════════════════

def crea_yaml(test_id: int, params: dict) -> str:
    """
    Carica il template YAML e sovrascrive solo i parametri dell'ostacolo.
    Il vento rimane commentato (assente) come nel template base.
    Restituisce il path del file YAML generato.
    """
    with open(TEMPLATE_FILE, "r") as f:
        missione = yaml.safe_load(f)

    # Sovrascrive size e position dell'ostacolo
    missione["simulation"]["obstacles"] = [{
        "size": {
            "l": params["l"],
            "w": params["w"],
            "h": params["h"],
        },
        "position": {
            "x": params["x"],
            "y": params["y"],
            "z": params["z"],
            "r": params["r"],
        }
    }]

    # Assicura che il vento NON sia presente (rimane commentato)
    if "wind" in missione.get("simulation", {}):
        del missione["simulation"]["wind"]

    output_path = os.path.join(OUTPUT_DIR, f"mission_{test_id:04d}.yaml")
    with open(output_path, "w") as f:
        yaml.dump(missione, f, default_flow_style=False, sort_keys=False)

    return output_path

# ════════════════════════════════════════════════════════════════════════════
# ESECUZIONE DOCKER
# ════════════════════════════════════════════════════════════════════════════

def avvia_simulazione(yaml_path: str) -> dict:
    """
    Lancia Aerialist via Docker per un singolo YAML.
    Gestisce timeout di 4 minuti.
    Restituisce dict con success, timed_out, durata_s.

    Strategia di mount (ricavata dal comando funzionante):
      - Solo il singolo YAML generato → path interno al container
      - La cartella results/ → dove Aerialist scrive i log .ulg
    Niente --env-file, niente -it (causa problemi con subprocess).
    """
    # Path assoluto del YAML sull'host
    yaml_abs = str(Path(yaml_path).resolve())

    # Path interno al container: stessa struttura relativa sotto /src/aerialist/
    percorso_relativo = os.path.relpath(yaml_path, os.getcwd())
    container_yaml    = f"samples/tests/generated/{Path(yaml_path).name}"

    # Path assoluto della cartella results sull'host
    results_abs = str(Path(RESULTS_DIR).resolve())

    # Nome univoco per run → evita conflitti se container precedente non chiuso
    container_name = f"drone_{Path(yaml_path).stem}"

    # Replica ESATTA del comando che funziona manualmente:
    #   docker run -it --rm --name drone_test --env-file .env
    #              -v $(pwd):/src/aerialist skhatiri/aerialist:latest
    #              python3 aerialist exec --test /src/aerialist/... --headless
    #
    # Chiave: -v monta TUTTA la dir di Aerialist → tutti i file di supporto
    # (mission1.plan, mission1-params.csv, mission1-commands.csv) sono accessibili.
    # Il path del test è ASSOLUTO dentro il container (/src/aerialist/...).
    percorso_in_container = f"/src/aerialist/{os.path.relpath(yaml_path, os.getcwd())}"

    comando = [
        "docker", "run", "-it", "--rm",
        "--name", container_name,
        # Mount chirurgico — NON sovrascrive l'installazione Aerialist nel container
        "-v", f"{yaml_abs}:/src/aerialist/{container_yaml}",
        "-v", f"{results_abs}:/src/aerialist/results",
        "skhatiri/aerialist:latest",
        "python3", "aerialist", "exec",
        "--test", container_yaml,
        "--simulator", "ros",
        "--robot", "px4_ros",
        "--headless",   # ← era questo il motivo del hang, ora risolto
    ]


    t0 = time.time()
    timed_out = False
    success   = False

    try:
        proc = subprocess.run(comando, timeout=TIMEOUT_S)
        success = (proc.returncode == 0)
    except subprocess.TimeoutExpired:
        timed_out = True
        print("⚠️  TIMEOUT — forzo chiusura container...")
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("✅ Container eliminato.")

    durata = round(time.time() - t0, 1)
    return {"success": success, "timed_out": timed_out, "durata_s": durata}

# ════════════════════════════════════════════════════════════════════════════
# PARSING LOG .ulg
# ════════════════════════════════════════════════════════════════════════════

def calcola_distanza_minima(ulg_path: str, yaml_container_path: str,
                            container_name: str) -> dict:
    """
    Calcola la distanza minima usando le funzioni NATIVE di Aerialist
    dentro lo stesso container — stesso identico valore mostrato nel PNG.

    Strategia (come fa Surrealist):
      1. Lancia un secondo container che monta lo stesso volume
      2. Dentro, usa aerialist.px4.trajectory.Trajectory + Obstacle
      3. Scrive il risultato in un JSON che leggiamo dall'host

    Restituisce: distanza_minima_m, landato, errore
    """
    result = {"distanza_minima_m": None, "landato": None, "errore": None}

    if not Path(ulg_path).exists():
        result["errore"] = "ulg_non_trovato"
        return result

    # Path del .ulg relativo alla cwd (per il mount nel container)
    ulg_rel = os.path.relpath(ulg_path, os.getcwd())
    json_out_rel = f"results/_dist_{Path(ulg_path).stem}.json"

    # Script Python eseguito DENTRO il container Aerialist
    inner_script = f'''
import json, yaml, sys
try:
    from aerialist.px4.trajectory import Trajectory
    from aerialist.px4.obstacle import Obstacle

    # Carica geometria ostacolo dal YAML della missione
    with open("/src/aerialist/{yaml_container_path}") as f:
        cfg = yaml.safe_load(f)
    obs = cfg["simulation"]["obstacles"][0]
    size = Obstacle.Size(
        l=float(obs["size"]["l"]), w=float(obs["size"]["w"]),
        h=float(obs["size"]["h"]),
    )
    pos = Obstacle.Position(
        x=float(obs["position"]["x"]), y=float(obs["position"]["y"]),
        z=float(obs["position"].get("z", 0)),
        r=float(obs["position"].get("r", 0)),
    )
    obstacle = Obstacle(size, pos)

    # Estrai traiettoria dal log e calcola distanza minima (= valore PNG)
    traj = Trajectory.extract_from_log("/src/aerialist/{ulg_rel}")
    dist = traj.min_distance_to_obstacles([obstacle])

    json.dump({{"dist": float(dist), "err": None}},
              open("/src/aerialist/{json_out_rel}", "w"))
except Exception as e:
    json.dump({{"dist": None, "err": str(e)}},
              open("/src/aerialist/{json_out_rel}", "w"))
'''

    # Lancia il container per il solo calcolo della distanza
    calc_container = f"calc_{Path(ulg_path).stem}"
    cmd = [
        "docker", "run", "-i", "--rm",
        "--name", calc_container,
        "-v", f"{os.getcwd()}:/src/aerialist",
        "skhatiri/aerialist:latest",
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

    # Leggi il JSON prodotto dal container
    json_path = Path(os.getcwd()) / json_out_rel
    if json_path.exists():
        import json as _json
        try:
            data = _json.loads(json_path.read_text())
            if data.get("dist") is not None:
                result["distanza_minima_m"] = round(float(data["dist"]), 3)
            else:
                result["errore"] = f"aerialist_calc:{data.get('err')}"
        except Exception as e:
            result["errore"] = f"json_parse:{e}"

        # Il file è di proprietà di root (creato dal container Docker).
        # Lo cancelliamo via Docker (che gira come root), non dall'host.
        try:
            json_path.unlink()
        except PermissionError:
            subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{os.getcwd()}:/src/aerialist",
                 "skhatiri/aerialist:latest",
                 "rm", "-f", f"/src/aerialist/{json_out_rel}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except Exception:
            pass  # non bloccante: il file resta ma non è un problema
    else:
        result["errore"] = "no_json_output"

    # ── Landed state dal .ulg (via pyulog, sull'host) ───────────────────
    try:
        log = ULog(ulg_path)
        land = log.get_dataset("vehicle_land_detected").data
        result["landato"] = bool(land["landed"][-1])
    except Exception:
        result["landato"] = None

    return result

# ════════════════════════════════════════════════════════════════════════════
# AGGIORNAMENTO CSV
# ════════════════════════════════════════════════════════════════════════════

def salva_riga(row: dict) -> None:
    """Appende una singola riga al CSV (aggiornamento incrementale)."""
    df_row = pd.DataFrame([{col: row.get(col) for col in CSV_COLUMNS}])
    df_row.to_csv(DATASET_CSV, mode="a", header=False, index=False)

# ════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPALE
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"🚁 UAV Dataset Generation — {N_MISSIONI} scenari (solo ostacolo)")
    print("=" * 60)

    n_ok         = 0
    n_timeout    = 0
    n_errore     = 0
    n_collisione = 0

    for i in range(1, N_MISSIONI + 1):
        print(f"\n{'─'*55}")
        print(f"📦 Missione {i}/{N_MISSIONI}")

        # 1. Genera parametri casuali
        params = genera_parametri()
        print(
            f"   Ostacolo: l={params['l']} w={params['w']} h={params['h']} "
            f"| x={params['x']} y={params['y']} z=0 r={params['r']}"
        )

        # 2. Crea YAML
        yaml_path = crea_yaml(i, params)
        print(f"   YAML: {yaml_path}")

        # 3. Conta ULG prima del run (per rilevare se il log è stato salvato)
        ulgs_prima = set(glob.glob(f"{RESULTS_DIR}*.ulg"))

        # 4. Avvia simulazione Docker
        run = avvia_simulazione(yaml_path)
        print(
            f"   Docker: success={run['success']} "
            f"timeout={run['timed_out']} durata={run['durata_s']}s"
        )

        # Pausa fisiologica per chiusura porte di rete
        time.sleep(3)

        # 5. Trova il nuovo ULG generato da questo run
        ulgs_dopo = set(glob.glob(f"{RESULTS_DIR}*.ulg"))
        nuovi_ulg = list(ulgs_dopo - ulgs_prima)

        if not nuovi_ulg:
            # Nessun log salvato → run fallito (timeout precoce o crash)
            print(f"   🚨 Nessun log salvato — run fallito.")
            print(f"   🗑️  Cancello YAML per mantenere cartelle allineate.")
            os.remove(yaml_path)

            salva_riga({
                "test_id": f"mission_{i:04d}",
                **{f"obs_{k}": v for k, v in params.items()},
                "distanza_minima_m": None,
                "landato":           None,
                "timeout":           run["timed_out"],
                "durata_s":          run["durata_s"],
                "errore":            "no_log_saved",
            })
            n_errore += 1
            continue

        # Prendi il log più recente tra i nuovi
        ulg_path = max(nuovi_ulg, key=os.path.getmtime)
        print(f"   Log: {ulg_path}")

        # 6. Calcola metriche usando Aerialist nativo (= valore PNG)
        yaml_in_container = os.path.relpath(yaml_path, os.getcwd())
        metriche = calcola_distanza_minima(
            ulg_path, yaml_in_container, container_name=f"drone_{Path(yaml_path).stem}"
        )

        dist = metriche["distanza_minima_m"]
        land = metriche["landato"]
        collision = (dist is not None and dist == COLLISION_THRESHOLD)

        print(
            f"   ✅ dist_min={dist}m  landato={land}  "
            f"collisione={collision}  errore={metriche['errore']}"
        )

        # 7. Salva riga nel CSV
        salva_riga({
            "test_id": f"mission_{i:04d}",
            **{f"obs_{k}": v for k, v in params.items()},
            "distanza_minima_m": dist,
            "landato":           land,
            "timeout":           run["timed_out"],
            "durata_s":          run["durata_s"],
            "errore":            metriche["errore"],
        })

        # Aggiorna contatori
        n_ok += 1
        if run["timed_out"]:
            n_timeout += 1
        if collision:
            n_collisione += 1

    # ── Riepilogo finale ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("🎉 GENERAZIONE COMPLETATA")
    print(f"   Totale missioni : {N_MISSIONI}")
    print(f"   Run con log     : {n_ok}")
    print(f"   Run falliti     : {n_errore}")
    print(f"   Timeout         : {n_timeout}")
    print(f"   Collisioni      : {n_collisione} ({100*n_collisione/max(n_ok,1):.1f}%)")
    print(f"   Dataset salvato : {DATASET_CSV}")
    print("=" * 60)

    # Mostra anteprima dataset finale
    df = pd.read_csv(DATASET_CSV)
    print(f"\n📊 Anteprima dataset ({len(df)} righe):")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()