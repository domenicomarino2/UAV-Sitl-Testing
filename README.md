# UAV-Sitl-Tesing-on-Aerialist

**Progetto finale del corso di AI System Engineering**


## 📋 Indice

1. [Descrizione del progetto](#1-descrizione-del-progetto)
2. [Architettura della soluzione](#2-architettura-della-soluzione)
3. [Requisiti di sistema](#3-requisiti-di-sistema)
4. [Struttura del repository](#5-struttura-del-repository)
5. [Pipeline completa — comandi passo-passo](#6-pipeline-completa--comandi-passo-passo)
6. [Riferimenti](#9-riferimenti)

---

## 1. Descrizione del progetto

Questo progetto applica tecniche di **Search-Based Software Testing** al sistema di obstacle
avoidance **PX4-Avoidance** per UAV, utilizzando il simulatore **Aerialist** in ambiente Docker.

L'obiettivo è confrontare tre strategie di generazione di scenari di test critici
(collisioni drone-ostacolo) a **parità di budget di simulazioni reali (100 sim/replica)**:

| Strategia | Descrizione | Sim reali |
|-----------|-------------|-----------|
| **Random Search** | Campionamento uniforme dello spazio di ricerca | 100 |
| **GA Puro (Micro-GA)** | Algoritmo Genetico che valuta direttamente sul simulatore | 92 |
| **GA + Surrogato** | GA che ottimizza su un modello surrogato, poi valida sul simulatore | 50 train + 50 val = 100 |

Lo studio è strutturato come **Design of Experiments accoppiato** con 3 repliche
indipendenti (seed 42, 123, 456) per quantificare la variabilità inter-replica.

### Cromosoma dell'ostacolo (6 parametri reali)

| Parametro | Range | Significato |
|-----------|------:|-------------|
| `obs_l` (m)   | [3.0, 6.0]   | Lunghezza |
| `obs_w` (m)   | [3.0, 6.0]   | Larghezza |
| `obs_h` (m)   | [3.0, 5.0]   | Altezza |
| `obs_x` (m)   | [-12.0, -5.0]| Posizione longitudinale |
| `obs_y` (m)   | [-2.0, 6.0]  | Posizione laterale |
| `obs_r` (°)   | [0, 180]     | Rotazione |

### Funzione di fitness

**Distanza minima drone-ostacolo** durante la missione, da minimizzare.
**Soglia di crash strict**: `dist == 0.0m` (solo collisioni reali).

---

## 2. Architettura della soluzione

```
┌────────────────────────────────────────────────────────────────────────┐
│                         DOE — 3 REPLICHE                               │
│                  seed accoppiati: 42, 123, 456                         │
└────────────────────────────────────────────────────────────────────────┘

       ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
       │  STRATEGIA A     │    │  STRATEGIA B     │    │  STRATEGIA C     │
       │  Random Search   │    │   GA Puro        │    │ GA + Surrogato   │
       └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
                │                       │                       │
                │ 100 sim               │ 20 + 4×18 = 92        │ 50 random     
                │ random                │ (con elitismo)        │   
                ▼                       ▼                       ▼
       ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
       │ dataset_random_  │    │ tutte_le_        │    │ dataset_pipeline │
       │ rN.csv           │    │ simulazioni.csv  │    │ _rN_50.csv       │
       └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
                │                       │                       │
                │                       │                       ▼
                │                       │             ┌──────────────────┐
                │                       │             │ Training         │
                │                       │             │ Surrogato        │
                │                       │             │ (RF, GB, MLP)    │
                │                       │             └────────┬─────────┘
                │                       │                      │
                │                       │                      ▼
                │                       │             ┌──────────────────┐
                │                       │             │ GA su Surrogato  │
                │                       │             │ (pop=200,        │
                │                       │             │  gen=300)        │
                │                       │             └────────┬─────────┘
                │                       │                      │ + 50 val
                │                       │                      ▼
                │                       │             ┌──────────────────┐
                │                       │             │ Validazione 50   │
                │                       │             │ candidati su     │
                │                       │             │ Aerialist        │
                │                       │             └────────┬─────────┘
                ▼                       ▼                      ▼
       ┌────────────────────────────────────────────────────────────────┐
       │                  CONFRONTO FINALE                              │
       │   confronto.py                                                 │
       │   - N. crash assoluti per strategia                            │
       │   - Crash rate %                                               │
       │   - Distribuzione distanze (boxplot)                           │
       │   - Variabilità inter-replica                                  │
       │   - Accuratezza surrogato (pred vs reale)                      │
       |   - Confronto prestazioni surrogati                            |
       |   - Confornto diversity per  scenari di crash                  |
       └────────────────────────────────────────────────────────────────┘
```

---

## 3. Requisiti di sistema
Tutti i riferimenti per i requisiti di sistema necessari sono contenuti nella modalità di esecuzione della repository ufficiale:
https://github.com/skhatiri/Aerialist/tree/master

### Librerie Python

Tutte le dipendenze sono in `requirements.txt`:


## 4. Struttura del repository

```
Aerialist-UAV-Testing/
│
├── README.md                        # Questo file
├── LICENSE.md                       # MIT License
├── requirements.txt                 # Dipendenze Python
├── scripts/                         # ★ TUTTI GLI SCRIPT DELLA PIPELINE
│   ├── generatore_dataset_random.py    # Genera dataset (random o training surrogato)
│   ├── addestra_surrogato_r1.py        # Addestra surrogati per Replica 1
│   ├── addestra_surrogato_r2.py        # Addestra surrogati per Replica 2
│   ├── addestra_surrogato_r3.py        # Addestra surrogati per Replica 3
│   ├── algoritmo_genetico.py           # GA su surrogato → 50 candidati
│   ├── GA.py                            # GA puro (Micro-GA) sul simulatore reale
│   ├── validation.py                   # Valida candidati GA sul simulatore reale
│   └── confronto.py                    # Confronto finale a 3 vie con grafici
│
├── dataset/                         # Dataset generati
│   ├── random/                      # Random Search (3 repliche × 100 sim)
│   │   ├── dataset_random_r1.csv
│   │   ├── dataset_random_r2.csv
│   │   └── dataset_random_r3.csv
│   └── pipeline_train/              # Training surrogato (3 repliche × 50 sim)
│       ├── dataset_pipeline_r1_50.csv
│       ├── dataset_pipeline_r2_50.csv
│       └── dataset_pipeline_r3_50.csv
│
├── models/                          # Surrogati addestrati (output)
│   ├── pipeline_r1_50/
│   │   ├── gradientboosting.pkl
│   │   ├── randomforest.pkl
│   │   ├── neuralnetwork.pkl
│   │   ├── confronto_modelli.csv
│   │   └── feature_importance.csv
│   ├── pipeline_r2_50/  (idem)
│   └── pipeline_r3_50/  (idem)
│
├── ga_results/                      # Output del GA con surrogato
│   ├── pipeline_r1_50/
│   │   ├── candidati_top50.csv     # 50 candidati selezionati dal GA
│   │   ├── convergenza_ga.png       # Plot convergenza
│   │   ├── ga_config.json           # Metadati esecuzione
│   │   ├── validazione_risultati.csv  # Predetto vs reale (50 sim)
│   │   └── validazione_summary.txt
│   ├── pipeline_r2_50/  (idem)
│   └── pipeline_r3_50/  (idem)
│
├── risultati_solo_GA/               # Output del GA Puro
│   ├── replica1/
│   │   ├── tutte_le_simulazioni.csv   # 92 sim eseguite
│   │   ├── convergenza_ga_puro.png
│   │   └── ga_puro_summary.json
│   ├── replica2/  (idem)
│   └── replica3/  (idem)
│
├── confronto/                       # Output del confronto finale a 3 vie
│   ├── 01_crash_3strategie.png
│   ├── 02_distance_boxplot.png
│   ├── 03_per_replica_crash.png
│   ├── 04_surrogate_accuracy.png
│   ├── 05_feature_importance.png
│   ├── 06_surrogate_cv_metrics.png
│   ├── 07_evolution_ga_puro.png
│   ├── 08_diversity_3strategie.png
│   ├── 09_diversity_ga_surr_per_replica.png
│   ├── metriche_aggregate.csv
│   └── report_finale.txt
│
├── samples/tests/                   # Template missione + YAML generati
│   └── mission1.yaml                # Template base della missione

---

## 5. Pipeline completa — comandi passo-passo


### Step 1 — Genera i 3 dataset Random Search

Modificare in `scripts/generatore_dataset_random.py` per ogni replica:

```python
RANDOM_SEED = 42    # → 123 per Rep2, → 456 per Rep3
DATASET_CSV = "dataset_random_r1.csv"
OUTPUT_DIR  = "samples/tests/generated_random_r1/"
N_MISSIONI  = 100
```

Poi lanciare (sequenzialmente):

```bash
# Replica 1
python3 scripts/generatore_dataset_random.py 

# Replica 2 (dopo aver modificato i parametri)
python3 scripts/generatore_dataset_random.py 

# Replica 3 (dopo aver modificato i parametri)
python3 scripts/generatore_dataset_random.py 
```

**Output**: `dataset_random_r{1,2,3}.csv` nella root.

---

### Step 2 — Genera i 3 dataset di training del surrogato (50 sim ciascuno)

Stesso script di Step 1, modificare i parametri:

```python
RANDOM_SEED = 42    # → 123, 456
DATASET_CSV = "dataset_pipeline_r1_50.csv"
OUTPUT_DIR  = "samples/tests/generated_pipeline_r1_50/"
N_MISSIONI  = 50
```

```bash
python3 scripts/generatore_dataset_random.py 
# ... ripeti per r2 e r3
```

**Output**: `dataset_pipeline_r{1,2,3}_50.csv` nella root.

---

### Step 3 — Addestra i 3 surrogati

Per ogni replica c'è uno script dedicato con il seed pre-configurato:

```bash
python3 scripts/addestra_surrogato_r1.py
python3 scripts/addestra_surrogato_r2.py
python3 scripts/addestra_surrogato_r3.py
```

Ogni script:
1. Carica `dataset_pipeline_rN_50.csv`
2. Addestra **RandomForest**, **GradientBoosting**, **MLPRegressor**
3. Esegue **5-fold cross-validation** con fold identici
4. Salva i tre `.pkl` + metriche CV 

**Output**: `models/pipeline_r{1,2,3}_50/` con i seguenti file:
- `gradientboosting.pkl` (modello vincitore)
- `randomforest.pkl`, `neuralnetwork.pkl`
- `confronto_modelli.csv`

---

### Step 4 — Esegui il GA su surrogato (3 repliche)

Usa il **modello vincitore**  per ogni replica:

```bash
# Replica 1 — seed 42
python3 scripts/algoritmo_genetico.py \
    --surrogate models/pipeline_r1_50/gradientboosting.pkl \
    --output    ga_results/pipeline_r1_50/ \
    --top-k     50 \
    --seed      42

# Replica 2 — seed 123
python3 scripts/algoritmo_genetico.py \
    --surrogate models/pipeline_r2_50/gradientboosting.pkl \
    --output    ga_results/pipeline_r2_50/ \
    --top-k     50 \
    --seed      123

# Replica 3 — seed 456
python3 scripts/algoritmo_genetico.py \
    --surrogate models/pipeline_r3_50/gradientboosting.pkl \
    --output    ga_results/pipeline_r3_50/ \
    --top-k     50 \
    --seed      456
```

- Output: top-50 candidati **diversificati** (selezione greedy)

**Output per replica**:
- `candidati_top50.csv`
- `convergenza_ga.png`
- `ga_config.json`

---

### Step 5 — Valida i 50 candidati su Aerialist (3 repliche)

```bash
# Replica 1
 python3 scripts/validation.py \
    --input  ga_results/pipeline_r1_50/candidati_top50.csv \
    --output ga_results/pipeline_r1_50/ 
    

# Replica 2
 python3 scripts/validation.py \
    --input  ga_results/pipeline_r2_50/candidati_top50.csv \
    --output ga_results/pipeline_r2_50/ 
   

# Replica 3
 python3 scripts/validation.py \
    --input  ga_results/pipeline_r3_50/candidati_top50.csv \
    --output ga_results/pipeline_r3_50/ 
   
```

**Output**: `validazione_risultati.csv`. `validazione_summary.txt`

---

### Step 6 — Esegui il GA Puro su simulatore reale (3 repliche)

```bash
# Replica 1
 python3 scripts/GA.py \
    --output risultati_solo_GA/replica1/ \
    --seed   42 
    

# Replica 2
 python3 scripts/GA.py \
    --output risultati_solo_GA/replica2/ \
    --seed   123 
    

# Replica 3
 python3 scripts/GA.py \
    --output risultati_solo_GA/replica3/ \
    --seed   456 

```

**Output**: `risultati_solo_GA/replica{1,2,3}/`
- `tutte_le_simulazioni.csv` 
- `convergenza_ga_puro.png`
- `ga_puro_summary.json`
- `ga_puro_summary.txt`

---

### Step 7 — Confronto finale a 3 vie

```bash
python3 scripts/confronto.py --output confronto/
```

**Genera 7 grafici + 2 file riepilogativi**:

| File | Contenuto |
|------|-----------|
| `01_crash_3strategie.png` | Crash assoluti e crash rate per strategia (media ± std) |
| `02_distance_boxplot.png` | Distribuzione distanze sull'intero budget |
| `03_per_replica_crash.png` | Variabilità inter-replica |
| `04_surrogate_accuracy.png` | Predetto vs reale per ogni replica |
| `06_surrogate_cv_metrics.png` | MAE / R² in CV dei surrogati |
| `07_evolution_ga_puro.png` | Convergenza del GA puro per generazione |
| `08_diversity_3strategie.png` | confornto della diversity per gli scenari di crash tra i 3 approcci |
| `09_diversity_ga_surr_per_replica.png` | analisi della diversity per le singole repliche relative a surr+ga |
| `metriche_aggregate.csv` | Metriche per replica + media e std |
| `report_finale.txt` | Riepilogo testuale con interpretazione |

---


## 6. Riferimenti

### Articoli scientifici principali

- **Aerialist**: Sajad Khatiri, Sebastiano Panichella, and Paolo Tonella, 
"Simulation-based Test Case Generation for Unmanned Aerial Vehicles in the Neighborhood of Real Flights," 
In 2023 IEEE 16th International Conference on Software Testing, Verification and Validation (ICST)
- **Surrealist**: Sajad Khatiri, Sebastiano Panichella, and Paolo Tonella, "Simulation-based Testing of Unmanned Aerial Vehicles with Aerialist,"
In 2024 International Conference on Software Engineering (ICSE)


### Strumenti

- **Aerialist**: https://github.com/skhatiri/Aerialist
- **Docker image**: `skhatiri/aerialist:latest`


## 👥 Autori

- **Marino Domenico** 
- **Mennillo Domenico** 
- **Perillo Gabriele** 

