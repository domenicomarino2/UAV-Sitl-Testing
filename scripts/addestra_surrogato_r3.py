"""
addestra_surrogato_aerialist.py
================================
Addestra e confronta tre modelli surrogati sul dataset Ramo A (Aerialist):
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Neural Network (MLP)

Target: distanza_minima_m (regressione)
Feature: obs_l, obs_w, obs_h, obs_x, obs_y, obs_r  (obs_z escluso perché costante)
Validazione: 5-fold cross-validation con fold IDENTICI per i 3 modelli (confronto equo)

Output:
  - models/aerialist/random_forest.pkl
  - models/aerialist/gradient_boosting.pkl
  - models/aerialist/neural_network.pkl
  - models/aerialist/confronto_modelli.csv  (metriche di CV)
  - models/aerialist/feature_importance.csv (RF + GB)

Uso:
    python3 addestra_surrogato_aerialist.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Silenzia warning MLP su convergenza (gestito con early stopping)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ════════════════════════════════════════════════════════════════════════════

INPUT_CSV  = "dataset_pipeline_r3_50.csv"
OUTPUT_DIR = Path("models/pipeline_r3_50")
TARGET_COL   = "distanza_minima_m"

# 6 feature di input (obs_z escluso: costante a 0)
FEATURE_COLS = ["obs_l", "obs_w", "obs_h", "obs_x", "obs_y", "obs_r"]

RANDOM_SEED = 456
CV_FOLDS    = 5

# ════════════════════════════════════════════════════════════════════════════
# CARICAMENTO E PULIZIA DATI
# ════════════════════════════════════════════════════════════════════════════

def load_and_clean(csv_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Carica il CSV ed esclude le righe con distanza_minima_m mancante
    (run falliti con no_log_saved o errori di parsing).

    Restituisce: (dataframe pulito, X array, y array)
    """
    print(f"📂 Caricamento dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    n_tot = len(df)

    # Drop righe senza distanza minima
    df_clean = df.dropna(subset=[TARGET_COL]).copy()
    n_clean = len(df_clean)
    n_dropped = n_tot - n_clean

    print(f"   Totale righe:       {n_tot}")
    print(f"   Righe valide:       {n_clean}")
    print(f"   Righe escluse:      {n_dropped} (no_log_saved / errori)")

    # Verifica che tutte le feature siano presenti
    missing = [c for c in FEATURE_COLS if c not in df_clean.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {missing}")

    # Verifica obs_z costante (sanity check)
    if "obs_z" in df_clean.columns:
        if df_clean["obs_z"].nunique() > 1:
            print(f"   ⚠️  obs_z NON è costante! Considera di includerlo.")
        else:
            print(f"   ✓ obs_z costante ({df_clean['obs_z'].iloc[0]}) → escluso dal modello")

    X = df_clean[FEATURE_COLS].values.astype(np.float64)
    y = df_clean[TARGET_COL].values.astype(np.float64)

    print(f"   Shape X: {X.shape}, Shape y: {y.shape}")
    print(f"   Target range: [{y.min():.3f}, {y.max():.3f}] m")
    print(f"   Target mean:  {y.mean():.3f} m, std: {y.std():.3f} m")
    print(f"   Collisioni (y == 0): {(y == 0).sum()}/{len(y)}")

    return df_clean, X, y

# ════════════════════════════════════════════════════════════════════════════
# DEFINIZIONE MODELLI
# ════════════════════════════════════════════════════════════════════════════

def build_models() -> dict[str, Pipeline]:
    """
    Costruisce i tre modelli da confrontare.

    Note:
    - RF e GB non hanno bisogno di standardizzazione (tree-based)
    - MLP richiede standardizzazione obbligatoria → pipeline con StandardScaler
    """
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,            # crescita libera, RF è robusto a overfitting con 200 alberi
        min_samples_leaf=2,        # con 75 sample, evita foglie troppo specifiche
        max_features="sqrt",       # sub-sampling delle feature: standard per regressione
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,        # learning rate basso = più alberi ma generalizza meglio
        max_depth=4,               # profondità contenuta: GB con dataset piccoli overfitta facile
        min_samples_leaf=3,
        subsample=0.8,             # stochastic gradient boosting: regolarizza
        random_state=RANDOM_SEED,
    )

    # MLP con architettura piccola (75 sample → rete piccola obbligatoria)
    # (32, 16) = 6→32→16→1 = ~770 parametri, ragionevole per ~75 sample
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=0.01,
            max_iter=3000,
            early_stopping=True,    # ferma quando validation loss non migliora
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=RANDOM_SEED,
        )),
    ])

    return {
        "RandomForest":     rf,
        "GradientBoosting": gb,
        "NeuralNetwork":    mlp,
    }

# ════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION CONFRONTO EQUO
# ════════════════════════════════════════════════════════════════════════════

def evaluate_with_cv(models: dict, X: np.ndarray, y: np.ndarray
                     ) -> pd.DataFrame:
    """
    Esegue k-fold cross-validation con fold IDENTICI per tutti i modelli.

    Perché fold identici? Per confronto equo: se RF e MLP vedono split diversi
    dei dati, le metriche non sono comparabili. KFold(random_state=fisso)
    garantisce stessa partizione.

    Restituisce un DataFrame con metriche per ogni modello.
    """
    print(f"\n{'='*60}")
    print(f"🔬 Cross-validation {CV_FOLDS}-fold (fold identici per tutti)")
    print(f"{'='*60}")

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    results = []
    for model_name, model in models.items():
        print(f"\n   ▶ {model_name}")
        mae_scores  = []
        rmse_scores = []
        r2_scores   = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Re-instanzia il modello per evitare contaminazione tra fold
            # (importante per pipeline con stato come MLP)
            model_clone = _clone_model(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)

            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2   = r2_score(y_test, y_pred)

            mae_scores.append(mae)
            rmse_scores.append(rmse)
            r2_scores.append(r2)

            print(f"      Fold {fold_idx+1}: MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

        mae_mean,  mae_std  = np.mean(mae_scores),  np.std(mae_scores)
        rmse_mean, rmse_std = np.mean(rmse_scores), np.std(rmse_scores)
        r2_mean,   r2_std   = np.mean(r2_scores),   np.std(r2_scores)

        print(f"      ──────────────────────────────────────────────")
        print(f"      Media:   MAE={mae_mean:.3f}±{mae_std:.3f}  "
              f"RMSE={rmse_mean:.3f}±{rmse_std:.3f}  "
              f"R²={r2_mean:.3f}±{r2_std:.3f}")

        results.append({
            "model":     model_name,
            "mae_mean":  round(mae_mean, 4),
            "mae_std":   round(mae_std, 4),
            "rmse_mean": round(rmse_mean, 4),
            "rmse_std":  round(rmse_std, 4),
            "r2_mean":   round(r2_mean, 4),
            "r2_std":    round(r2_std, 4),
        })

    return pd.DataFrame(results)


def _clone_model(model):
    """Clona un modello sklearn (anche Pipeline) per evitare stato persistente."""
    from sklearn.base import clone
    return clone(model)

# ════════════════════════════════════════════════════════════════════════════
# TRAINING FINALE + SALVATAGGIO
# ════════════════════════════════════════════════════════════════════════════

def train_and_save(models: dict, X: np.ndarray, y: np.ndarray,
                   output_dir: Path) -> dict:
    """
    Riallena ciascun modello sull'INTERO dataset e lo salva su disco.

    Perché tutto il dataset? La CV serve a stimare l'accuratezza.
    Il modello finale che useremo nel GA deve sfruttare al massimo
    i dati disponibili (75 sample sono già pochi).
    """
    print(f"\n{'='*60}")
    print(f"💾 Training finale sull'intero dataset e salvataggio")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    trained = {}

    for model_name, model in models.items():
        model_clone = _clone_model(model)
        model_clone.fit(X, y)
        trained[model_name] = model_clone

        filename = output_dir / f"{model_name.lower()}.pkl"
        joblib.dump(model_clone, filename)
        print(f"   ✓ {model_name:20s} → {filename}")

    return trained

# ════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════

def feature_importance_report(trained_models: dict,
                              X: np.ndarray, y: np.ndarray,
                              feature_names: list[str],
                              output_dir: Path) -> pd.DataFrame:
    """
    Calcola e salva la feature importance per ogni modello.

    - RF e GB: importance nativa (basata su quanto ogni feature riduce l'impurità)
    - MLP: permutation importance (quanto peggiora la performance se shufflo
      i valori di quella feature)
    """
    print(f"\n{'='*60}")
    print(f"📊 Feature importance")
    print(f"{'='*60}")

    rows = []

    for model_name, model in trained_models.items():
        if model_name in ("RandomForest", "GradientBoosting"):
            # Importance nativa
            importances = model.feature_importances_
        else:
            # Permutation importance per la NN (più lento ma robusto)
            print(f"   Calcolo permutation importance per {model_name}...")
            result = permutation_importance(
                model, X, y,
                n_repeats=20,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                scoring="neg_mean_absolute_error",
            )
            importances = result.importances_mean

        # Normalizza a somma 1 per confronto omogeneo
        if importances.sum() > 0:
            importances_norm = importances / importances.sum()
        else:
            importances_norm = importances

        print(f"\n   {model_name}:")
        for feat, imp in sorted(zip(feature_names, importances_norm),
                                 key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"      {feat:10s} {imp:.4f}  {bar}")
            rows.append({
                "model": model_name,
                "feature": feat,
                "importance": round(float(imp), 6),
            })

    df_imp = pd.DataFrame(rows)
    out_path = output_dir / "feature_importance.csv"
    df_imp.to_csv(out_path, index=False)
    print(f"\n   Salvato in: {out_path}")
    return df_imp

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("🧠 ADDESTRAMENTO SURROGATO — RAMO A (Aerialist)")
    print("="*60)

    # 1. Carica e pulisci
    df, X, y = load_and_clean(INPUT_CSV)

    # 2. Costruisci i tre modelli
    models = build_models()

    # 3. Cross-validation per confronto equo
    cv_results = evaluate_with_cv(models, X, y)

    # Salva tabella confronto
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_path = OUTPUT_DIR / "confronto_modelli.csv"
    cv_results.to_csv(cv_path, index=False)
    print(f"\n📋 Confronto modelli salvato in: {cv_path}")
    print(cv_results.to_string(index=False))

    # Identifica il "migliore" (MAE più basso)
    best_idx  = cv_results["mae_mean"].idxmin()
    best_name = cv_results.loc[best_idx, "model"]
    best_mae  = cv_results.loc[best_idx, "mae_mean"]
    print(f"\n🏆 Modello con minor MAE (CV): {best_name} (MAE={best_mae:.3f}m)")

    # 4. Training finale e salvataggio
    trained = train_and_save(models, X, y, OUTPUT_DIR)

    # 5. Feature importance
    feature_importance_report(trained, X, y, FEATURE_COLS, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("🎉 COMPLETATO")
    print(f"   Modelli salvati in: {OUTPUT_DIR}/")
    print(f"   - random_forest.pkl")
    print(f"   - gradient_boosting.pkl")
    print(f"   - neural_network.pkl")
    print(f"   - confronto_modelli.csv")
    print(f"   - feature_importance.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()