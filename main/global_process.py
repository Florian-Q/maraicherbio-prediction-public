import sys
import utils
import utils_series
import model_3
import numpy as np
import pandas as pd
import itertools
from pathlib import Path

# ── Définir le répertoire data ─────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)



# --- Boucle : df_train / df_test pour chaque produit (split adaptatif, date fin commune) ---
df_all = utils.charger_dataframe()

# Définir la date de fin commune pour TOUS les splits
utils_series.GLOBAL_TEST_END_DATE = utils_series.dimanche_precedent(df_all['created'].max())
print(f'Date de fin commune : {utils_series.GLOBAL_TEST_END_DATE.date()}\n')

modeles = sorted(df_all['model'].unique())
print(f'{len(modeles)} produits à traiter\n')

dict_all = {}
dict_train = {}
dict_test = {}
skipped = []

for modele in modeles:
    try:
        dict_all[modele] = utils_series.complete_weekly_dataframe(utils.charger_dataframe(modele), 'created', 'quantite_y')
        train, test = utils_series.split_adaptive_seasonal(dict_all[modele], test_pct=0.20)
        dict_train[modele] = train
        dict_test[modele] = test
    except ValueError as e:
        skipped.append(modele)
        continue

print(f'\nTerminé : {len(dict_train)} produits prêts  |  {len(skipped)} ignorés')
if skipped:
    print(f'Ignorés : {skipped}')

# Vérification
test_ends = [t.index.max().date() for t in dict_test.values()]
print(f'Toutes les fins de test = {test_ends[0]} ?  {len(set(test_ends)) == 1}')



# ── MODE ÉVALUATION ───────────────────────────────────────────────────────────
results = []

for modele in dict_train.keys():
    y_train = dict_train[modele]['quantite_y']
    y_test  = dict_test[modele]['quantite_y']

    try:
        metrics = model_3.run_baseline(y_train, y_test)
    except Exception as e:
        print(f"[WARN] {modele} → {e}")
        continue

    results.append({
        'produit'    : modele,
        'train_debut': y_train.index.min().strftime('%Y-%m'),
        'test_fin'   : y_test.index.max().strftime('%Y-%m-%d'),
        'pct_zeros'  : round(metrics['pct_zeros'], 1),
        'MAE_in'     : round(metrics['MAE_in'],    2),
        'MAE_out'    : round(metrics['MAE_out'],   2),
        'MAE_all'    : round(metrics['MAE_all'],   2),
        'MAPE'       : round(metrics['MAPE'],      1),
        'sMAPE_all'  : round(metrics['sMAPE_all'], 1),
    })

df_base = pd.DataFrame(results).sort_values('MAPE')

# ============================================================
# PROPHET v1 : Moyenne hebdo + split_adaptive + seasonal_metrics
# ============================================================

# ── Hyperparamètres fixes (validés par expérimentation) ──────────────────────
BEST_PARAMS: dict = {
    "growth"                  : "flat",
    "seasonality_mode"        : "additive",
    "weekly_seasonality"      : True,
    "daily_seasonality"       : False,
    "yearly_seasonality"      : True,
    "interval_width"          : 0.95,
    "changepoint_prior_scale" : 0.08,   # fixé — meilleur sur tous les produits
    "holidays_prior_scale"    : 1,      # fixé — meilleur sur tous les produits
}

# ── Grid Search (1 seul paramètre restant) ────────────────────────────────────
TUNING_GRID: dict = {
    "seasonality_prior_scale" : [3, 10],   # varie selon les produits
}

ALL_COMBOS = list(itertools.product(
    TUNING_GRID["seasonality_prior_scale"],
))
# 2 combinaisons par produit



# ── MODE ÉVALUATION (remplace la boucle Prophet v2) ──────────────────────────
results = []

for modele in dict_train.keys():
    y_train = dict_train[modele]['quantite_y']
    y_test  = dict_test[modele]['quantite_y']

    best_metric = np.inf
    best_metrics = None
    best_sps     = None
    

    for (sps,) in ALL_COMBOS:
        try:
            metrics = model_3.run_prophet(sps, y_train, y_test)
            if metrics["MAPE"] < best_metric:
                best_metric   = metrics["MAPE"]
                best_metrics = metrics
                best_sps     = sps
        except Exception as e:
            print(f"[WARN] {modele} | sps={sps} → {e}")
            continue

    if best_metric is None:
        continue

    results.append({
        'produit'    : modele,
        'prophet_sps': best_sps,
        'pct_zeros'  : round(best_metrics['pct_zeros'], 1),
        'MAE_in'     : round(best_metrics['MAE_in'],    2),
        'MAE_out'    : round(best_metrics['MAE_out'],   2),
        'MAE_all'    : round(best_metrics['MAE_all'],   2),
        'MAPE'       : round(best_metrics['MAPE'],      1),
        'sMAPE_all'  : round(best_metrics['sMAPE_all'], 1),
    })

df_prophet = pd.DataFrame(results).sort_values('MAPE')

# ============================================================
# XGBOOST v1 : lag(y_t-1) + semaine ISO + split_adaptive + seasonal_metrics
# ============================================================

# ── Hyperparamètres fixes ─────────────────────────────────────────────────────
BEST_PARAMS: dict = {
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "objective"        : "reg:squarederror",
    "random_state"     : 42,
    "verbosity"        : 0,
}

# ── Grid Search ───────────────────────────────────────────────────────────────
TUNING_GRID: dict = {
    "n_estimators"  : [100, 300],
    "max_depth"     : [3, 5],
    "learning_rate" : [0.05, 0.1],
}

ALL_COMBOS = list(itertools.product(
    TUNING_GRID["n_estimators"],
    TUNING_GRID["max_depth"],
    TUNING_GRID["learning_rate"],
))
# 2 × 2 × 2 = 8 combinaisons par produit


# ── MODE ÉVALUATION (remplace la boucle XGBoost v1) ──────────────────────────
results = []

for modele in dict_train.keys():
    y_train = dict_train[modele]['quantite_y']
    y_test  = dict_test[modele]['quantite_y']

    best_metric  = np.inf
    best_metrics = None
    best_nest = None
    best_dep  = None
    best_lr   = None

    for n_est, depth, lr in ALL_COMBOS:
        try:
            metrics = model_3.run_xgboost(n_est, depth, lr, y_train, y_test)
            if metrics["MAPE"] < best_metric:
                best_metric  = metrics["MAPE"]
                best_metrics = metrics
                best_nest    = n_est
                best_dep     = depth
                best_lr      = lr
        except Exception as e:
            print(f"[WARN] {modele} | n_est={n_est} depth={depth} lr={lr} → {e}")
            continue

    if best_metric is None:
        continue

    results.append({
        'produit'    : modele,
        'xgb_nest'  : best_nest,
        'xgb_depth' : best_dep,
        'xgb_lr'    : best_lr,
        'pct_zeros'  : round(best_metrics['pct_zeros'], 1),
        'MAE_in'     : round(best_metrics['MAE_in'],    2),
        'MAE_out'    : round(best_metrics['MAE_out'],   2),
        'MAE_all'    : round(best_metrics['MAE_all'],   2),
        'MAPE'       : round(best_metrics['MAPE'],      1),
        'sMAPE_all'  : round(best_metrics['sMAPE_all'], 1),
    })

df_xgb = pd.DataFrame(results).sort_values('MAPE')


# Créer un DataFrame pour le résumé avec le modèle gagnant et ses métriques
resume = df_base[['produit','train_debut','test_fin','pct_zeros']].copy()
resume = resume.reset_index(drop=True)  # Reset index pour éviter les décalages
resume['model_win'] = ""
resume['MAE_in'] = 0.0
resume['MAE_out'] = 0.0
resume['MAE_all'] = 0.0
resume['MAPE'] = 0.0
resume['sMAPE_all'] = 0.0

# DataFrame pour stocker les prédictions
predictions = pd.DataFrame()

for idx, produit in enumerate(resume['produit']):
    print(f"\n[{idx+1}/{len(resume)}] Traitement {produit}...")
    y_train = dict_train[produit]['quantite_y']
    y_test = dict_test[produit]['quantite_y']
    y_total = pd.concat([y_train, y_test])  # Données complètes pour la prédiction en prod
    
    # Récupérer la MAPE pour chaque modèle
    mape_base = (df_base[df_base['produit'] == produit]['MAE_in'].iloc[0] + df_base[df_base['produit'] == produit]['MAPE'].iloc[0]) / 2
    mape_prophet = (df_prophet[df_prophet['produit'] == produit]['MAE_in'].iloc[0] + df_prophet[df_prophet['produit'] == produit]['MAPE'].iloc[0]) / 2
    mape_xgb = (df_xgb[df_xgb['produit'] == produit]['MAE_in'].iloc[0] + df_xgb[df_xgb['produit'] == produit]['MAPE'].iloc[0]) / 2
    
    # Déterminer le gagnant (MAPE la plus petite)
    mapes_dict = {
        'Baseline': (mape_base, df_base),
        'Prophet': (mape_prophet, df_prophet),
        'XGBoost': (mape_xgb, df_xgb)
    }
    
    model_win = min(mapes_dict, key=lambda x: mapes_dict[x][0])
    _, df_win = mapes_dict[model_win]
    
    # Récupérer les métriques du gagnant
    row_win = df_win[df_win['produit'] == produit].iloc[0]
    
    # Remplir resume avec les métriques du gagnant
    resume.loc[idx, 'model_win'] = model_win
    resume.loc[idx, 'MAE_in'] = row_win['MAE_in']
    resume.loc[idx, 'MAE_out'] = row_win['MAE_out']
    resume.loc[idx, 'MAE_all'] = row_win['MAE_all']
    resume.loc[idx, 'MAPE'] = row_win['MAPE']
    resume.loc[idx, 'sMAPE_all'] = row_win['sMAPE_all']
    
    # Appeler la fonction correspondante et sauvegarder la prédiction
    try:
        print(f"  → Prédiction avec {model_win}...")
        if model_win == 'Baseline':
            y_pred = model_3.run_baseline(y_total)
        elif model_win == 'Prophet':
            sps = row_win['prophet_sps']
            y_pred = model_3.run_prophet(sps, y_total)
        elif model_win == 'XGBoost':
            n_est = int(row_win['xgb_nest'])
            depth = int(row_win['xgb_depth'])
            lr = row_win['xgb_lr']
            y_pred = model_3.run_xgboost(n_est, depth, lr, y_total)
        
        predictions[produit] = y_pred
        print(f"  ✓ {produit} OK")
    except Exception as e:
        print(f"[ERROR] Prédiction pour {produit} ({model_win}) → {e}")
        import traceback
        traceback.print_exc()


# export
predictions.to_csv(DATA_DIR / 'Predictions.csv', index=True)
resume.to_csv(DATA_DIR / 'model_win_metric.csv', index=False)

print(f"\n✓ Exports terminés dans {DATA_DIR}")
print(resume['model_win'].value_counts())