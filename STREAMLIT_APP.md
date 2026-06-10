# 🥦 Dashboard MaraîcherBio — Documentation

## Architecture

```
app.py
├── 1. Configuration de la page (wide, sidebar collapsed)
├── 2. Chargement des données
│   ├── Données réelles : utils.charger_dataframe() + uc_products.csv
│   ├── Métriques        : model_win_metric.csv (meilleur modèle + MAE/MAPE par produit)
│   └── Prédictions      : Predictions.csv (52 semaines, export de Global_process.ipynb)
├── 3. Moteur de prédiction (fallback)
│   ├── train_baseline_model()     → moyenne + écart-type par semaine ISO
│   └── generate_forecast()        → 52 semaines de prévisions + IC 95%
│   (utilisé UNIQUEMENT si Predictions.csv est absent)
├── 4. Préparation des données
│   └── prepare_all_data()         → charge, joint, agrège en une fois (@st.cache_data)
├── 5. Graphiques Plotly
│   └── plot_view()                → courbes individuelles (≤4 produits) ou cumulative (5+)
├── 6. Interface Streamlit
│   ├── Tableau éditable    → prix modifiables, checkboxes de sélection, recherche
│   ├── Graphique principal → historique + prévisions + IC 95%
│   ├── KPIs                → Qté prévue, CA prévisionnel, nombre de produits
│   └── Reset prix          → boutons de réinitialisation par produit
└── 7. Application principale (main)
```

---

## Données

| Composant | Détail |
|---|---|
| **Source réelle** | `utils.charger_dataframe()` → `uc_order_products.csv` + `uc_orders.csv` + `uc_products.csv` |
| **Prédictions** | `Predictions.csv` généré par `Global_process.ipynb` → 52 semaines × 84 produits |
| **Métriques modèles** | `model_win_metric.csv` : meilleur modèle (Baseline/Prophet/XGBoost) + MAE/MAPE par produit |
| **Prix** | `uc_products.csv` → mapping `model → sell_price` |
| **Dates** | Du 1er dimanche 2014 au 31 mai 2026 |
| **Fréquence** | Hebdomadaire (`W-SUN`) |

---

## Modèles de prédiction

La source principale de prédictions est `Predictions.csv`, exporté par `notebooks/Global_process.ipynb`.

### Pipeline d'entraînement (Global_process.ipynb)

Pour chaque produit :
1. Agrégation hebdomadaire des ventes (`complete_weekly_dataframe`)
2. Split train/test adaptatif (20% de test, arrondi à l'année)
3. Grid search sur 3 modèles :
   - **Baseline** : moyenne par semaine ISO
   - **Prophet** : saisonnalité additive, `seasonality_prior_scale` ∈ [3, 10]
   - **XGBoost** : features `[week, week_sin, week_cos, lag_1, lag_52, rolling_52]`, grid 2×2×2
4. Sélection du meilleur modèle par MAPE minimale (en saison)
5. Prédictions 52 semaines avec le modèle gagnant
6. Export → `Predictions.csv` + `model_win_metric.csv`

### Résultats

| Modèle gagnant | Produits | % |
|---|---|---|
| **XGBoost** | 52 | 62% |
| **Prophet** | 22 | 26% |
| **Baseline** | 10 | 12% |

### Fallback Baseline (dans app.py)

Si `Predictions.csv` est absent, l'app utilise un fallback baseline (moyenne par semaine ISO + IC 95%). Ce fallback est intentionnellement simple — les vraies prédictions viennent du notebook.

---

## Interface utilisateur

### Tableau principal (droite)

| Colonne | Rôle |
|---|---|
| **✅ Sél.** | Checkbox pour afficher le produit dans le graphique |
| **Produit** | Nom du produit (recherche textuelle) |
| **Unité** | kg, Pièce, Botte (depuis `weight_units`) |
| **Prix (€)** | Éditable — modifie le CA en temps réel |
| **Modèle** | Baseline / Prophet / XGBoost (meilleur modèle) |
| **MAE in** | Mean Absolute Error en saison |
| **MAPE** | Mean Absolute Percentage Error en saison |

### Filtres

| Widget | Options | Rôle |
|---|---|---|
| **Période** | Futur / 1 an / 2 ans / 3 ans / Tout | Zoom temporel |
| **Métrique** | Quantité / Chiffre d'affaires (€) | Basculer l'axe Y |
| **Recherche** | Texte libre | Filtrer les produits |
| **✅ Tous** | Bouton | Sélectionner tous les produits |
| **🔄 Aucun** | Bouton | Désélectionner tous les produits |

### Graphique (gauche)

- **1 à 4 produits** : courbes individuelles de couleurs différentes (historique = trait plein, prévisions = pointillés)
- **5+ produits** : courbe cumulative unique (historique = noir, prévisions = rouge + IC 95% rose)
- Ligne verticale à la dernière date connue
- Mode CA = toutes les valeurs × prix (réel ou modifié)
- Hover unifié avec tooltips

### KPIs

| KPI | Description |
|---|---|
| 📦 Qté prévue | Somme des prédictions sur 52 semaines |
| 💰 CA prévisionnel | Somme (prédiction × prix), avec delta si prix modifiés |
| 🧺 Produits | Nombre de produits sélectionnés |

---

## Simulateur de prix

| Élément | Comportement |
|---|---|
| **Prix saisi** | Modifie le CA en temps réel (graphique + KPIs) |
| **Bouton Reset** | Réinitialise au `sell_price` d'origine |
| **Plage** | 0.01 € → pas de limite supérieure |
| **Pas** | 0.05 € |
| **Impact** | Delta de CA en € par rapport au prix de base |

---

## Graphiques Plotly

### Historique & Prévisions
- **Courbes colorées** = produits individuels (1-4) ou tout cumulé (5+)
- **Trait plein** = historique réel
- **Pointillés** = prévisions
- **Bande rose** = intervalle de confiance à 95% (mode cumulatif uniquement)
- **Mode CA** = toutes les valeurs × prix (réel ou simulé)

---

## Points d'entrée pour modifications futures

| Tu veux... | Fichier/modifier |
|---|---|
| Changer le modèle de prédiction | Modifier `Global_process.ipynb` + réexporter `Predictions.csv` |
| Ajouter un nouveau modèle | Ajouter la fonction dans `model_3.py` + l'intégrer dans `Global_process.ipynb` |
| Ajouter/supprimer des filtres | Section `main()` dans `app.py`, bloc colonne tableau |
| Modifier l'apparence des graphiques | Fonction `plot_view()` dans `app.py` |
| Changer le simulateur de prix | Colonne `Prix (€)` dans le `data_editor` |
| Ajouter des KPIs | Bloc KPIs dans `main()` |
| Régénérer les prédictions | `make train` ou exécuter `Global_process.ipynb` |
