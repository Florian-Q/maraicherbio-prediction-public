# 🥦 Dashboard MaraîcherBio — Documentation

## Architecture

```
app.py
├── 1. Configuration de la page (wide, sidebar expanded)
├── 2. Chargement des données
│   ├── Données réelles : utils.charger_dataframe() + uc_products.csv
│   └── Données démo   : _generate_demo_data() (12 produits fictifs)
├── 3. Moteur de prédiction
│   ├── train_baseline_model()     → moyenne + écart-type par semaine ISO
│   └── generate_forecast()        → 52 semaines de prévisions + IC 95%
├── 4. Préparation des données
│   └── prepare_product_data()     → agrège, entraîne, produit df_forecast + df_summary
├── 5. Graphiques Plotly
│   ├── plot_history_forecast()    → courbe historique (noir) + prévisions (rouge) + IC
│   └── plot_top10_bar()           → barres horizontales top 10
├── 6. Composants Streamlit
│   ├── render_kpi_row()           → 5 métriques clés (Qté prévue, CA, Impact prix...)
│   ├── render_product_card()      → fiche produit (MAE, MAPE, prix, unité)
│   └── render_price_simulator()   → number_input prix + bouton reset
└── 7. Application principale (main)
```

---

## Données

| Composant | Détail |
|---|---|
| **Source réelle** | `utils.charger_dataframe()` → `uc_order_products.csv` + `uc_orders.csv` + `uc_products.csv` |
| **Source démo** | 12 produits fictifs avec saisonnalité été/hiver (si CSV indisponibles) |
| **Dates** | Du 1er dimanche 2021 au 31 mai 2026 |
| **Fréquence** | Hebdomadaire (`W-SUN`) |

---

## Modèle de prédiction

**Baseline : moyenne par semaine ISO.**

Pour chaque produit :
1. Les ventes sont agrégées par semaine (somme des quantités)
2. On calcule la moyenne et l'écart-type pour chaque semaine calendaire (1→53)
3. Les 52 semaines futures sont prédites en appliquant la moyenne de la semaine correspondante
4. L'intervalle de confiance à 95% = moyenne ± 1.96 × écart-type

---

## Interface utilisateur

### Sidebar (gauche)

| Widget | Options | Rôle |
|---|---|---|
| **Produit** | `selectbox` avec recherche | Choisir le produit à afficher |
| **Période** | Futur / 1 an / 2 ans / 3 ans / Complet | Zoomer sur l'historique |
| **Métrique** | Quantité vendue / CA (€) | Basculer l'axe Y du graphique |
| **Classement** | Top CA / Volume / Croissance / Part CA | Choisir le graphique portefeuille |

### Zone principale (2 colonnes)

**Colonne gauche (3/4) :**
- Graphique Plotly : historique (noir) + prévisions (rouge) + bande d'incertitude (rose)
- Ligne verticale « Aujourd'hui » à la jonction historique/prévisions
- 5 KPI cards : Qté prévue, CA prévisionnel, Impact prix, Prix actuel, Prix simulé
- 2 graphiques portefeuille (top 10 selon classement + part du CA total)

**Colonne droite (1/4) :**
- Fiche produit (nom, unité, modèle, MAE, MAPE, prix)
- Simulateur de prix (`number_input` + bouton reset → `sell_price` du produit)
- Résumé 52 semaines (Qté totale, moyenne/semaine, pic, CA, nb semaines sans vente)

---

## Simulateur de prix

| Élément | Comportement |
|---|---|
| **Prix saisi** | Mis à jour en temps réel dans le graphique (si mode CA) et les KPI |
| **Bouton Reset** | Réinitialise le prix au `sell_price` du produit (depuis `uc_products.csv`) |
| **Plage** | 0.01 € → 5× le prix de base |
| **Pas** | 0.01 € (centime) |
| **Impact** | Affiche le delta de CA en € et % par rapport au prix de base |

---

## Graphiques Plotly

### Historique & Prévisions
- **Courbe noire** = ventes réelles hebdomadaires
- **Courbe rouge** = prévisions baseline (52 semaines)
- **Bande rose** = intervalle de confiance à 95%
- **Ligne grise verticale** = dernière date connue
- **Mode CA** = toutes les valeurs × prix (réel ou simulé)

### Portefeuille (2 graphiques côte à côte)
- **Gauche** = top 10 selon le classement choisi (barres horizontales)
- **Droite** = part en % de chaque produit dans le CA prévisionnel total

---

## Points d'entrée pour modifications futures

| Tu veux... | Fichier/modifier |
|---|---|
| Changer le modèle de prédiction | `train_baseline_model()` et `generate_forecast()` dans `app.py` |
| Ajouter/supprimer des filtres | Section 7 (`main()`), bloc `st.sidebar` |
| Modifier l'apparence des graphiques | Fonctions `plot_history_forecast()` et `plot_top10_bar()` |
| Changer le simulateur de prix | `render_price_simulator()` |
| Utiliser un autre modèle (XGBoost, etc.) | Importer depuis `moi/Baseline/` + remplacer le moteur de prédiction |
| Connecter aux vrais résultats de modélisation | Remplacer `_generate_demo_data()` par le chargement des CSV de résultats |