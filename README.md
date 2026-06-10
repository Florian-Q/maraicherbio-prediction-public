# 🥦 Prédiction des Ventes — Maraîcher Bio Lorient

> Projet Data Science — Séries Temporelles · Modèles Prédictifs · Dashboard

---

## 📌 Contexte

Ce projet est réalisé en partenariat avec un **maraîcher bio basé à Lorient (Bretagne)** qui a accordé un accès à sa base de données de ventes en ligne. L'objectif est de prédire les quantités de légumes et fruits vendus chaque semaine, afin d'aider le maraîcher à **anticiper sa production, réduire le gaspillage et optimiser ses stocks**.

Choisir ce projet, c'est aussi choisir de **soutenir une agriculture locale et biologique**.

---

## 🎯 Objectif

1. Analyser l'historique des commandes depuis 2014 afin d'identifier les produits les plus rentables, la saisonnalité des ventes et élaborer un modèle prédictif pour optimiser les stocks.
2. Construire un ou plusieurs modèles de **séries temporelles supervisés** capables de prédire les quantités hebdomadaires vendues pour chaque produit.

| Paramètre | Valeur |
|---|---|
| **Type of Challenge** | Model Training — Supervisé |
| **Variable X** | Quantités de légumes/fruits vendues par semaine |
| **Variable Y** | Prédiction des quantités futures |
| **Type de valeur** | Values (continue) |
| **ML Approach** | Time Series — Baseline · Prophet (Meta) · XGBoost |
| **Exploration & Viz** | Matplotlib · Plotly · Dashboard Streamlit |

---

## 🗃️ Données

> ⚠️ Les données sont privées et ne sont pas versionnées dans ce dépôt.

BaseDeDonnes
![Base De Donnes](BaseDeDonnees.PNG)

La base de données contient deux tables principales :

### `produit_vendu`
Lignes de ventes détaillées par commande et par produit.

| Colonne | Description |
|---|---|
| `order_product_id` | Identifiant unique de la ligne de vente |
| `order_id` | Référence de la commande |
| `nid` | Identifiant du produit |
| `title` | Nom du produit |
| `qty` | Quantité vendue |
| `price` | Prix unitaire (€) |
| `weight` | Poids unitaire (g) |

nid (Node ID) : L'identifiant unique du "contenu". C'est votre clé primaire pour un produit. Si vous avez 500 produits, chaque produit a un nid différent. Utilisez ceci pour toutes vos jointures.

vid (Version ID) : L'identifiant de la version. Chaque fois qu'une ligne est modifiée, un nouveau vid peut être créé. Pour l'analyse de données, on ignore généralement le vid pour se concentrer sur l'état actuel du produit via le nid.

### `orders`
Historique des commandes clients.

| Colonne | Description |
|---|---|
| `order_id` | Identifiant unique de la commande |
| `uid` | Identifiant client |
| `order_status` | Statut (`completed`, `processing`, …) |
| `order_total` | Montant total de la commande (€) |
| `product_count` | Nombre de produits dans la commande |
| `created` | Date de création |
| `modified` | Date de dernière modification |

uid (User ID) : L'identifiant de l'utilisateur. Il relie vos commandes à des profils clients.

**Volume de données :**
- ~142 600 lignes de ventes produit
- ~21 900 commandes
- Historique de **plus de 10 ans** (2014 – 2026)
- **84 produits** retenus après nettoyage (présents sur les 3 dernières années avec continuité + >10 ventes)

Le projet repose sur 3 fichiers sources :
- **Orders (`uc_orders.csv`)** : Liste des transactions globales.
- **Order Products (`uc_order_products.csv`)** : Table de détail (contient les articles par commande).
- **Products (`uc_products.csv`)** : Catalogue technique (prix, poids, unités).

*Le lien pivot entre les tables est le champ `nid`.*

---

## 📊 Quelques Insights Clés

- 🥇 La **Courgette** est le produit n°1 avec 1 039 unités vendues en 2025, soit +46% par rapport au second produit
- 🍅 Top 5 annuel : Courgette · Tomate rouge ronde · Poireau · Aubergine noire · Pomme de terre allians
- ☀️ Forte **saisonnalité estivale** : Courgette, Tomate et Aubergine explosent de mai à septembre
- ❄️ **Poireau et Pomme de terre** assurent les ventes hivernales (octobre → avril)

---

## 🔬 Approche Technique

### 1. Exploration & Nettoyage

- Analyse des séries temporelles par produit
- Détection des anomalies et valeurs aberrantes
- Gestion des valeurs manquantes et des semaines sans ventes
- 84 produits conservés après filtrage (présence sur 3 ans glissants, >10 ventes)
- Correction des poids mal annotés (grammes → kilogrammes)
- Uniformisation des noms de produits

### 2. Un Modèle par Légume/Fruit

Chaque produit ayant ses propres cycles saisonniers, un **modèle dédié** est entraîné par produit pour garantir une précision maximale par rapport à un modèle global.

### 3. Trois Modèles en Compétition

| Modèle | Type | Caractéristiques |
|---|---|---|
| **Baseline** | Moyenne hebdomadaire | Moyenne + écart-type par semaine ISO (référence) |
| **Prophet** (Meta) | Time Series bayésien | Saisonnalité annuelle + hebdo, robuste aux zéros |
| **XGBoost** | Gradient boosting | Features : semaine ISO + lags + rolling mean (52 sem.) |

- **XGBoost** gagne sur **52 produits** sur 84 (62%)
- **Prophet** gagne sur **22 produits** (26%)
- **Baseline** gagne sur **10 produits** (12%)

Le meilleur modèle est sélectionné par **MAPE minimale** (calculée uniquement sur les semaines en saison, y_true > 0).

### 4. Évaluation

- **MAE_in** — Mean Absolute Error (en saison, y > 0)
- **MAE_out** — Mean Absolute Error (hors saison, y = 0)
- **MAPE** — Mean Absolute Percentage Error (en saison uniquement)
- **sMAPE_all** — Symmetric MAPE (toutes les semaines, incluant les zéros)
- **Cross-validation temporelle** — split adaptatif : 20% de test arrondi à l'année la plus proche (1 à 3 ans)

---

## 🗺️ Roadmap

```
Phase 1 — Modèles Time Series        [✅ Terminé]
  ├── Nettoyage & feature engineering
  ├── Baseline, Prophet, XGBoost par produit
  ├── Grid search par modèle
  ├── Validation croisée temporelle (split adaptatif)
  └── Export des prédictions (Predictions.csv + model_win_metric.csv)

Phase 2 — Dashboard Interactif        [✅ Terminé]
  ├── Visualisation des historiques + prévisions (Plotly)
  ├── Courbes individuelles (1-4 produits) ou cumulatives (5+)
  ├── Simulateur de prix (impact CA en temps réel)
  ├── Métriques par produit (MAE, MAPE, modèle gagnant)
  └── Interface Streamlit responsive

Phase 3 — Conteneurisation            [En cours]
  ├── Makefile (install, run, train, docker)
  ├── Dockerfile
  └── Déploiement chez le maraîcher

Phase 4 — Base Clients Restaurants    [Futur]
  ├── Intégration des commandes BtoB
  ├── Clients réguliers → prédictions plus fiables
  ├── Modèles dédiés par client restaurant
  └── Optimisation logistique des livraisons
```

---

## 🛠️ Stack Technique

```
Python 3.10+
├── pandas / numpy              — manipulation des données
├── matplotlib / plotly         — visualisation
├── prophet                     — modèle Time Series (Meta)
├── xgboost                     — gradient boosting
├── scikit-learn                — métriques & preprocessing
├── streamlit                   — dashboard interactif
└── jupyter                     — notebooks d'analyse
```

---

## 📁 Structure du Projet

```
.
├── data/
│   ├── uc_orders.csv              # Commandes (privé)
│   ├── uc_order_products.csv      # Lignes de commandes (privé)
│   ├── uc_products.csv            # Catalogue produits (privé)
│   ├── Predictions.csv            # Prédictions 52 semaines (export)
│   └── model_win_metric.csv       # Meilleur modèle + métriques par produit
│
├── notebooks/
│   ├── utils.py                   # Chargement & nettoyage des données
│   ├── utils_series.py            # Time Series : split, complétion, métriques
│   ├── model_3.py                 # Modèles unifiés : Baseline + Prophet + XGBoost
│   ├── model_prophet.py           # Prophet standalone (legacy)
│   ├── model_xgboost.py           # XGBoost standalone (legacy)
│   ├── Global_process.ipynb       # Pipeline maître : train, éval, export
│   ├── Baseline_produits.ipynb    # Évaluation Baseline (legacy)
│   ├── Prophet.ipynb              # Évaluation Prophet (legacy)
│   └── XGboost.ipynb              # Évaluation XGBoost (legacy)
│
├── app.py                         # Dashboard Streamlit
├── Makefile                       # Commandes standardisées
├── requirements.txt
├── STREAMLIT_APP.md               # Documentation du dashboard
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Florian-Q/maraicherbio-prediction.git
cd maraicherbio-prediction
make install
```

> Les données ne sont pas incluses dans ce dépôt. Placez les fichiers CSV dans le dossier `data/` avant d'exécuter les notebooks.

### Commandes Makefile

```bash
make install        # Installe les dépendances Python
make run            # Lance le dashboard Streamlit (port 8501)
make train          # Exécute le pipeline d'entraînement (Global_process.ipynb)
make docker-build   # Construit l'image Docker
make docker-run     # Lance le conteneur Docker
make clean          # Nettoie les fichiers temporaires
```

---

## 🌱 Pourquoi Ce Projet ?

Ce projet n'est pas un exercice sur un dataset Kaggle — c'est une **collaboration réelle** avec un producteur bio breton. Les prédictions produites auront un **impact direct** sur sa gestion de production et la réduction du gaspillage alimentaire.

---

## 📄 Licence

Ce projet est à usage privé. Les données appartiennent au maraîcher partenaire et ne peuvent être redistribuées.
