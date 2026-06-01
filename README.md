# 🥦 Prédiction des Ventes — Maraîcher Bio Lorient

> Projet Data Science — Séries Temporelles · Modèles Prédictifs · Dashboard

---

## 📌 Contexte

Ce projet est réalisé en partenariat avec un **maraîcher bio basé à Lorient (Bretagne)** qui a accordé un accès à sa base de données de ventes en ligne. L'objectif est de prédire les quantités de légumes et fruits vendus chaque semaine, afin d'aider le maraîcher à **anticiper sa production, réduire le gaspillage et optimiser ses stocks**.

Choisir ce projet, c'est aussi choisir de **soutenir une agriculture locale et biologique**.

---

## 🎯 Objectif

Construire un ou plusieurs modèles de **séries temporelles supervisés** capables de prédire les quantités hebdomadaires vendues pour chaque produit.

| Paramètre | Valeur |
|---|---|
| **Type of Challenge** | Model Training — Supervisé |
| **Variable X** | Quantités de légumes/fruits vendues par semaine |
| **Variable Y** | Prédiction des quantités futures |
| **Type de valeur** | Values (continue) |
| **ML Approach** | Time Series — Modèle **Prophet** (Meta) |
| **Exploration & Viz** | Matplotlib · Plotly · Dashboard interactif |

---

## 🗃️ Données

> ⚠️ Les données sont privées et ne sont pas versionnées dans ce dépôt.

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

**Volume de données :**
- ~142 600 lignes de ventes produit
- ~21 900 commandes
- Historique de **plus de 10 ans** (2013 – 2025)
- **300+ produits** référencés

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

### 2. Un Modèle par Légume/Fruit
Chaque produit ayant ses propres cycles saisonniers, un **modèle dédié** est entraîné par produit pour garantir une précision maximale par rapport à un modèle global.

### 3. Modèle Prophet (Meta)
[Prophet](https://facebook.github.io/prophet/) est particulièrement adapté à ce cas d'usage :
- Conçu pour les séries avec **forte saisonnalité** (hebdomadaire, annuelle)
- Robuste aux **données manquantes** et aux valeurs aberrantes
- Intègre les **jours fériés** et événements ponctuels
- Interprétable et facilement ajustable

### 4. Évaluation
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Square Error
- **MAPE** — Mean Absolute Percentage Error
- **Cross-validation temporelle** (walk-forward validation)

---

## 🗺️ Roadmap

```
Phase 1 — Modèles Time Series        [En cours]
  ├── Nettoyage & feature engineering
  ├── Modèle Prophet par produit
  ├── Validation croisée temporelle
  └── Export des prédictions hebdomadaires

Phase 2 — Dashboard Interactif        [Planifié]
  ├── Visualisation des quantités prédites
  ├── Comparaison saison N vs N-1
  ├── Alertes de stock
  └── Interface accessible au maraîcher

Phase 3 — Base Clients Restaurants    [Futur]
  ├── Intégration des commandes BtoB
  ├── Clients réguliers → prédictions plus fiables
  ├── Modèles dédiés par client restaurant
  └── Optimisation logistique des livraisons
```

---

## 🛠️ Stack Technique

```
Python 3.11+
├── pandas / numpy          — manipulation des données
├── matplotlib / plotly     — visualisation
├── prophet                 — modèle Time Series
├── scikit-learn            — métriques & preprocessing
└── dash / streamlit        — dashboard interactif (Phase 2)
```

---

## 📁 Structure du Projet

```
.
├── data/
│   └── .gitkeep            # données privées — non versionnées
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling_prophet.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── outputs/
│   └── charts/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<votre-repo>/maraicher-prediction.git
cd maraicher-prediction
pip install -r requirements.txt
```

> Les données ne sont pas incluses dans ce dépôt. Placez vos fichiers dans le dossier `data/` avant d'exécuter les notebooks.

---

## 🌱 Pourquoi Ce Projet ?

Ce projet n'est pas un exercice sur un dataset Kaggle — c'est une **collaboration réelle** avec un producteur bio breton. Les prédictions produites auront un **impact direct** sur sa gestion de production et la réduction du gaspillage alimentaire.

---

## 📄 Licence

Ce projet est à usage privé. Les données appartiennent au maraîcher partenaire et ne peuvent être redistribuées.