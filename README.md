# 🥦 BioPredict — Prédiction des ventes d'un maraîcher bio (Bretagne)

![Statut](https://img.shields.io/badge/Statut-Projet%20termin%C3%A9-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white) [![Démo](https://img.shields.io/badge/D%C3%A9mo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://maraicherbio-prediction.streamlit.app/)

> Projet de Data Science — Séries temporelles · Modélisation prédictive · Dashboard interactif
> Réalisé en équipe de 3 en deux semaines, en partenariat avec un maraîcher bio réel.

**🔗 Démo en ligne : [maraicherbio-prediction.streamlit.app](https://maraicherbio-prediction.streamlit.app/)** — aucune installation nécessaire, les données affichées ont été adaptées pour la consultation publique (voir [Données & confidentialité](#-données--confidentialité)).

---

## En bref

- **Problème métier** : un maraîcher bio ne sait pas à l'avance combien de légumes il vendra chaque semaine, ce qui complique la planification des récoltes et génère du gaspillage.
- **Données** : 12 ans de ventes réelles (2014–2026), 142 600 lignes de vente, 21 900 commandes → 84 produits retenus après nettoyage.
- **Approche** : un modèle de prédiction dédié par produit (Baseline, Prophet, XGBoost), sélectionné automatiquement selon sa précision.
- **Résultat** : ~30 % de MAPE en moyenne, jusqu'à 25 % sur les produits à historique stable.
- **Livrable** : un dashboard Streamlit interactif, déployé publiquement, avec simulateur de prix en temps réel.

---

## 📌 Contexte

Ce projet a été réalisé en partenariat avec un **maraîcher bio basé en Bretagne**, qui nous a donné accès à l'historique de ventes de sa boutique en ligne. Sans visibilité sur ses ventes futures, il devait récolter "à l'aveugle" — au risque de jeter des dizaines de kilos de légumes invendus ou d'en manquer en pleine saison.

L'objectif : exploiter 12 ans d'historique pour **prédire les quantités vendues, produit par produit et semaine par semaine**, afin de l'aider à anticiper sa production, réduire le gaspillage et optimiser ses stocks.

Au-delà de l'exercice technique, c'est une collaboration réelle avec un producteur bio breton : les prédictions livrées ont un usage concret pour sa gestion de production.

---

## 🗃️ Données

Le projet s'appuie sur l'export de la boutique en ligne du maraîcher.

**Volumes :** ~142 600 lignes de vente · ~21 900 commandes · historique 2014 → 2026.

**Nettoyage effectué :**
- **Filtrage** — conservation des produits avec au moins 3 ans d'historique continu et plus de 10 ventes (300 → 84 produits retenus)
- **Conversion des unités** — grammes → kilogrammes, harmonisation des unités hétérogènes ("botte", "pièce")
- **Agrégation hebdomadaire** — les ventes, enregistrées commande par commande, sont regroupées par semaine : c'est l'échelle à laquelle le maraîcher planifie ses récoltes, et donc celle à laquelle la prédiction est utile
- **Gestion des semaines sans vente** et des valeurs aberrantes

> ⚠️ Les fichiers de données bruts (CSV) ne sont jamais commités dans ce dépôt (voir `.gitignore`). Les données originales appartiennent au maraîcher partenaire et restent confidentielles — voir [Données & confidentialité](#-données--confidentialité).

### 📊 Quelques insights

- 🥇 La **courgette** est le produit n°1, avec 1 039 unités vendues en 2025 — soit +46 % par rapport au second produit
- 🍅 Top 5 annuel : courgette · tomate rouge ronde · poireau · aubergine noire · pomme de terre allians
- ☀️ Forte **saisonnalité estivale** : courgette, tomate et aubergine explosent de mai à septembre
- ❄️ **Poireau et pomme de terre** portent les ventes hivernales (octobre → avril)
- 🥔 Certains produits (ex. jeunes pousses, variétés très ponctuelles) ont un historique trop irrégulier pour être prédits de façon fiable, même par les meilleurs modèles — une limite assumée plutôt que masquée

---

## 🔬 Approche technique

### Un modèle par produit

Chaque légume a son propre cycle de vente (saisonnalité, durée de vie commerciale, régularité). Plutôt qu'un modèle global, **un modèle dédié est entraîné pour chacun des 84 produits**, ce qui maximise la précision par rapport à une approche unique.

Il s'agit formellement d'un problème de régression supervisée sur séries temporelles : à partir de l'historique hebdomadaire d'un produit (X), prédire les quantités vendues sur les 52 semaines suivantes (Y).

### Trois modèles en compétition

| Modèle | Type | Caractéristiques |
|---|---|---|
| **Baseline** | Moyenne hebdomadaire | Moyenne + écart-type par semaine ISO — sert de référence |
| **Prophet** (Meta) | Séries temporelles bayésien | Saisonnalité annuelle + hebdomadaire, robuste aux zéros et au bruit |
| **XGBoost** | Gradient boosting | Features : semaine ISO, lags, moyenne glissante (52 sem.) |

Pour chaque produit, le meilleur modèle est sélectionné automatiquement par **MAPE minimale** (calculée uniquement sur les semaines en saison). Aucun modèle ne domine sur tous les produits — ce qui confirme la pertinence de l'approche "un modèle par produit" : la simplicité de la Baseline suffit pour certains, tandis que XGBoost ou Prophet font la différence sur d'autres.

### Validation

- **Cross-validation temporelle adaptative** : la période de test (~20 %) s'ajuste à l'ancienneté de chaque produit (1 à 3 ans), au lieu d'un split fixe
- **Métriques** : MAE en saison / hors saison, MAPE (en saison), sMAPE sur toutes les semaines (zéros inclus)

---

## 📈 Résultats

- **~30 % de MAPE en moyenne** sur l'ensemble des 84 produits
- Sur les produits à historique stable, la précision descend nettement en dessous de la moyenne : de **25 % à 32 %** de MAPE pour les 5 meilleurs (ex. carotte de terre, tomate ancienne, poireau, courgette)
- Répartition du modèle gagnant sur les 84 produits :

  | Modèle | Produits gagnés | Part |
  |---|---|---|
  | **XGBoost** | 52 | 62 % |
  | **Prophet** | 22 | 26 % |
  | **Baseline** | 10 | 12 % |

---

## 🖥️ Dashboard interactif

Le pipeline de prédiction est exposé via un dashboard **Streamlit** déployé publiquement : **[maraicherbio-prediction.streamlit.app](https://maraicherbio-prediction.streamlit.app/)**

Ce qu'on peut y faire :
- Visualiser l'historique et les prévisions (52 semaines, intervalle de confiance à 95 %) pour un ou plusieurs produits
- Basculer entre vue par quantité et vue par chiffre d'affaires
- **Simuler un changement de prix** et voir l'impact sur le CA prévisionnel en temps réel
- Comparer le modèle gagnant et les métriques (MAE, MAPE) produit par produit
- Rechercher, sélectionner et filtrer parmi les 84 produits

---

## 🛠️ Stack technique

```
Python 3.10+
├── pandas / numpy              — manipulation des données
├── matplotlib / plotly         — visualisation
├── prophet                     — modèle de séries temporelles (Meta)
├── xgboost / lightgbm          — gradient boosting
├── scikit-learn                — métriques & preprocessing
├── streamlit                   — dashboard interactif
└── jupyter                     — notebooks d'exploration
```

---

## ✅ Bilan du projet

**Livré :**
- Pipeline de nettoyage, feature engineering et entraînement (Baseline, Prophet, XGBoost) sur 84 produits
- Sélection automatique du meilleur modèle par produit, validation croisée temporelle adaptative
- Dashboard interactif déployé publiquement, avec simulateur de prix et KPIs en temps réel

**Pistes d'évolution identifiées (non réalisées à ce stade) :**
- Conteneurisation (Docker) et déploiement directement chez le maraîcher
- Intégration de nouvelles données : clients professionnels / restaurants
- Étude de l'impact du changement climatique sur la saisonnalité
- Corrélation entre quantités vendues et surfaces cultivées

---

## 👥 Équipe

Projet réalisé en deux semaines par une équipe de 3, dans le cadre d'une formation Data Science :

- **Florian Quintin**
- **Habiba Jouan**
- **Matis Rocher**

---

## 📄 Données & confidentialité

Le code de ce dépôt (pipeline, modèles, dashboard) est partagé à titre de démonstration technique.

Les données de vente originales appartiennent au maraîcher partenaire et sont strictement confidentielles : les fichiers bruts n'ont jamais été commités dans ce dépôt. Les données affichées dans la démo publique ont été adaptées afin de pouvoir être présentées sans exposer d'informations commerciales réelles.
