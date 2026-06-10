"""
Fonctions utilitaires pour le projet de prédiction maraîchère.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from dateutil.relativedelta import relativedelta


def charger_dataframe(model_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Charge et nettoie les données maraîchères, puis retourne un DataFrame prêt
    pour l'analyse et la modélisation.

    Parameters
    ----------
    model_filter : str, optional
        Filtre optionnel sur la colonne 'model' pour retourner uniquement les lignes
        correspondant à ce modèle de produit. Si None, retourne tous les produits.

    Returns
    -------
    pd.DataFrame
        DataFrame nettoyé avec les colonnes :
        - order_id, nid, qty, weight
        - created (datetime)
        - model (nom du produit)
        - weight_units (kg, Pièce, Botte, panier)
        - quantite_y (quantité cible = weight x qty)
    """
    # --- Chargement des CSV ---
    data_path = Path(__file__).parent.parent / "data"
    order_products = pd.read_csv(data_path / "uc_order_products.csv", decimal=",", na_filter=False)
    orders = pd.read_csv(data_path / "uc_orders.csv", decimal=",", na_filter=False)
    uc_products = pd.read_csv(data_path / "uc_products.csv", decimal=",", na_filter=False)

    # --- Conversions numériques ---
    order_products = order_products.copy()
    order_products["price"] = pd.to_numeric(order_products["price"], errors="coerce")
    order_products["qty"]   = pd.to_numeric(order_products["qty"],   errors="coerce")
    order_products["weight"] = pd.to_numeric(order_products["weight"], downcast="float")

    orders = orders.copy()
    orders["created"]  = pd.to_datetime(orders["FROM_UNIXTIME(created)"])
    orders["modified"] = pd.to_datetime(orders["FROM_UNIXTIME(modified)"])

    uc_products = uc_products.copy()
    uc_products["sell_price"] = pd.to_numeric(uc_products["sell_price"], errors="coerce")

    # --- Nettoyage des colonnes ---
    uc_products = uc_products[["nid", "model", "weight_units", "weight", "sell_price"]]

    # Renommage des unités
    uc_products["weight_units"] = uc_products["weight_units"].replace({
        "oz": "Pièce",
        "lb": "Botte",
    })

    # Colonnes utiles pour la jointure
    uc_products_clean = uc_products[["nid", "model", "weight_units"]]
    orders_clean = orders[["order_id", "created"]]
    order_products_clean = order_products[["order_id", "nid", "qty", "weight"]]

    # --- Jointures (merge) ---
    df = order_products_clean.merge(orders_clean, on="order_id", how="inner")
    df = df.merge(uc_products_clean, on="nid", how="inner")

    # --- Uniformisation des noms de produits ---
    df["model"] = df["model"].replace({
        "tomate cerise": "Tomates cerises",
    })
    
    # Filtre spécifique pour certain model qui on des valeurs problèmatique
    # Filtre spécifique pour Pain campagne 500g
    model_mask = df['model'] == 'Pain campagne 500g'
    df = df[~model_mask | (df['created'].dt.year >= 2024)]
    
    # Filtre spécifique pour Cidre demi-sec
    model_mask = df['model'] == 'Cidre demi-sec'
    df = df[~model_mask | (df['created'].dt.year >= 2022)]
    
    # Filtre spécifique pour Pomme de terre alix
    model_mask = df['model'] == 'Pomme de terre alix'
    df = df[~model_mask | (df['created'].dt.year >= 2024)]
    
    # Filtre spécifique pour Carotte de terre
    model_mask = df['model'] == 'Carotte de terre'
    df = df[~model_mask | (df['created'].dt.year >= 2021)]
    
    # Filtre spécifique pour Courge Musquée
    model_mask = df['model'] == 'Courge Musquée'
    df = df[~model_mask | (df['created'].dt.year >= 2023)]
    
    # Filtre spécifique pour chicorée frisé
    model_mask = df['model'] == 'chicorée frisé'
    df = df[~model_mask | (df['created'].dt.year >= 2020)]
    
    # Filtre spécifique pour Roquette
    model_mask = df['model'] == 'Roquette'
    df = df[~model_mask | (df['created'] >= '2020-08-01')]
    
    # Filtre spécifique pour Chou rouge
    model_mask = df['model'] == 'Chou rouge'
    df = df[~model_mask | (df['created'] >= '2023-11-01')]
    
    # Filtre spécifique pour Pain multigraines 1kg
    model_mask = df['model'] == 'Pain multigraines 1kg'
    df = df[~model_mask | (df['created'].dt.year >= 2024)]
    
        # Filtre spécifique pour Celeri boule
    model_mask = df['model'] == 'Celeri boule'
    df = df[~model_mask | (df['created'] >= '2021-08-01')]
    
    # Filtre spécifique pour Jeune pousse epinard
    model_mask = df['model'] == 'Jeune pousse epinard'
    df = df[~model_mask | (df['created'].dt.year >= 2022)]
    
    # Filtre spécifique pour Radis noir rond
    model_mask = df['model'] == 'Radis noir rond'
    df = df[~model_mask | (df['created'] >= '2023-09-01')]
    
    # Filtre spécifique pour Betterave botte
    model_mask = df['model'] == 'Betterave botte'
    df = df[~model_mask | (df['created'].dt.year >= 2018)]
    
    # Filtre spécifique pour chou de Bruxelles
    model_mask = df['model'] == 'chou de Bruxelles'
    df = df[~model_mask | (df['created'] >= '2023-12-01')]
    
    # Filtre spécifique pour Rutabaga
    model_mask = df['model'] == 'Rutabaga'
    df = df[~model_mask | (df['created'] >= '2023-09-01')]
    
    # Filtre spécifique pour Plant de tomate 
    model_mask = df['model'] == 'Plant de tomate '
    df = df[~model_mask | (df['created'].dt.year >= 2018)]
    
    # Filtre spécifique pour radis violet
    model_mask = df['model'] == 'radis violet'
    df = df[~model_mask | (df['created'] >= '2023-10-01')]
    

    # --- Correction des poids pour certains aliments mal annotés ---
    aliments = ['Celeri boule', 'Pomme de terre aniel', 'Pomme de terre Désiré',
            'Petit pois', 'Kiwi jaune', 'Prune', 'Betterave chioggia',
            'Betterave Jaune', 'Carotte de terre']

    mask_alim = df["model"].isin(aliments) & (df["weight"] < 100)
    df.loc[mask_alim, "weight"] = df.loc[mask_alim, "weight"] * 1000

    # Supprimer les lignes Kiwi jaune avec weight = 5.0 (probable erreur de saisie)
    df = df[~((df["model"] == "Kiwi jaune") & (df["weight"] == 5.0))]

    # Remplacer weight = 4.0 par 2000 pour Kiwi (probable erreur de saisie)
    df.loc[(df["model"] == "Kiwi") & (df["weight"] == 4.0), "weight"] = 2000

    # --- Conversion grammes → kilogrammes ---
    mask_g = df["weight_units"] == "g"
    df.loc[mask_g, "weight"] = df.loc[mask_g, "weight"] / 1000
    df.loc[mask_g, "weight_units"] = "kg"

    # --- Forcer weight = 1 pour les unités à la pièce / botte ---
    mask_piece_botte = df["weight_units"].isin(["Botte", "Pièce"]) & (df["weight"] > 1)
    df.loc[mask_piece_botte, "weight"] = 1

    # --- Quantité cible ---
    df["quantite_y"] = df["weight"] * df["qty"]

    # --- Tri par date ---
    df = df.sort_values("created").reset_index(drop=True)

    # supprime 13 ligne avec des quantités = 0
    df = df[df["qty"] > 0]

    # Selection products - vérifier les 3 dernières années avec continuité annuelle
    last_date = df["created"].max()
    
    # Définir les 3 périodes annuelles
    period_1_end = last_date
    period_1_start = last_date - relativedelta(years=1)
    
    period_2_end = period_1_start
    period_2_start = period_2_end - relativedelta(years=1)
    
    period_3_end = period_2_start
    period_3_start = period_3_end - relativedelta(years=1)
    
    # Compter les ventes par produit dans chaque période
    df_p1 = df[(df["created"] > period_1_start) & (df["created"] <= period_1_end)]
    df_p2 = df[(df["created"] > period_2_start) & (df["created"] <= period_2_end)]
    df_p3 = df[(df["created"] > period_3_start) & (df["created"] <= period_3_end)]
    
    products_p1 = set(df_p1[df_p1["model"].isin(df_p1["model"].value_counts()[df_p1["model"].value_counts() > 0].index)]["model"].unique())
    products_p2 = set(df_p2[df_p2["model"].isin(df_p2["model"].value_counts()[df_p2["model"].value_counts() > 0].index)]["model"].unique())
    products_p3 = set(df_p3[df_p3["model"].isin(df_p3["model"].value_counts()[df_p3["model"].value_counts() > 0].index)]["model"].unique())
    
    # Garder seulement les produits présents dans les 3 périodes ET avec au total plus de 10 ventes
    products_in_all_periods = products_p1 & products_p2 & products_p3
    
    df_last_3_years = df[df["created"] >= period_3_start]
    products_select = df_last_3_years[df_last_3_years["model"].isin(products_in_all_periods)]["model"].value_counts().reset_index()
    products_select.columns = ["model", "count"]
    
    # Vérifier au minimum 10 ventes au total sur les 3 ans
    products_select = products_select[products_select['count'] > 10]

    # garde que les ventes qui on les produits selectionnés
    df = df[df["model"].isin(products_select["model"].tolist())]

    # --- Garde uniquement les ventes à partir du 01/01/2014 ---
    df = df[df["created"] >= "2014-01-01"]

    # --- Filtre optionnel sur le modèle ---
    if model_filter is not None:
        df = df[df["model"] == model_filter]

    return df
