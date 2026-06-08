"""
Dashboard MaraîcherBio — Application Streamlit
Exploration des ventes historiques et prévisions hebdomadaires par produit.

Usage :
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ajouter le dossier notebooks/ au path pour les imports
NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional, Tuple

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="Dashboard MaraîcherBio",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

# Tentative d'import des fonctions du projet — utilise les données réelles si disponibles
_DATA_REAL_AVAILABLE = False
try:
    import utils
    import utils_series

    # Vérifier que les CSV sont présents
    if (NOTEBOOKS_DIR.parent / "data" / "uc_order_products.csv").exists():
        _DATA_REAL_AVAILABLE = True
except Exception:
    pass


def _generate_demo_data() -> pd.DataFrame:
    """
    Génère un jeu de données fictif pour démonstration.
    Produits avec saisonnalité réaliste : légumes d'été et d'hiver.

    Returns
    -------
    pd.DataFrame avec colonnes : date, produit, quantite, prix, unite, modele, mae, mape
    """
    np.random.seed(42)

    products = {
        "Courgette": {
            "prix": 2.50,
            "unite": "kg",
            "saison": "ete",
            "amplitude": 40,
            "base": 25,
            "tendance": 0.08,
            "mae": 4.2,
            "mape": 22.5,
        },
        "Tomate rouge ronde": {
            "prix": 3.00,
            "unite": "kg",
            "saison": "ete",
            "amplitude": 50,
            "base": 30,
            "tendance": 0.05,
            "mae": 5.1,
            "mape": 18.3,
        },
        "Poireau": {
            "prix": 2.20,
            "unite": "kg",
            "saison": "hiver",
            "amplitude": 30,
            "base": 20,
            "tendance": 0.03,
            "mae": 3.8,
            "mape": 25.0,
        },
        "Aubergine noire": {
            "prix": 2.80,
            "unite": "kg",
            "saison": "ete",
            "amplitude": 25,
            "base": 15,
            "tendance": 0.10,
            "mae": 3.2,
            "mape": 28.1,
        },
        "Pomme de terre Allians": {
            "prix": 1.80,
            "unite": "kg",
            "saison": "hiver",
            "amplitude": 35,
            "base": 28,
            "tendance": -0.02,
            "mae": 5.5,
            "mape": 15.7,
        },
        "Tomates cerises": {
            "prix": 3.50,
            "unite": "kg",
            "saison": "ete",
            "amplitude": 35,
            "base": 20,
            "tendance": 0.12,
            "mae": 2.8,
            "mape": 30.5,
        },
        "Carotte de terre": {
            "prix": 1.90,
            "unite": "kg",
            "saison": "hiver",
            "amplitude": 22,
            "base": 18,
            "tendance": 0.01,
            "mae": 2.5,
            "mape": 20.0,
        },
        "Betterave chioggia": {
            "prix": 2.10,
            "unite": "kg",
            "saison": "ete",
            "amplitude": 18,
            "base": 12,
            "tendance": 0.06,
            "mae": 1.9,
            "mape": 35.0,
        },
        "Salade verte": {
            "prix": 1.50,
            "unite": "Pièce",
            "saison": "ete",
            "amplitude": 30,
            "base": 25,
            "tendance": 0.04,
            "mae": 6.0,
            "mape": 22.0,
        },
        "Epinard": {
            "prix": 2.00,
            "unite": "Botte",
            "saison": "hiver",
            "amplitude": 15,
            "base": 10,
            "tendance": 0.02,
            "mae": 2.3,
            "mape": 40.0,
        },
        "Navet": {
            "prix": 1.70,
            "unite": "kg",
            "saison": "hiver",
            "amplitude": 20,
            "base": 14,
            "tendance": -0.01,
            "mae": 1.8,
            "mape": 32.1,
        },
        "Haricot vert": {
            "prix": 3.20,
            "unite": "kg",
            "saison": "ete",
            "amplitude": 28,
            "base": 18,
            "tendance": 0.07,
            "mae": 3.5,
            "mape": 24.8,
        },
    }

    records = []
    start_date = datetime(2021, 1, 3)  # Premier dimanche de 2021
    end_date = datetime(2026, 5, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq="W-SUN")

    for product, cfg in products.items():
        for i, d in enumerate(dates):
            week_of_year = d.isocalendar().week
            year_progress = week_of_year / 52.0

            if cfg["saison"] == "ete":
                seasonal = cfg["amplitude"] * np.sin(np.pi * (year_progress - 0.15)) * max(0, np.sin(np.pi * year_progress))
            else:
                seasonal = cfg["amplitude"] * np.sin(np.pi * (year_progress + 0.35)) * max(0, np.sin(np.pi * (year_progress + 0.1)))

            seasonal = max(seasonal, -cfg["base"] * 0.3)
            trend = cfg["tendance"] * i
            noise = np.random.normal(0, cfg["base"] * 0.15)
            quantite = max(0, cfg["base"] + seasonal + trend + noise)
            quantite = round(quantite, 2)

            records.append({
                "date": d,
                "produit": product,
                "quantite": quantite,
                "prix": cfg["prix"],
                "unite": cfg["unite"],
                "modele": "Baseline (moy. hebdo)",
                "mae": cfg["mae"],
                "mape": cfg["mape"],
            })

    return pd.DataFrame(records)


def _load_real_prices() -> dict:
    """
    Charge les derniers prix référencés depuis uc_products.csv.
    Retourne un mapping model -> sell_price.
    """
    try:
        data_path = Path(__file__).parent / "data"
        products = pd.read_csv(data_path / "uc_products.csv", decimal=",")
        products["sell_price"] = pd.to_numeric(products["sell_price"], errors="coerce")
        # Dernier prix par produit (au cas où il y aurait des doublons)
        return products.dropna(subset=["sell_price"]).groupby("model")["sell_price"].last().to_dict()
    except Exception:
        return {}


def _get_product_price_map(df_all: pd.DataFrame) -> dict:
    """Construit un mapping produit -> prix à partir du DataFrame complet."""
    if "prix" in df_all.columns:
        return df_all.groupby("produit")["prix"].first().to_dict()
    return {}


def _get_product_unit_map(df_all: pd.DataFrame) -> dict:
    """Construit un mapping produit -> unité."""
    if "unite" in df_all.columns:
        return df_all.groupby("produit")["unite"].first().to_dict()
    return {}


# ============================================================
# 3. MOTEUR DE PRÉDICTION (Baseline : moyenne hebdomadaire)
# ============================================================

def train_baseline_model(ts: pd.Series) -> dict:
    """
    Entraîne un modèle baseline : moyenne et écart-type par semaine ISO.

    Parameters
    ----------
    ts : pd.Series avec DateTimeIndex et valeurs numériques

    Returns
    -------
    dict avec clés : weekly_mean (Series), weekly_std (Series)
    """
    weekly_mean = ts.groupby(ts.index.isocalendar().week).mean()
    weekly_std = ts.groupby(ts.index.isocalendar().week).std().fillna(0)
    return {"weekly_mean": weekly_mean, "weekly_std": weekly_std}


def generate_forecast(
    model: dict,
    last_date: pd.Timestamp,
    n_weeks: int = 52,
) -> pd.DataFrame:
    """
    Génère les prévisions hebdomadaires pour n_weeks à partir de last_date.

    Parameters
    ----------
    model : dict
        Sortie de train_baseline_model (weekly_mean, weekly_std)
    last_date : pd.Timestamp
        Dernière date connue dans l'historique
    n_weeks : int
        Nombre de semaines à prédire (défaut 52)

    Returns
    -------
    pd.DataFrame avec colonnes : date, prediction, borne_inf, borne_sup
    """
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=n_weeks,
        freq="W-SUN",
    )
    weeks = forecast_dates.isocalendar().week

    preds = weeks.map(model["weekly_mean"]).fillna(0).values
    stds = weeks.map(model["weekly_std"]).fillna(0).values

    return pd.DataFrame({
        "date": forecast_dates,
        "prediction": np.clip(preds, 0, None),
        "borne_inf": np.clip(preds - 1.96 * stds, 0, None),
        "borne_sup": preds + 1.96 * stds,
    })


# ============================================================
# 4. PRÉPARATION DES DONNÉES PRODUIT
# ============================================================

def prepare_product_data(df_demo: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prépare les données pour tous les produits :
    - Agrège l'historique par semaine
    - Entraîne le modèle baseline
    - Génère les prévisions sur 52 semaines

    Returns
    -------
    (df_forecast_all, df_summary) : Tuple[pd.DataFrame, pd.DataFrame]
        df_forecast_all : historique + prévisions combinées pour tous les produits
        df_summary : résumé par produit (CA total, volume, croissance, etc.)
    """
    products = sorted(df_demo["produit"].unique())
    price_map = _get_product_price_map(df_demo)
    unit_map = _get_product_unit_map(df_demo)

    all_forecasts = []
    summaries = []

    for product in products:
        df_p = df_demo[df_demo["produit"] == product].copy()
        df_p = df_p.sort_values("date")

        # Agréger par semaine (somme des quantités)
        df_p["semaine"] = df_p["date"].dt.to_period("W-SUN").dt.start_time + pd.offsets.Week(weekday=6)
        weekly = df_p.groupby("semaine")["quantite"].sum()
        weekly.index = pd.DatetimeIndex(weekly.index)

        if len(weekly) < 10:
            continue

        # Entraîner modèle baseline sur tout l'historique
        model = train_baseline_model(weekly)
        last_date = weekly.index.max()

        # Générer prévisions 52 semaines
        forecast = generate_forecast(model, last_date, n_weeks=52)

        # Construire le DataFrame produit complet (historique + prévisions)
        prix_base = price_map.get(product, 0)
        unite = unit_map.get(product, "")
        modele = df_p["modele"].iloc[0] if "modele" in df_p.columns else "Baseline (moy. hebdo)"
        mae = df_p["mae"].iloc[0] if "mae" in df_p.columns else 0.0
        mape = df_p["mape"].iloc[0] if "mape" in df_p.columns else 0.0

        # Partie historique
        hist_df = pd.DataFrame({
            "date": weekly.index,
            "produit": product,
            "quantite_reelle": weekly.values,
            "prediction": np.nan,
            "prix": prix_base,
            "unite": unite,
            "modele": modele,
            "mae": mae,
            "mape": mape,
            "borne_inf": np.nan,
            "borne_sup": np.nan,
        })

        # Partie prévisions
        fut_df = pd.DataFrame({
            "date": forecast["date"],
            "produit": product,
            "quantite_reelle": np.nan,
            "prediction": forecast["prediction"],
            "prix": prix_base,
            "unite": unite,
            "modele": modele,
            "mae": mae,
            "mape": mape,
            "borne_inf": forecast["borne_inf"],
            "borne_sup": forecast["borne_sup"],
        })

        combined = pd.concat([hist_df, fut_df], ignore_index=True)
        all_forecasts.append(combined)

        # Résumé pour le portefeuille
        total_qty_hist = weekly.sum()
        total_qty_forecast = forecast["prediction"].sum()
        total_ca_hist = total_qty_hist * prix_base
        total_ca_forecast = total_qty_forecast * prix_base
        growth = ((total_qty_forecast - total_qty_hist / max(len(weekly) / 52, 1))
                   / (total_qty_hist / max(len(weekly) / 52, 1))) * 100 if total_qty_hist > 0 else 0

        summaries.append({
            "produit": product,
            "prix": prix_base,
            "unite": unite,
            "volume_total_kg": round(total_qty_hist, 1),
            "ca_historique": round(total_ca_hist, 2),
            "ca_previsionnel": round(total_ca_forecast, 2),
            "volume_previsionnel": round(total_qty_forecast, 1),
            "croissance_pct": round(growth, 1),
            "mae": mae,
            "mape": mape,
        })

    df_forecast_all = pd.concat(all_forecasts, ignore_index=True)
    df_summary = pd.DataFrame(summaries)

    return df_forecast_all, df_summary


# ============================================================
# 5. GRAPHIQUES PLOTLY
# ============================================================

def plot_history_forecast(
    df_product: pd.DataFrame,
    metric_mode: str = "quantite",
    simulated_price: Optional[float] = None,
) -> go.Figure:
    """
    Graphique interactif : historique réel + prévisions.

    Parameters
    ----------
    df_product : pd.DataFrame
        Données filtrées pour un seul produit (colonnes : date, quantite_reelle,
        prediction, borne_inf, borne_sup, prix)
    metric_mode : str
        "quantite" ou "ca" (chiffre d'affaires)
    simulated_price : float, optional
        Prix simulé pour le calcul du CA

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = df_product.sort_values("date").copy()
    price = simulated_price if simulated_price is not None else df["prix"].iloc[0]

    # Séparation historique / prévisions
    hist_mask = df["quantite_reelle"].notna()
    fut_mask = df["prediction"].notna()

    # Facteur de conversion selon le mode
    if metric_mode == "ca":
        factor = price
        y_label = "Chiffre d'affaires (€)"
    else:
        factor = 1
        y_label = "Quantité vendue"

    fig = go.Figure()

    # Courbe noire : historique
    df_hist = df[hist_mask]
    fig.add_trace(go.Scatter(
        x=df_hist["date"],
        y=df_hist["quantite_reelle"] * factor,
        mode="lines",
        name="Historique réel",
        line=dict(color="black", width=2),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>"
        + ("%{y:.2f} €" if metric_mode == "ca" else "%{y:.1f}")
        + "<extra></extra>",
    ))

    # Bande d'incertitude
    df_fut = df[fut_mask]
    if len(df_fut) > 0:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_fut["date"], df_fut["date"][::-1]]),
            y=pd.concat([
                df_fut["borne_inf"] * factor,
                df_fut["borne_sup"][::-1] * factor,
            ]),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.12)",
            line=dict(color="rgba(255,0,0,0)"),
            name="Intervalle de confiance (95%)",
            showlegend=True,
            hoverinfo="skip",
        ))

        # Courbe rouge : prévisions
        fig.add_trace(go.Scatter(
            x=df_fut["date"],
            y=df_fut["prediction"] * factor,
            mode="lines",
            name="Prévisions",
            line=dict(color="red", width=2, dash="solid"),
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>"
            + ("%{y:.2f} €" if metric_mode == "ca" else "%{y:.1f}")
            + "<br>IC: [%{customdata[0]:.1f}, %{customdata[1]:.1f}]"
            + "<extra></extra>",
            customdata=np.column_stack([
                df_fut["borne_inf"] * factor,
                df_fut["borne_sup"] * factor,
            ]),
        ))

    # Ligne verticale de séparation
    last_hist_date = df_hist["date"].max()
    if pd.notna(last_hist_date):
        y_min = 0
        y_all = pd.concat([
            df_hist["quantite_reelle"] * factor,
            df_fut["prediction"] * factor if len(df_fut) > 0 else pd.Series([0]),
        ])
        y_max = y_all.max() * 1.1

        fig.add_shape(
            type="line",
            x0=last_hist_date, x1=last_hist_date,
            y0=y_min, y1=y_max,
            line=dict(color="gray", width=1, dash="dash"),
            name="Aujourd'hui",
        )
        fig.add_annotation(
            x=last_hist_date, y=y_max,
            text="Aujourd'hui",
            showarrow=False,
            yshift=10,
            font=dict(size=10, color="gray"),
        )

    # Layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
        dragmode="pan",
    )
    fig.update_xaxes(rangeslider_visible=False)

    return fig


def plot_top10_bar(
    df_summary: pd.DataFrame,
    metric: str = "ca_historique",
    title: str = "Top 10 produits",
) -> go.Figure:
    """
    Diagramme en barres horizontal pour le top 10.

    Parameters
    ----------
    df_summary : pd.DataFrame
        Résumé par produit
    metric : str
        Colonne à utiliser pour le classement
    title : str
        Titre du graphique

    Returns
    -------
    go.Figure
    """
    df_top = df_summary.nlargest(10, metric).sort_values(metric, ascending=True)

    color_map = {
        "ca_historique": "#1f77b4",
        "ca_previsionnel": "#ff7f0e",
        "volume_total_kg": "#2ca02c",
        "croissance_pct": "#d62728",
    }
    color = color_map.get(metric, "#1f77b4")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_top["produit"],
        x=df_top[metric],
        orientation="h",
        marker=dict(color=color, opacity=0.85),
        text=df_top[metric].apply(
            lambda v: f"{v:,.0f} €" if "ca" in metric else
            f"{v:,.1f} kg" if "volume" in metric else
            f"{v:+.1f} %"
        ),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=20, r=60, t=40, b=20),
        height=350,
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
    )

    return fig


# ============================================================
# 6. COMPOSANTS STREAMLIT
# ============================================================

def render_kpi_row(df_product: pd.DataFrame, simulated_price: Optional[float]) -> None:
    """
    Affiche une ligne de KPI cards (quantité prévue, CA prévisionnel,
    gain/perte prix, prix actuel, prix simulé).
    """
    df_fut = df_product[df_product["prediction"].notna()]
    price_base = df_product["prix"].iloc[0] if len(df_product) > 0 else 0
    price_sim = simulated_price if simulated_price is not None else price_base

    qty_total = df_fut["prediction"].sum()
    ca_base = qty_total * price_base
    ca_sim = qty_total * price_sim
    delta_ca = ca_sim - ca_base
    delta_pct = (delta_ca / ca_base * 100) if ca_base > 0 else 0

    cols = st.columns(5)
    with cols[0]:
        st.metric("📦 Qté prévue (52 sem.)", f"{qty_total:,.0f}")
    with cols[1]:
        st.metric("💰 CA prévisionnel", f"{ca_sim:,.0f} €")
    with cols[2]:
        st.metric(
            "📈 Impact prix",
            f"{delta_ca:+,.0f} €",
            delta=f"{delta_pct:+.1f} %",
        )
    with cols[3]:
        st.metric("🏷️ Prix actuel", f"{price_base:.2f} €")
    with cols[4]:
        delta_price = price_sim - price_base
        st.metric(
            "🔧 Prix simulé",
            f"{price_sim:.2f} €",
            delta=f"{delta_price:+.2f} €" if delta_price != 0 else None,
        )


def render_product_card(df_product: pd.DataFrame) -> None:
    """Fiche produit (colonne droite)."""
    if len(df_product) == 0:
        st.warning("Aucune donnée produit disponible.")
        return

    product = df_product["produit"].iloc[0]
    unite = df_product["unite"].iloc[0]
    modele = df_product["modele"].iloc[0]
    mae = df_product["mae"].iloc[0]
    mape = df_product["mape"].iloc[0]
    prix = df_product["prix"].iloc[0]

    st.markdown("### 📋 Fiche produit")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Nom :** {product}")
        st.markdown(f"**Unité :** {unite or '—'}")
        st.markdown(f"**Modèle :** {modele or '—'}")
    with col_b:
        st.markdown(f"**MAE :** {mae:.2f}")
        st.markdown(f"**MAPE :** {mape:.1f} %")
        st.markdown(f"**Prix :** {prix:.2f} €")


def render_price_simulator(
    df_product: pd.DataFrame,
    current_simulated_price: Optional[float],
) -> Optional[float]:
    """
    Affiche le simulateur de prix (slider + reset).

    Returns
    -------
    float ou None : le nouveau prix simulé, ou None si reset
    """
    if len(df_product) == 0:
        return None

    prix_base = df_product["prix"].iloc[0]
    step = max(0.05, round(prix_base * 0.05, 2))

    st.markdown("---")
    st.markdown("### 💶 Simulateur de prix")

    new_price = st.slider(
        "Ajuster le prix de vente (€)",
        min_value=max(0.10, round(prix_base * 0.3, 2)),
        max_value=round(prix_base * 2.5, 2),
        value=current_simulated_price if current_simulated_price is not None else prix_base,
        step=step,
        format="%.2f €",
        key="price_slider",
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("↩️ Réinitialiser au prix réel", use_container_width=True):
            st.session_state.simulated_price = None
            st.rerun()
    with btn_col2:
        st.caption(f"Prix réel : {prix_base:.2f} €")

    return new_price if new_price != prix_base else None


# ============================================================
# 7. APPLICATION PRINCIPALE
# ============================================================

def main():
    # --- Titre ---
    st.title("🥦 Dashboard MaraîcherBio")
    st.caption(
        "Analyse des ventes historiques et prévisions hebdomadaires "
        "— outil d'aide à la décision pour la production maraîchère."
    )

    # --- Chargement des données ---
    with st.spinner("Chargement des données..."):
        if _DATA_REAL_AVAILABLE:
            try:
                # Utiliser les données réelles
                df_all = utils.charger_dataframe()
                # Récupérer prix et unités
                price_map = _load_real_prices()
                unit_map = df_all.groupby("model")["weight_units"].first().to_dict()
                # Construire le DataFrame au format attendu (vectorisé)
                df_demo = pd.DataFrame({
                    "date": df_all["created"],
                    "produit": df_all["model"],
                    "quantite": df_all["quantite_y"],
                    "prix": df_all["model"].map(price_map).fillna(0),
                    "unite": df_all["model"].map(unit_map).fillna(""),
                    "modele": "Baseline (moy. hebdo)",
                    "mae": 0,
                    "mape": 0,
                })
                st.success(f"✅ Données réelles chargées — {df_demo['produit'].nunique()} produits")
            except Exception as e:
                st.warning(f"Données réelles indisponibles ({e}), utilisation du jeu de démo.")
                df_demo = _generate_demo_data()
        else:
            df_demo = _generate_demo_data()
            st.info("🔬 Mode démo — données fictives générées pour illustration")

    # --- Préparation des données (modèles + prévisions) ---
    df_forecast, df_summary = prepare_product_data(df_demo)
    products = sorted(df_forecast["produit"].unique())

    # --- Sidebar : filtres ---
    st.sidebar.header("🔍 Filtres")

    # Sélecteur de produit avec recherche
    selected_product = st.sidebar.selectbox(
        "Produit",
        options=products,
        index=0,
        help="Recherchez un produit dans la liste",
    )

    # Slider temporel
    time_options = {
        "Prévisions futures uniquement": "futur",
        "1 an d'historique": "1y",
        "2 ans d'historique": "2y",
        "3 ans d'historique": "3y",
        "Historique complet": "all",
    }
    time_selection = st.sidebar.selectbox(
        "Période d'affichage",
        options=list(time_options.keys()),
        index=2,
    )
    time_mode = time_options[time_selection]

    # Mode quantité / CA
    metric_mode = st.sidebar.radio(
        "Métrique affichée",
        options=["Quantité vendue", "Chiffre d'affaires (€)"],
        index=0,
        horizontal=True,
    )
    metric_key = "quantite" if metric_mode.startswith("Quantité") else "ca"

    # Sélecteur de classement
    ranking_option = st.sidebar.selectbox(
        "Classement des produits",
        options=[
            "Top produits par chiffre d'affaires",
            "Top produits par volume vendu",
            "Top produits par croissance prévue",
            "Top produits par contribution au CA futur",
        ],
        index=0,
    )

    # --- Initialiser le prix simulé dans la session ---
    if "simulated_price" not in st.session_state:
        st.session_state.simulated_price = None

    # --- Filtrage du produit sélectionné ---
    df_product = df_forecast[df_forecast["produit"] == selected_product].copy()

    # Appliquer le filtre temporel
    if time_mode == "futur":
        df_product = df_product[df_product["prediction"].notna()]
    elif time_mode in ("1y", "2y", "3y"):
        max_date = df_product["date"].max()
        years = int(time_mode[0])
        cutoff = max_date - pd.DateOffset(years=years)
        df_product = df_product[df_product["date"] >= cutoff]

    # --- Layout principal (2 colonnes) ---
    col_left, col_right = st.columns([3, 1], gap="medium")

    # === COLONNE GAUCHE : Graphique ===
    with col_left:
        st.markdown("### 📈 Historique & Prévisions")

        fig = plot_history_forecast(
            df_product,
            metric_mode=metric_key,
            simulated_price=st.session_state.simulated_price,
        )
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"previsions_{selected_product}",
            },
        })

        # KPI row
        render_kpi_row(df_product, st.session_state.simulated_price)

        # --- Panneau Analyse Portefeuille ---
        st.markdown("---")
        st.markdown("### 📊 Analyse portefeuille produits")

        # Mapping option de classement -> colonne
        ranking_map = {
            "Top produits par chiffre d'affaires": ("ca_historique", "Chiffre d'affaires historique (€)"),
            "Top produits par volume vendu": ("volume_total_kg", "Volume total vendu (kg)"),
            "Top produits par croissance prévue": ("croissance_pct", "Croissance prévue (%)"),
            "Top produits par contribution au CA futur": ("ca_previsionnel", "CA prévisionnel (€)"),
        }
        rank_col, rank_title = ranking_map[ranking_option]

        portfolio_col1, portfolio_col2 = st.columns(2)
        with portfolio_col1:
            fig_top = plot_top10_bar(df_summary, metric=rank_col, title=rank_title)
            st.plotly_chart(fig_top, use_container_width=True)
        with portfolio_col2:
            # Part de chaque produit dans le CA total prévu
            total_ca = df_summary["ca_previsionnel"].sum()
            df_parts = df_summary.copy()
            df_parts["part_ca_pct"] = df_parts["ca_previsionnel"] / total_ca * 100
            df_parts = df_parts.nlargest(10, "part_ca_pct").sort_values("part_ca_pct", ascending=True)

            fig_parts = go.Figure()
            fig_parts.add_trace(go.Bar(
                y=df_parts["produit"],
                x=df_parts["part_ca_pct"],
                orientation="h",
                marker=dict(color="#9467bd", opacity=0.85),
                text=df_parts["part_ca_pct"].apply(lambda v: f"{v:.1f} %"),
                textposition="outside",
            ))
            fig_parts.update_layout(
                title="Part du CA prévisionnel total",
                template="plotly_white",
                margin=dict(l=20, r=60, t=40, b=20),
                height=350,
                xaxis_title="% du CA total",
                yaxis_title=None,
                showlegend=False,
            )
            st.plotly_chart(fig_parts, use_container_width=True)

    # === COLONNE DROITE : Fiche produit + Simulateur ===
    with col_right:
        render_product_card(df_product)

        simulated = render_price_simulator(df_product, st.session_state.simulated_price)
        if simulated is not None:
            st.session_state.simulated_price = simulated

        # --- Infos complémentaires ---
        st.markdown("---")
        st.markdown("### 📋 Résumé 52 semaines")
        fut_data = df_product[df_product["prediction"].notna()]
        if len(fut_data) > 0:
            price = (
                st.session_state.simulated_price
                if st.session_state.simulated_price is not None
                else df_product["prix"].iloc[0]
            )
            st.markdown(f"- **Qté totale prévue :** {fut_data['prediction'].sum():,.0f}")
            st.markdown(f"- **Moyenne / semaine :** {fut_data['prediction'].mean():,.1f}")
            st.markdown(f"- **Pic prévu :** {fut_data['prediction'].max():,.0f} "
                        f"(sem. {fut_data.loc[fut_data['prediction'].idxmax(), 'date'].strftime('%d/%m/%Y')})")
            st.markdown(f"- **CA prévisionnel :** {fut_data['prediction'].sum() * price:,.0f} €")
            st.markdown(f"- **Nb sem. sans vente :** {(fut_data['prediction'] < 0.5).sum()}/52")


if __name__ == "__main__":
    main()
