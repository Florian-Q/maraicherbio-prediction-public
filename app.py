"""
Dashboard MaraîcherBio — Application Streamlit v2
Tableau interactif + graphique agrégé des produits maraîchers.

Usage :
    streamlit run app.py
"""

import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="Dashboard MaraîcherBio",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

_DATA_REAL_AVAILABLE = False
try:
    import utils
    import utils_series
    if (NOTEBOOKS_DIR.parent / "data" / "uc_order_products.csv").exists():
        _DATA_REAL_AVAILABLE = True
except Exception:
    pass


def _load_real_prices() -> dict:
    """Charge les sell_price depuis uc_products.csv → mapping model -> prix."""
    try:
        data_path = Path(__file__).parent / "data"
        products = pd.read_csv(data_path / "uc_products.csv", decimal=",")
        products["sell_price"] = pd.to_numeric(products["sell_price"], errors="coerce")
        return products.dropna(subset=["sell_price"]).groupby("model")["sell_price"].last().to_dict()
    except Exception:
        return {}


def _load_model_metrics() -> pd.DataFrame:
    """Charge le CSV model_win_metric.csv (meilleur modèle + métriques par produit)."""
    path = Path(__file__).parent / "data" / "model_win_metric.csv"
    if path.exists():
        df = pd.read_csv(path)
        # Renommer pour compatibilité
        if "model_win" in df.columns:
            df["best_model"] = df["model_win"]
        return df
    # Fallback sur l'ancien CSV
    old = Path(__file__).parent / "data" / "Baseline_metrics.csv"
    if old.exists():
        df = pd.read_csv(old)
        df["best_model"] = "Baseline"
        return df
    return pd.DataFrame()


def _load_predictions() -> Optional[pd.DataFrame]:
    """
    Charge le CSV de prédictions.
    Format attendu : colonne 'ds' (dates) + 1 colonne par produit (valeurs = prédictions).
    Retourne un DataFrame long : produit, date, prediction.
    """
    path = Path(__file__).parent / "data" / "Predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    # Wide → long
    df_long = df.melt(id_vars=[date_col], var_name="produit", value_name="prediction")
    df_long = df_long.rename(columns={date_col: "date"})
    df_long["borne_inf"] = np.clip(df_long["prediction"] * 0.7, 0, None)
    df_long["borne_sup"] = df_long["prediction"] * 1.3
    return df_long


# ============================================================
# 3. MOTEUR DE PRÉDICTION (fallback baseline)
# ============================================================

def train_baseline_model(ts: pd.Series) -> dict:
    """Moyenne et écart-type par semaine ISO."""
    weekly_mean = ts.groupby(ts.index.isocalendar().week).mean()
    weekly_std = ts.groupby(ts.index.isocalendar().week).std().fillna(0)
    return {"weekly_mean": weekly_mean, "weekly_std": weekly_std}


def generate_forecast(model: dict, last_date: pd.Timestamp, n_weeks: int = 52) -> pd.DataFrame:
    """Génère les prévisions hebdomadaires pour n_weeks."""
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
# 4. PRÉPARATION DES DONNÉES
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_all_data() -> dict:
    """
    Prépare l'historique (via complete_weekly_dataframe) + prévisions.
    """
    df_metrics = _load_model_metrics()
    price_map = _load_real_prices()
    df_forecast = _load_predictions()

    df_hist = pd.DataFrame(columns=["date", "produit", "quantite"])

    weight_units_map = {}
    if _DATA_REAL_AVAILABLE:
        df_all = utils.charger_dataframe()
        # Aligner tous les historiques sur la même date de fin
        utils_series.GLOBAL_TEST_END_DATE = df_all["created"].max()
        produits_all = sorted(df_all["model"].unique())
        # Mapping weight_units depuis charger_dataframe
        weight_units_map = df_all.groupby("model")["weight_units"].first().to_dict()

        hist_records = []
        for produit in produits_all:
            df_p = df_all[df_all["model"] == produit]
            if len(df_p) < 10:
                continue
            try:
                weekly = utils_series.complete_weekly_dataframe(df_p, "created", "quantite_y")
                y = weekly["quantite_y"]
                for d, v in y.items():
                    hist_records.append({"date": d, "produit": produit, "quantite": v})
            except Exception:
                continue

        if hist_records:
            df_hist = pd.DataFrame(hist_records)
            df_hist["date"] = pd.to_datetime(df_hist["date"])

    if df_forecast is None:
        df_forecast = pd.DataFrame()

    return {
        "df_hist": df_hist,
        "df_forecast": df_forecast,
        "df_metrics": df_metrics,
        "price_map": price_map,
        "weight_units_map": weight_units_map,
    }


# ============================================================
# 5. GRAPHIQUE PLOTLY
# ============================================================

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_view(
    df_hist: pd.DataFrame,
    df_forecast: pd.DataFrame,
    checked_products: list,
    prices: dict,
    metric_mode: str,
    title: str = "",
) -> go.Figure:
    """
    - 1 à 4 produits → courbes individuelles de couleurs différentes
    - 5+ produits    → courbe cumulative unique
    """
    fig = go.Figure()
    y_label = "Chiffre d'affaires (€)" if metric_mode == "ca" else "Quantité vendue"

    # --- Mode individuel : 1 à 4 produits ---
    if len(checked_products) <= 4:
        for i, produit in enumerate(checked_products):
            color = COLORS[i % len(COLORS)]
            price = prices.get(produit, 0)

            # Historique
            hp = df_hist[df_hist["produit"] == produit]
            if not hp.empty:
                hp = hp.sort_values("date")
                y_hist = hp["quantite"].values * price if metric_mode == "ca" else hp["quantite"].values
                fig.add_trace(go.Scatter(
                    x=hp["date"], y=y_hist,
                    mode="lines", name=produit,
                    line=dict(color=color, width=2, dash="solid"),
                    legendgroup=produit,
                    hovertemplate=f"<b>{produit}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.1f}}<extra></extra>",
                ))

            # Prévisions
            fp = df_forecast[df_forecast["produit"] == produit]
            if not fp.empty:
                fp = fp.sort_values("date")
                y_fut = fp["prediction"].values * price if metric_mode == "ca" else fp["prediction"].values
                fig.add_trace(go.Scatter(
                    x=fp["date"], y=y_fut,
                    mode="lines", name=f"{produit}",
                    line=dict(color=color, width=2, dash="dot"),
                    legendgroup=produit, showlegend=False,
                    hovertemplate=f"<b>{produit}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.1f}}<extra></extra>",
                ))

    # --- Mode cumulatif : 5+ produits ---
    else:
        hist_all = df_hist[df_hist["produit"].isin(checked_products)].copy()
        if not hist_all.empty:
            if metric_mode == "ca":
                hist_all["val"] = hist_all["quantite"] * hist_all["produit"].map(prices).fillna(0)
            else:
                hist_all["val"] = hist_all["quantite"]
            hist_agg = hist_all.groupby("date")["val"].sum().reset_index().sort_values("date")
            fig.add_trace(go.Scatter(
                x=hist_agg["date"], y=hist_agg["val"],
                mode="lines", name="Historique réel",
                line=dict(color="black", width=2),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>%{y:.1f}<extra></extra>",
            ))

        fut_all = df_forecast[df_forecast["produit"].isin(checked_products)].copy()
        if not fut_all.empty:
            if metric_mode == "ca":
                fut_all["pred"] = fut_all["prediction"] * fut_all["produit"].map(prices).fillna(0)
                fut_all["inf"] = fut_all["borne_inf"] * fut_all["produit"].map(prices).fillna(0)
                fut_all["sup"] = fut_all["borne_sup"] * fut_all["produit"].map(prices).fillna(0)
            else:
                fut_all["pred"] = fut_all["prediction"]
                fut_all["inf"] = fut_all["borne_inf"]
                fut_all["sup"] = fut_all["borne_sup"]
            fut_agg = fut_all.groupby("date")[["pred", "inf", "sup"]].sum().reset_index().sort_values("date")

            fig.add_trace(go.Scatter(
                x=pd.concat([fut_agg["date"], fut_agg["date"][::-1]]),
                y=pd.concat([fut_agg["inf"], fut_agg["sup"][::-1]]),
                fill="toself", fillcolor="rgba(255,0,0,0.12)",
                line=dict(color="rgba(255,0,0,0)"),
                name="IC 95%", hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=fut_agg["date"], y=fut_agg["pred"],
                mode="lines", name="Prévisions",
                line=dict(color="red", width=2),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>%{y:.1f}<extra></extra>",
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        height=500,
    )
    return fig


# ============================================================
# 7. APPLICATION PRINCIPALE
# ============================================================

def main():
    st.title("🥦 Dashboard MaraîcherBio")
    st.caption(
        "Cochez 1 à 4 produits pour des courbes individuelles, "
        "5+ pour une courbe cumulative. Modifiez les prix dans le tableau."
    )

    # --- Chargement ---
    with st.spinner("Chargement des données..."):
        data = prepare_all_data()
        df_hist = data["df_hist"]
        df_forecast = data["df_forecast"]
        df_metrics = data["df_metrics"]
        price_map = data["price_map"]
        weight_units_map = data["weight_units_map"]

    if df_hist.empty and df_metrics.empty:
        st.error("Aucune donnée disponible.")
        return

    products_list = sorted(df_hist["produit"].unique()) if not df_hist.empty else sorted(df_metrics["produit"].unique())

    # --- Tableau fusionné ---
    table_data = pd.DataFrame({"produit": products_list})
    if not df_metrics.empty:
        metric_cols = ["produit", "MAE_in", "MAPE"]
        if "best_model" in df_metrics.columns:
            metric_cols.append("best_model")
        if "sMAPE_all" in df_metrics.columns:
            metric_cols.append("sMAPE_all")
        table_data = table_data.merge(df_metrics[metric_cols], on="produit", how="left")
    table_data["sell_price"] = table_data["produit"].map(price_map).fillna(0)
    table_data["weight_units"] = table_data["produit"].map(weight_units_map).fillna("?")
    table_data["MAE_in"] = table_data["MAE_in"].fillna(0).round(2)
    table_data["MAPE"] = table_data["MAPE"].fillna(0).round(1)
    if "best_model" not in table_data.columns:
        table_data["best_model"] = "—"

    # --- Session state ---
    if "edited_prices" not in st.session_state:
        st.session_state.edited_prices = {}
    if "checked_products" not in st.session_state:
        st.session_state.checked_products = []

    # --- Layout 50/50 ---
    col_graph, col_table = st.columns([1, 1], gap="medium")

    # ===================== COLONNE TABLEAU =====================
    with col_table:
        st.markdown("### 📋 Produits")

        # Filtres
        time_options = {
            "Futur uniquement": "futur",
            "1 an": "1y",
            "2 ans": "2y",
            "3 ans": "3y",
            "Tout l'historique": "all",
        }
        f1, f2, f3 = st.columns([2, 2, 1])
        with f1:
            time_selection = st.selectbox("Période", list(time_options.keys()), index=2, key="time_select")
        with f2:
            metric_mode = st.radio("Métrique", ["Quantité", "Chiffre d'affaires (€)"],
                                   index=0, horizontal=True, key="metric_radio")
        with f3:
            st.caption("")  # spacer
        time_mode = time_options[time_selection]
        metric_key = "quantite" if metric_mode == "Quantité" else "ca"

        # Recherche + Boutons Tous/Aucun
        sr1, sr2, sr3 = st.columns([3, 1, 1])
        with sr1:
            search = st.text_input("🔍 Rechercher un produit", placeholder="ex: Courgette, Tomate...")
        with sr2:
            if st.button("✅ Tous", use_container_width=True):
                st.session_state.checked_products = list(products_list)
                st.session_state.pop("product_editor", None)  # reset editor state
                st.rerun()
        with sr3:
            if st.button("🔄 Aucun", use_container_width=True):
                st.session_state.checked_products = []
                st.session_state.pop("product_editor", None)
                st.rerun()

        # Filtrer par recherche (clear editor si le filtre change)
        if "last_search" not in st.session_state:
            st.session_state.last_search = ""
        if search.strip() != st.session_state.last_search:
            st.session_state.last_search = search.strip()
            st.session_state.pop("product_editor", None)

        df_table = table_data.copy()
        if search.strip():
            df_table = df_table[df_table["produit"].str.contains(search.strip(), case=False)]

        # Construire le DataFrame éditable
        df_edit = df_table[["produit", "weight_units", "sell_price", "best_model", "MAE_in", "MAPE"]].copy()
        df_edit = df_edit.rename(columns={"sell_price": "Prix (€)"})
        df_edit["✅"] = df_table["produit"].isin(st.session_state.checked_products)

        # Appliquer les prix modifiés (depuis session_state)
        for prod, p in st.session_state.edited_prices.items():
            mask = df_edit["produit"] == prod
            df_edit.loc[mask, "Prix (€)"] = p

        # Appliquer les checkboxes (depuis session_state)
        for prod in st.session_state.checked_products:
            mask = df_edit["produit"] == prod
            df_edit.loc[mask, "✅"] = True

        column_config = {
            "produit": st.column_config.TextColumn("Produit", disabled=True),
            "weight_units": st.column_config.TextColumn("Unité", disabled=True, width="small"),
            "Prix (€)": st.column_config.NumberColumn(
                "Prix (€)", min_value=0.01, step=0.05, format="%.2f",
            ),
            "best_model": st.column_config.TextColumn("Modèle", disabled=True, width="small"),
            "MAE_in": st.column_config.NumberColumn("MAE in", disabled=True, format="%.2f", width="small"),
            "MAPE": st.column_config.NumberColumn("MAPE", disabled=True, format="%.1f", width="small"),
            "✅": st.column_config.CheckboxColumn("Sél.", width="small"),
        }

        edited = st.data_editor(
            df_edit,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            height=500,
            key="product_editor",
        )

        # Extraire prix modifiés et checkboxes depuis l'état du widget
        if edited is not None:
            for _, row in edited.iterrows():
                produit = row["produit"]
                new_price = row["Prix (€)"]
                original = df_table[df_table["produit"] == produit]["sell_price"].values
                if len(original) > 0:
                    orig_price = float(original[0])
                    if abs(new_price - orig_price) > 0.001:
                        st.session_state.edited_prices[produit] = new_price
                    elif produit in st.session_state.edited_prices:
                        del st.session_state.edited_prices[produit]
            st.session_state.checked_products = edited[edited["✅"]]["produit"].tolist()

        # --- Boutons Reset ---
        modified = dict(sorted(st.session_state.edited_prices.items()))
        if modified:
            st.markdown("---")
            st.caption("Prix modifiés — cliquer pour réinitialiser au prix réel :")
            cols = st.columns(min(len(modified), 4))
            for i, (prod, p) in enumerate(modified.items()):
                orig = price_map.get(prod, 0)
                with cols[i % 4]:
                    if st.button(f"↩ {prod} ({p:.2f}→{orig:.2f})", key=f"rst_{prod}"):
                        del st.session_state.edited_prices[prod]
                        st.rerun()

    # ===================== COLONNE GAUCHE : Graphique =====================
    with col_graph:
        checked = st.session_state.checked_products
        if not checked:
            st.info("👉 Cochez des produits dans le tableau de droite pour afficher le graphique.")
        else:
            effective_prices = {
                p: st.session_state.edited_prices.get(p, price_map.get(p, 0))
                for p in checked
            }

            # Filtre temporel
            if time_mode != "all" and not df_hist.empty:
                max_date_hist = df_hist["date"].max()
                if time_mode != "futur":
                    years = int(time_mode[0])
                    cutoff = max_date_hist - pd.DateOffset(years=years)
                    df_hist_view = df_hist[df_hist["date"] >= cutoff]
                else:
                    df_hist_view = df_hist[df_hist["date"] > max_date_hist]
            else:
                df_hist_view = df_hist.copy()

            # Titre + CA prévu
            n = len(checked)
            total_ca = 0
            if metric_key == "ca" and not df_forecast.empty:
                fut_all = df_forecast[df_forecast["produit"].isin(checked)]
                if not fut_all.empty:
                    total_ca = sum(
                        fut_all[fut_all["produit"] == p]["prediction"].sum() * effective_prices.get(p, 0)
                        for p in checked
                    )

            if n <= 4:
                title = f"{n} produit(s) — Courbes individuelles"
            else:
                title = f"{n} produit(s) — Courbe cumulative"
            if total_ca > 0:
                title += f" | CA prévu : {total_ca:,.0f} €"

            fig = plot_view(
                df_hist_view, df_forecast, checked, effective_prices, metric_key, title
            )
            st.plotly_chart(fig, use_container_width=True, config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "displaylogo": False,
            })

            # --- KPI row ---
            if not df_forecast.empty:
                fut_checked = df_forecast[df_forecast["produit"].isin(checked)]
                qty_total = fut_checked["prediction"].sum() if not fut_checked.empty else 0

                # CA avec prix par défaut (sell_price)
                ca_default = sum(
                    fut_checked[fut_checked["produit"] == p]["prediction"].sum() * price_map.get(p, 0)
                    for p in checked
                )
                # CA avec prix modifiés
                ca_modified = sum(
                    fut_checked[fut_checked["produit"] == p]["prediction"].sum() * effective_prices.get(p, 0)
                    for p in checked
                )
                has_modifs = any(p in st.session_state.edited_prices for p in checked)

                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("📦 Qté prévue (52 sem.)", f"{qty_total:,.0f}")
                with k2:
                    if has_modifs:
                        delta_ca = ca_modified - ca_default
                        st.metric(
                            "💰 CA prévisionnel",
                            f"{ca_modified:,.0f} €",
                            delta=f"{delta_ca:+,.0f} €",
                        )
                    else:
                        st.metric("💰 CA prévisionnel", f"{ca_default:,.0f} €")
                with k3:
                    st.metric("🧺 Produits", f"{n}")


if __name__ == "__main__":
    main()
