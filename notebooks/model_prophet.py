# ============================================================
# PROPHET — fonction universelle train/predict
# ============================================================
import numpy as np
import pandas as pd
import utils_series
from prophet import Prophet

# ── Hyperparamètres fixes (validés par expérimentation) ──────────────────────
_PROPHET_FIXED: dict = {
    "growth"                  : "flat",
    "seasonality_mode"        : "additive",
    "weekly_seasonality"      : True,
    "daily_seasonality"       : False,
    "yearly_seasonality"      : True,
    "interval_width"          : 0.95,
    "changepoint_prior_scale" : 0.08,
    "holidays_prior_scale"    : 1,
}

# ── Helpers (privés) ──────────────────────────────────────────────────────────

def _extract_series(series: pd.DataFrame | pd.Series) -> pd.Series:
    """Accepte une Series ou un DataFrame avec colonne 'quantite_y'."""
    if isinstance(series, pd.Series):
        return series
    return series["quantite_y"]


def _to_prophet_df(series: pd.Series) -> pd.DataFrame:
    """Convertit une Series indexée datetime → DataFrame Prophet (ds, y)."""
    return (
        series
        .reset_index()
        .rename(columns={series.index.name or "index": "ds", series.name or 0: "y"})
        [["ds", "y"]]
    )


# ── Fonction principale ───────────────────────────────────────────────────────

def run_prophet(
    seasonality_prior_scale : float,
    data_train              : pd.DataFrame | pd.Series,
    data_test               : pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame | dict:
    """
    Deux modes selon les arguments fournis :

    MODE PRÉDICTION (data_test=None)
    ─────────────────────────────────
    - Entraîne sur l'ensemble de data_train
    - Retourne un DataFrame de prédictions sur 52 semaines
      Colonnes : ['yhat', 'yhat_lower', 'yhat_upper'] + index datetime

    MODE ÉVALUATION (data_test fourni)
    ────────────────────────────────────
    - Entraîne sur data_train, prédit sur data_test
    - Retourne les métriques seasonal_metrics (dict)

    Paramètres
    ----------
    seasonality_prior_scale : seul hyperparamètre tunable (validé : [3, 10])
    data_train              : Series ou DataFrame (col 'quantite_y' + index datetime)
    data_test               : idem, optionnel
    """

    y_train  = _extract_series(data_train)
    df_train = _to_prophet_df(y_train)

    # ── Instanciation du modèle ───────────────────────────────────────────────
    m = Prophet(
        growth                 = _PROPHET_FIXED["growth"],
        seasonality_mode       = _PROPHET_FIXED["seasonality_mode"],
        weekly_seasonality     = _PROPHET_FIXED["weekly_seasonality"],
        daily_seasonality      = _PROPHET_FIXED["daily_seasonality"],
        yearly_seasonality     = _PROPHET_FIXED["yearly_seasonality"],
        interval_width         = _PROPHET_FIXED["interval_width"],
        changepoint_prior_scale= _PROPHET_FIXED["changepoint_prior_scale"],
        seasonality_prior_scale= seasonality_prior_scale,
        holidays_prior_scale   = _PROPHET_FIXED["holidays_prior_scale"],
    )
    m.fit(df_train)

    # ── MODE ÉVALUATION ──────────────────────────────────────────────────────
    if data_test is not None:
        y_test   = _extract_series(data_test)
        future   = pd.DataFrame({"ds": y_test.index})
        forecast = m.predict(future)
        y_pred   = np.clip(forecast["yhat"].values, 0, None)
        metrics  = utils_series.seasonal_metrics(y_test, y_pred)
        return metrics

    # ── MODE PRÉDICTION ──────────────────────────────────────────────────────
    future   = m.make_future_dataframe(periods=52, freq="W", include_history=False)
    forecast = m.predict(future)

    df_pred = (
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .assign(
            yhat       = lambda d: np.clip(d["yhat"],       0, None),
            yhat_lower = lambda d: np.clip(d["yhat_lower"], 0, None),
            yhat_upper = lambda d: np.clip(d["yhat_upper"], 0, None),
        )
        .set_index("ds")
    )
    return df_pred