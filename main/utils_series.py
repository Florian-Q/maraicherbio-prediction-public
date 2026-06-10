"""
Fonctions utilitaires pour les Time Series
"""

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error
from datetime import date, timedelta


# Variable globale optionnelle : si définie, TOUS les appels à
# split_adaptive_seasonal l'utilisent comme date de fin de test.
GLOBAL_TEST_END_DATE = None

def dimanche_precedent(d: date) -> date:
    jours_depuis_dimanche = (d.weekday() + 1) % 7
    return d - timedelta(days=jours_depuis_dimanche)

def complete_weekly_dataframe(
    df: pd.DataFrame,
    datetime_col: str = 'datetime',
    cible_col: str = 'quantite_y',
) -> pd.DataFrame:
    """
    Complète un DataFrame avec toutes les semaines manquantes.

    Paramètres:
    -----------
    df : pd.DataFrame
        DataFrame source avec les données
    datetime_col : str
        Nom de la colonne datetime (défaut: 'datetime')
    cible_col : str
        Nom de la colonne cible à prédire (défaut: 'quantite_y')

    Retour:
    -------
    df_complet : pd.DataFrame
        DataFrame avec un DatetimeIndex de Dimanches,
        chaque valeur = somme de la semaine (Lun→Dim)
    """
    time_serie_df = pd.DataFrame({
        datetime_col: df[datetime_col],
        cible_col: df[cible_col]
    })

    # Associer chaque ligne au Dimanche de sa semaine (période W-SUN)
    time_serie_df['semaine'] = (
        time_serie_df[datetime_col]
        .dt.to_period('W-SUN')
        .dt.start_time          # → Lundi de la période…
        + pd.offsets.Week(weekday=6)  # … + 6 jours = Dimanche
    )

    # Regrouper par Dimanche : somme de toute la semaine
    df_grouped = (
        time_serie_df
        .groupby('semaine', as_index=False)[[cible_col]]
        .sum()
    )

    # Créer la plage complète de Dimanches entre min et max
    min_sunday = time_serie_df['semaine'].min()
    max_sunday = time_serie_df['semaine'].max()

    # Si GLOBAL_TEST_END_DATE est définie (déjà un dimanche), prolonger/tronquer jusqu'à cette date
    if GLOBAL_TEST_END_DATE is not None:
        end_sunday = pd.Timestamp(GLOBAL_TEST_END_DATE)
        if end_sunday > max_sunday:
            # Prolonger si la date de fin est après les données existantes
            max_sunday = end_sunday
        elif end_sunday < max_sunday:
            # Tronquer si la date de fin est avant les données existantes
            df_grouped = df_grouped[df_grouped['semaine'] <= end_sunday]
            max_sunday = end_sunday
        else:
            # La date de fin est exactement celle des données
            max_sunday = end_sunday

    tous_les_dimanches = pd.date_range(start=min_sunday, end=max_sunday, freq='W-SUN')

    # DataFrame avec tous les Dimanches
    df_complet = pd.DataFrame({'semaine': tous_les_dimanches})

    # Fusionner pour conserver les données existantes
    df_complet = df_complet.merge(df_grouped, on='semaine', how='left')

    # Mettre 'semaine' en DatetimeIndex
    df_complet = df_complet.set_index('semaine')

    # Remplacer les NaN (semaines sans données) par 0
    df_complet = df_complet.fillna(0)

    return df_complet


def split_train_test_timeseries(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Découpe un DataFrame de time series en jeux train / test,
    en réservant la dernière année pour le test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec un DatetimeIndex trié croissant.

    Returns
    -------
    df_train, df_test : tuple[pd.DataFrame, pd.DataFrame]
    ex:
    df_train, df_test = split_train_test_timeseries(time_serie)

    Raises
    ------
    TypeError
        Si l'index n'est pas un DatetimeIndex.
    ValueError
        Si le découpage est impossible (pas assez de données).

    Examples
    --------
    >>> df_train, df_test = split_train_test_timeseries(df)
    """
    TEST_YEARS = 1  # toujours 1 an de test

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"L'index doit être un DatetimeIndex, reçu : {type(df.index).__name__}"
        )

    dates = df.index
    cutoff = dates.max() - relativedelta(years=TEST_YEARS)

    train_mask = dates <= cutoff
    test_mask = dates > cutoff

    if train_mask.sum() == 0:
        raise ValueError("Pas assez de données : aucun point dans le train.")
    if test_mask.sum() == 0:
        raise ValueError("Pas assez de données : aucun point dans le test.")

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    print(f"Train : {df_train.index[0].strftime('%Y-%m-%d')}  →  "
          f"{df_train.index[-1].strftime('%Y-%m-%d')}  ({len(df_train)} semaines)")
    print(f"Test  : {df_test.index[0].strftime('%Y-%m-%d')}   →  "
          f"{df_test.index[-1].strftime('%Y-%m-%d')}   ({len(df_test)} semaines)")

    return df_train, df_test


def split_adaptive_seasonal(
    df: pd.DataFrame,
    test_pct: float = 0.20,
    min_train_years: int = 2,
    min_test_years: int = 1,
    max_test_years: int = 3,
    test_end_date=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split adaptatif par produit : le pourcentage de test est arrondi à l'année
    la plus proche (1, 2 ou 3 ans).  Le test se termine **toujours** à la
    dernière date disponible, ce qui rend la fonction évolutive quand la base
    est mise à jour.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec DatetimeIndex trié (sortie de complete_weekly_dataframe).
    test_pct : float
        Pourcentage cible pour le test (ex: 0.20 = 20 %).
    min_train_years : int
        Années minimum à conserver dans le train (2 par défaut).
    min_test_years : int
        Années minimum exigées pour le test (1 par défaut).
    max_test_years : int
        Années maximum allouées au test (3 par défaut).
    test_end_date : datetime-like, optional
        Date de fin commune pour tous les tests (ex: date max globale).
        Si fournie, toutes les DataFrames sont prolongées avec des zéros
        jusqu'à cette date.

    Returns
    -------
    df_train, df_test : tuple[pd.DataFrame, pd.DataFrame]

    Raises
    ------
    TypeError
        Si l'index n'est pas un DatetimeIndex.
    ValueError
        Si les données sont insuffisantes pour le split demandé.

    Examples
    --------
    >>> df_train, df_test = split_adaptive_seasonal(df)
    >>> df_train, df_test = split_adaptive_seasonal(df, test_pct=0.25)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"L'index doit être un DatetimeIndex, reçu : {type(df.index).__name__}"
        )

    first_date = df.index.min()
    last_date = df.index.max()

    # Si une date de fin commune est fournie (paramètre ou variable globale)
    end_date = test_end_date if test_end_date is not None else GLOBAL_TEST_END_DATE
    if end_date is not None:
        end_date_ts = pd.Timestamp(end_date)
        if end_date_ts > last_date:
            # Prolonger si la date de fin est après les données existantes
            extra = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1),
                end=end_date_ts,
                freq='W-SUN',
            )
            df = df.reindex(df.index.union(extra), fill_value=0)
            last_date = end_date_ts
        elif end_date_ts < last_date:
            # Tronquer si la date de fin est avant les données existantes
            df = df[df.index <= end_date_ts]
            last_date = end_date_ts
        else:
            # La date de fin est exactement celle des données
            last_date = end_date_ts

    total_years = (last_date - first_date).days / 365.25

    # Années de test cibles (arrondies à l'entier le plus proche)
    target_test_years = round(total_years * test_pct)

    # Appliquer les contraintes min / max
    max_from_data = int(total_years) - min_train_years
    test_years = max(min_test_years, min(target_test_years, max_test_years, max_from_data))

    if test_years < min_test_years:
        raise ValueError(
            f"Pas assez de données pour le split : {len(df)} semaines "
            f"({total_years:.1f} ans), il faut au moins "
            f"{min_train_years + min_test_years} ans"
        )

    # Cutoff = dernière date − N années (même jour, N ans avant)
    cutoff = last_date - relativedelta(years=test_years)

    train_mask = df.index <= cutoff
    test_mask = df.index > cutoff

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Pas assez de données pour le split : {len(df)} semaines "
            f"({total_years:.1f} ans). Le cutoff ({cutoff.date()}) tombe "
            f"en dehors des dates disponibles "
            f"({df.index.min().date()} → {df.index.max().date()})."
        )

    # Vérifier que le train a bien au moins min_train_years années
    train_years = (cutoff - first_date).days / 365.25
    if train_years < min_train_years:
        raise ValueError(
            f"Pas assez d'années de train : {train_years:.1f} ans "
            f"(minimum requis : {min_train_years} ans). "
            f"Il faut au moins {min_train_years + test_years} ans de données."
        )

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    pct_real = len(df_test) / len(df) * 100
    print(f"Train : {df_train.index[0].strftime('%Y-%m-%d')}  →  "
          f"{df_train.index[-1].strftime('%Y-%m-%d')}  ({len(df_train)} semaines)")
    print(f"Test  : {df_test.index[0].strftime('%Y-%m-%d')}   →  "
          f"{df_test.index[-1].strftime('%Y-%m-%d')}   ({len(df_test)} semaines)")
    print(f"  → test={test_years} an(s) sur {total_years:.1f} ans ({pct_real:.0f}%)")

    return df_train, df_test


def seasonal_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    """
    Métriques désagrégées par saisonnalité :
    - En saison (y_true > 0) : MAE_in, MAPE
    - Hors saison (y_true == 0) : MAE_out
    - Global : MAE_all, sMAPE_all

    Parameters
    ----------
    y_true : pd.Series
        Valeurs réelles.
    y_pred : np.ndarray
        Valeurs prédites.

    Returns
    -------
    dict avec les clés :
        MAE_in, MAE_out, MAE_all, MAPE, sMAPE_all, pct_zeros
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask_in = y_true > 0
    mask_out = y_true == 0

    # --- MAE ---
    mae_in = (
        mean_absolute_error(y_true[mask_in], y_pred[mask_in])
        if mask_in.sum() > 0
        else 0.0
    )
    mae_out = (
        mean_absolute_error(y_true[mask_out], y_pred[mask_out])
        if mask_out.sum() > 0
        else 0.0
    )
    mae_all = mean_absolute_error(y_true, y_pred)

    # --- MAPE (standard, uniquement sur y_true > 0) ---
    mape = (
        (np.abs((y_true[mask_in] - y_pred[mask_in]) / y_true[mask_in]).mean()) * 100
        if mask_in.sum() > 0
        else np.nan
    )

    # --- sMAPE (symétrique, gère les zeros, sur toutes les données) ---
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Éviter division par zéro (arrive si y_true=0 et y_pred=0)
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape_all = (2.0 * np.abs(y_true - y_pred) / denominator).mean() * 100

    return {
        'MAE_in': mae_in,
        'MAE_out': mae_out,
        'MAE_all': mae_all,
        'MAPE': mape,
        'sMAPE_all': smape_all,
        'pct_zeros': (y_true == 0).mean() * 100,
    }
