"""
data_prep.py
============
Build the NBA spread-error analysis dataframe from the raw odds SQLite database.

Source: kyleskom/NBA-Machine-Learning-Sports-Betting OddsData.sqlite, seasons 2014-15
through 2023-24, regular season only.

Convention discovered empirically (see writeup §Data):
    |Spread|  = magnitude of the favorite's spread (always positive)
    favorite  = home if ML_Home < ML_Away else away
    error     = fav_actual_margin - |Spread|
              = (favorite covered by more than expected when positive)

Equivalent to the model's notation: spread_fav = -|Spread|, and
    X = actual_margin + spread_fav = fav_margin - |Spread|.
"""

from __future__ import annotations
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
]

# Approximate regular-season end dates. 2019-20 was paused by COVID and
# resumed in the Orlando bubble; 2020-21 had a delayed start due to COVID.
REG_SEASON_END = {
    "2014-15": "2015-04-15",
    "2015-16": "2016-04-13",
    "2016-17": "2017-04-12",
    "2017-18": "2018-04-11",
    "2018-19": "2019-04-10",
    "2019-20": "2020-08-14",  # bubble end
    "2020-21": "2021-05-16",
    "2021-22": "2022-04-10",
    "2022-23": "2023-04-09",
    "2023-24": "2024-04-14",
}

# Seasons with anomalous conditions that we'll flag for a robustness check
ANOMALOUS_SEASONS = {"2019-20", "2020-21"}

REST_CAP = 5


def load_season(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    """Load one season from the `_new` view of the odds database."""
    df = pd.read_sql_query(f'SELECT * FROM "odds_{season}_new"', conn)
    df["season"] = season
    return df


def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_dataframe(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    frames = [load_season(conn, s) for s in SEASONS]
    conn.close()

    df = pd.concat(frames, ignore_index=True)

    # Coerce numeric columns (Spread is sometimes text in older rows)
    for col in ["Spread", "ML_Home", "ML_Away", "Win_Margin",
                "Days_Rest_Home", "Days_Rest_Away", "Points"]:
        df[col] = to_numeric(df[col])

    df["date"] = pd.to_datetime(df["Date"])

    # ----- Normalize team names (franchise relabels in source data) ----------
    # The Charlotte franchise was officially renamed Bobcats -> Hornets in
    # 2014-15. Some rows in this source DB still carry the old "Bobcats"
    # label through 2021-22; collapse them into the modern franchise name.
    name_fix = {"Charlotte Bobcats": "Charlotte Hornets"}
    df["Home"] = df["Home"].replace(name_fix)
    df["Away"] = df["Away"].replace(name_fix)

    # ----- Regular-season filter ---------------------------------------------
    keep = []
    for season, end in REG_SEASON_END.items():
        m = (df["season"] == season) & (df["date"] <= pd.Timestamp(end))
        keep.append(m)
    reg_mask = np.logical_or.reduce(keep)
    df = df.loc[reg_mask].copy()

    # ----- Drop missing / invalid -------------------------------------------
    n0 = len(df)
    df = df.dropna(subset=["Spread", "ML_Home", "ML_Away", "Win_Margin",
                           "Days_Rest_Home", "Days_Rest_Away"])

    # ----- Drop pick'ems, ties on ML, and implausible spreads ----------------
    df = df[df["Spread"].abs() > 0]            # drop pick'ems / missing-as-zero
    df = df[df["ML_Home"] != df["ML_Away"]]    # need a clear favorite
    # Three rows in the source DB have Spread set to the over/under or to a
    # nonsense value (>25 points, larger than any historical NBA spread).
    df = df[df["Spread"].abs() <= 25]

    # ----- Identify favorite -------------------------------------------------
    home_fav = df["ML_Home"] < df["ML_Away"]   # more negative ML = favorite
    df["fav_team"] = np.where(home_fav, df["Home"], df["Away"])
    df["dog_team"] = np.where(home_fav, df["Away"], df["Home"])
    df["home"] = home_fav.astype(int)

    df["spread_mag"] = df["Spread"].abs()
    fav_margin = np.where(home_fav, df["Win_Margin"], -df["Win_Margin"])
    df["actual_margin"] = fav_margin
    df["error"] = fav_margin - df["spread_mag"]   # X = actual + spread (spread<0)

    # ----- Rest covariates ---------------------------------------------------
    df["fav_rest"] = np.where(home_fav, df["Days_Rest_Home"], df["Days_Rest_Away"])
    df["dog_rest"] = np.where(home_fav, df["Days_Rest_Away"], df["Days_Rest_Home"])
    df["fav_rest"] = df["fav_rest"].clip(upper=REST_CAP)
    df["dog_rest"] = df["dog_rest"].clip(upper=REST_CAP)
    df["rest_diff_raw"] = df["fav_rest"] - df["dog_rest"]
    df["rest_diff"] = df["rest_diff_raw"] - df["rest_diff_raw"].mean()

    df["anomalous"] = df["season"].isin(ANOMALOUS_SEASONS).astype(int)

    # Final canonical column order
    out = df[[
        "date", "season", "fav_team", "dog_team", "spread_mag",
        "actual_margin", "error", "home",
        "fav_rest", "dog_rest", "rest_diff_raw", "rest_diff",
        "anomalous",
    ]].rename(columns={"spread_mag": "spread"}).reset_index(drop=True)

    out["date"] = out["date"].dt.date
    return out


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    raw_db = here / "data" / "raw" / "OddsData.sqlite"
    out_csv = here / "data" / "nba_spreads.csv"

    df = build_dataframe(raw_db)
    df.to_csv(out_csv, index=False)

    # Console summary
    print(f"Saved {out_csv} ({len(df):,} games)")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Teams (favorite role): {df['fav_team'].nunique()}")
    print(f"Mean error: {df['error'].mean():+.3f}  SD: {df['error'].std():.3f}")
    print(f"Mean spread: {df['spread'].mean():+.3f}  SD: {df['spread'].std():.3f}")
    print(f"Home-favorite share: {df['home'].mean():.3f}")
    print(f"Anomalous games (bubble + fan-less): {df['anomalous'].sum()}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
