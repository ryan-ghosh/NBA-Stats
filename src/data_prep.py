from __future__ import annotations
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
]

REG_SEASON_END = {
    "2014-15": "2015-04-15",
    "2015-16": "2016-04-13",
    "2016-17": "2017-04-12",
    "2017-18": "2018-04-11",
    "2018-19": "2019-04-10",
    "2019-20": "2020-08-14",
    "2020-21": "2021-05-16",
    "2021-22": "2022-04-10",
    "2022-23": "2023-04-09",
    "2023-24": "2024-04-14",
}

#covid seasons
ANOMALOUS_SEASONS = {"2019-20", "2020-21"}

REST_CAP = 5


def load_season(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
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

    for col in ["Spread", "ML_Home", "ML_Away", "Win_Margin",
                "Days_Rest_Home", "Days_Rest_Away", "Points"]:
        df[col] = to_numeric(df[col])

    df["date"] = pd.to_datetime(df["Date"])

    name_fix = {"Charlotte Bobcats": "Charlotte Hornets"}
    df["Home"] = df["Home"].replace(name_fix)
    df["Away"] = df["Away"].replace(name_fix)


    keep = []
    for season, end in REG_SEASON_END.items():
        m = (df["season"] == season) & (df["date"] <= pd.Timestamp(end))
        keep.append(m)
    reg_mask = np.logical_or.reduce(keep)
    df = df.loc[reg_mask].copy()


    n0 = len(df)
    df = df.dropna(subset=["Spread", "ML_Home", "ML_Away", "Win_Margin",
                           "Days_Rest_Home", "Days_Rest_Away"])

    #get rid of pickems (0 spread) and impossible spreads
    df = df[df["Spread"].abs() > 0]
    df = df[df["ML_Home"] != df["ML_Away"]]
    df = df[df["Spread"].abs() <= 25]


    #get favourites
    home_fav = df["ML_Home"] < df["ML_Away"]
    df["fav_team"] = np.where(home_fav, df["Home"], df["Away"])
    df["dog_team"] = np.where(home_fav, df["Away"], df["Home"])
    df["home"] = home_fav.astype(int)

    df["spread_mag"] = df["Spread"].abs()
    fav_margin = np.where(home_fav, df["Win_Margin"], -df["Win_Margin"])
    df["actual_margin"] = fav_margin
    df["error"] = fav_margin - df["spread_mag"]


    #rest days
    df["fav_rest"] = np.where(home_fav, df["Days_Rest_Home"], df["Days_Rest_Away"])
    df["dog_rest"] = np.where(home_fav, df["Days_Rest_Away"], df["Days_Rest_Home"])
    df["fav_rest"] = df["fav_rest"].clip(upper=REST_CAP)
    df["dog_rest"] = df["dog_rest"].clip(upper=REST_CAP)
    df["rest_diff_raw"] = df["fav_rest"] - df["dog_rest"]
    df["rest_diff"] = df["rest_diff_raw"] - df["rest_diff_raw"].mean()

    df["anomalous"] = df["season"].isin(ANOMALOUS_SEASONS).astype(int)



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
