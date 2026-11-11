from pathlib import Path
import pandas as pd

RENAME = {"HomeTeam": "home", "AwayTeam": "away", "FTHG": "home_goals",
          "FTAG": "away_goals", "Date": "date"}
ODDS = ["B365H", "B365D", "B365A"]   # Bet365 odds, used later as a benchmark

def load_matches(raw_dir="data/raw"):
    frames = []
    for path in sorted(Path(raw_dir).glob("*.csv")):
        df = pd.read_csv(path, encoding="latin-1")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
        keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"] + [c for c in ODDS if c in df.columns]
        df = df[keep].rename(columns=RENAME)
        df["season"] = path.stem.replace("E0_", "")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, format="mixed")
    out["outcome"] = out["FTR"].map({"H": 0, "D": 1, "A": 2})
    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    out["match_id"] = out.index
    return out.drop(columns="FTR")
