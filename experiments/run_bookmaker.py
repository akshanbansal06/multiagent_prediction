import numpy as np
from src.data.loader import load_matches
from src.evaluation.benchmarks import odds_to_probs
from src.evaluation.metrics import rps_batch

m = load_matches()
cols = ["B365H", "B365D", "B365A"]
print("rows missing odds:", int(m[cols].isna().any(axis=1).sum()))
m = m.dropna(subset=cols)

odds = m[cols].to_numpy()
print(f"mean overround: {(1.0 / odds).sum(axis=1).mean():.3f}")

probs = odds_to_probs(odds)
scores = rps_batch(probs, m["outcome"].to_numpy())
print(f"matches: {len(scores)}  mean RPS: {scores.mean():.4f}")
print(m.assign(rps=scores).groupby("season")["rps"].mean().round(4))
