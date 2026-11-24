import numpy as np
import pandas as pd
from src.faults.faults import FAULTS
from src.evaluation.metrics import rps_batch

TRAIN = {"1516", "1617", "1718", "1819"}
df = pd.read_csv("data/forecasts.csv", dtype={"season": str})
held = ~df["season"].isin(TRAIN)
y = df["outcome"].to_numpy()
base = df[["elo_H", "elo_D", "elo_A"]].to_numpy()
naive = df[["naive_H", "naive_D", "naive_A"]].to_numpy()
START = int(np.argmax(held.to_numpy()))        # first held-out match: sleeper wakes here


def held_rps(p):
    s = rps_batch(p, y)
    ok = ~np.isnan(s) & held.to_numpy()
    return float(s[ok].mean()), float(np.isnan(s)[held.to_numpy()].mean())


print("Faults applied to the Elo agent. Held-out seasons, lower RPS is better.")
print(f"{'Agent':<22}{'RPS':>8}{'Unusable rows':>16}")
print(f"{'Honest Elo':<22}{held_rps(base)[0]:>8.4f}{0:>16.1%}")
print(f"{'Naive (floor)':<22}{held_rps(naive)[0]:>8.4f}{0:>16.1%}")
for name, f in FAULTS.items():
    rng = np.random.default_rng(0)
    q = f(base, rng, start=START) if name == "sleeper" else f(base, rng)
    r, bad = held_rps(q)
    print(f"{'Elo + ' + name:<22}{r:>8.4f}{bad:>16.1%}")
