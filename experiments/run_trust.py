import time
import numpy as np
import pandas as pd
from src.faults.faults import FAULTS
from src.trust.methods import EqualWeights, Hedge
from src.trust.aggregator import run_trust
from src.evaluation.metrics import rps_batch

TRAIN = {"1516", "1617", "1718", "1819"}
AGENTS = ["naive", "elo", "logreg", "xgb", "dixon_coles"]
FAULTY = "elo"                                   # the agent we corrupt
ETA = 1.0                                        # fixed for now; tune on training seasons later

df = pd.read_csv("data/forecasts.csv", dtype={"season": str})
held = ~df["season"].isin(TRAIN).to_numpy()
y = df["outcome"].to_numpy()
honest = np.stack([df[[f"{a}_H", f"{a}_D", f"{a}_A"]].to_numpy() for a in AGENTS], axis=1)
START = int(np.argmax(held))                     # first held-out match
f_idx = AGENTS.index(FAULTY)


def scenario(name):
    f = honest.copy()
    if name == "none":
        return f
    rng = np.random.default_rng(0)
    p = f[:, f_idx]
    f[:, f_idx] = FAULTS[name](p, rng, start=START + 1000) if name == "sleeper" else FAULTS[name](p, rng)
    return f


def held_rps(p):
    return float(rps_batch(p[held], y[held]).mean())


print(f"Faulty agent: {FAULTY}. Held-out seasons, lower RPS is better.")
print(f"Best single honest agent (Dixon-Coles): {held_rps(honest[:, AGENTS.index('dixon_coles')]):.4f}\n")
print(f"{'Fault':<15}{'Equal RPS':>11}{'Hedge RPS':>11}{'Hedge weight on faulty':>25}")
t0 = time.time()
for name in ["none"] + list(FAULTS):
    f = scenario(name)
    eq, _ = run_trust(f, y, EqualWeights(len(AGENTS)))
    hd, w = run_trust(f, y, Hedge(len(AGENTS), eta=ETA))
    print(f"{name:<15}{held_rps(eq):>11.4f}{held_rps(hd):>11.4f}{w[held, f_idx].mean():>25.3f}")
print(f"\nTotal time: {time.time() - t0:.1f}s")
