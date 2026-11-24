import itertools
import numpy as np
import pandas as pd
from src.faults.faults import flipped
from src.trust.methods import Hedge, DiscountedHedge
from src.trust.aggregator import run_trust
from src.evaluation.metrics import rps_batch

TRAIN = {"1516", "1617", "1718", "1819"}
AGENTS = ["naive", "elo", "logreg", "xgb", "dixon_coles"]
df = pd.read_csv("data/forecasts.csv", dtype={"season": str})
held = ~df["season"].isin(TRAIN).to_numpy()
train = ~held
y = df["outcome"].to_numpy()
honest = np.stack([df[[f"{a}_H", f"{a}_D", f"{a}_A"]].to_numpy() for a in AGENTS], axis=1)
START = int(np.argmax(held))
ELO = AGENTS.index("elo")


def rps(p, mask):
    return float(rps_batch(p[mask], y[mask]).mean())


def delay(w, start, threshold=0.05):
    """Matches after the fault starts until the faulty agent's weight falls below threshold."""
    below = np.where(w[start:, ELO] < threshold)[0]
    return int(below[0]) if len(below) else None


# sleeper: honest Elo until 1000 matches into the held-out seasons, then flipped
sleep = honest.copy()
s0 = START + 1000
sleep[s0:, ELO] = flipped(honest[s0:, ELO])

print("Simple average of Elo and Dixon-Coles (held-out):",
      f"{rps(honest[:, [ELO, 4]].mean(1), held):.4f}\n")

# choose settings on the TRAINING seasons only, with no faults
configs = [("hedge", eta, 1.0) for eta in (0.1, 0.3, 1.0, 3.0)] + \
          [("discounted", eta, g) for eta, g in itertools.product((0.3, 1.0, 3.0), (0.99, 0.995, 0.999))]
print(f"{'Method':<12}{'eta':>6}{'gamma':>8}{'Train RPS':>11}{'Held RPS':>10}{'Sleeper RPS':>13}{'Delay':>8}")
for kind, eta, g in configs:
    make = (lambda: Hedge(5, eta)) if kind == "hedge" else (lambda: DiscountedHedge(5, eta, g))
    c, _ = run_trust(honest, y, make())
    cs, ws = run_trust(sleep, y, make())
    d = delay(ws, s0)
    print(f"{kind:<12}{eta:>6}{g:>8}{rps(c, train):>11.4f}{rps(c, held):>10.4f}"
          f"{rps(cs, held):>13.4f}{(str(d) if d is not None else 'never'):>8}")
print("\nPick the row with the best Train RPS. Held and sleeper columns are for reporting only.")
