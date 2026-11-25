"""Compare trust methods on every fault. Composite settings are chosen on the training seasons only."""
import itertools
import numpy as np
import pandas as pd
from src.faults.faults import FAULTS
from src.trust.methods import EqualWeights, Hedge, DiscountedHedge
from src.trust.composite import Composite
from src.trust.aggregator import run_trust
from src.evaluation.metrics import rps_batch

TRAIN = {"1516", "1617", "1718", "1819"}
AGENTS = ["naive", "elo", "logreg", "xgb", "dixon_coles"]
FAULTY = AGENTS.index("elo")
GOOD = [AGENTS.index("elo"), AGENTS.index("dixon_coles")]   # the two strong honest agents

df = pd.read_csv("data/forecasts.csv", dtype={"season": str})
held = ~df["season"].isin(TRAIN).to_numpy()
train = ~held
y = df["outcome"].to_numpy()
honest = np.stack([df[[f"{a}_H", f"{a}_D", f"{a}_A"]].to_numpy() for a in AGENTS], axis=1)
START = int(np.argmax(held))
SLEEP_HELD, SLEEP_TRAIN = START + 1000, 800                   # where the sleeper turns bad


def scenario(name, sleep_at):
    f = honest.copy()
    if name != "none":
        p, rng = f[:, FAULTY], np.random.default_rng(0)
        f[:, FAULTY] = FAULTS[name](p, rng, start=sleep_at) if name == "sleeper" else FAULTS[name](p, rng)
    return f


def rps(p, mask):
    return float(rps_batch(p[mask], y[mask]).mean())


SCENARIOS = ["none"] + list(FAULTS)

# ---- 1. tune Composite on TRAINING seasons, averaged over all scenarios ----
print("Tuning Composite on 2015-16 to 2018-19 ...")
train_sets = {s: scenario(s, SLEEP_TRAIN) for s in SCENARIOS}
best, best_score = None, 9.0
for eta, lc, ls in itertools.product((200, 400, 800, 1600), (0.0, 0.1, 0.3), (0.0, 0.5, 1.0)):
    score = np.mean([rps(run_trust(f, y, Composite(5, eta, lc, ls))[0], train) for f in train_sets.values()])
    if score < best_score:
        best, best_score = (eta, lc, ls), score
print(f"Chosen: eta={best[0]}, lam_cal={best[1]}, lam_shock={best[2]} (train RPS {best_score:.4f})\n")

METHODS = {
    "Equal": lambda: EqualWeights(5),
    "Hedge": lambda: Hedge(5, eta=3.0),
    "Disc. Hedge": lambda: DiscountedHedge(5, eta=3.0, gamma=0.995),
    "Composite": lambda: Composite(5, *best),
    "Comp. all": lambda: Composite(5, best[0], 0.1, 0.5),   # every signal switched on
}

# ---- 2. held-out RPS for every method and fault ----
results, weights = {}, {}
for s in SCENARIOS:
    f = scenario(s, SLEEP_HELD)
    for m, make in METHODS.items():
        c, w = run_trust(f, y, make())
        results[(s, m)] = rps(c, held)
        weights[(s, m)] = w
best_single = min(rps(honest[:, i], held) for i in range(len(AGENTS)))
print(f"Held-out RPS, lower is better. Best single honest agent: {best_single:.4f}")
print(f"{'Fault':<15}" + "".join(f"{m:>13}" for m in METHODS))
for s in SCENARIOS:
    print(f"{s:<15}" + "".join(f"{results[(s, m)]:>13.4f}" for m in METHODS))
print(f"{'average':<15}" + "".join(f"{np.mean([results[(s, m)] for s in SCENARIOS]):>13.4f}" for m in METHODS))

# ---- 3. sleeper detection and false alarms ----
print(f"\n{'Method':<13}{'Weight before':>15}{'Delay':>8}{'False alarm rate':>19}")
for m in METHODS:
    w = weights[("sleeper", m)][:, FAULTY]
    before = w[SLEEP_HELD - 1]
    below = np.where(w[SLEEP_HELD:] < 0.05)[0]
    delay = "n/a" if before < 0.05 else (str(below[0]) if len(below) else "never")
    clean = weights[("none", m)][held][:, GOOD]
    false_alarm = (clean < 0.05).any(axis=1).mean()          # a strong honest agent wrongly ignored
    print(f"{m:<13}{before:>15.3f}{delay:>8}{false_alarm:>19.1%}")
print("\nDelay: matches after the sleeper turns bad until its weight is below 5%.")
print("False alarm rate: share of held-out matches, with no faults, where Elo or Dixon-Coles gets under 5% weight.")
