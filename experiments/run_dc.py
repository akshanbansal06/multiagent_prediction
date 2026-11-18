import time
import numpy as np
from src.data.loader import load_matches
from src.data.feed import walk_forward
from src.agents.naive import NaiveAgent
from src.agents.elo import EloAgent
from src.agents.logreg_agent import LogRegAgent
from src.agents.dixon_coles import DixonColesAgent
from src.evaluation.metrics import rps, rps_batch
from src.evaluation.benchmarks import odds_to_probs

TRAIN = {"1516", "1617", "1718", "1819"}
matches = load_matches()


def run(agent):
    rows = []
    for match, result in walk_forward(matches):
        p = agent.predict(match)
        rows.append((match["season"], rps(p, result["outcome"])))
        agent.update(match, result)
    return rows


def mean_over(rows, train):
    return float(np.mean([s for season, s in rows if (season in TRAIN) == train]))


# tune the time-decay xi on the training seasons ONLY
print("Tuning xi on 2015-16 to 2018-19 ...")
best_xi, best = None, 9.0
for xi in (0.0005, 0.0015, 0.003):
    score = mean_over(run(DixonColesAgent(xi=xi)), train=True)
    print(f"  xi={xi:<7} train RPS {score:.4f}")
    if score < best:
        best_xi, best = xi, score
print(f"Chosen xi = {best_xi}")

odds = matches[["B365H", "B365D", "B365A"]].to_numpy(float)
book = rps_batch(odds_to_probs(odds), matches["outcome"].to_numpy())

t0 = time.time()
dc_rows = run(DixonColesAgent(xi=best_xi))
dc_time = time.time() - t0

results = {
    "Naive (floor)": run(NaiveAgent()),
    "Elo": run(EloAgent(k=30, home_adv=50, cut=0.5, carryover=0.9)),
    "LogReg": run(LogRegAgent()),
    "Dixon-Coles": dc_rows,
    "Bookmaker (ceiling)": list(zip(matches["season"], book)),
}
base = mean_over(results["Naive (floor)"], train=False)
print("\nHeld-out seasons (2019-20 to 2025-26). Lower RPS is better.")
print(f"{'Forecaster':<22}{'RPS':>8}{'Skill vs naive':>18}")
for name, rows in results.items():
    s = mean_over(rows, train=False)
    print(f"{name:<22}{s:>8.4f}{1 - s / base:>17.1%}")
print(f"\nDixon-Coles full walk-forward run: {dc_time:.1f}s")
