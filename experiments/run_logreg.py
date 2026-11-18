import numpy as np
from src.data.loader import load_matches
from src.data.feed import walk_forward
from src.agents.naive import NaiveAgent
from src.agents.elo import EloAgent
from src.agents.logreg_agent import LogRegAgent
from src.evaluation.metrics import rps, rps_batch
from src.evaluation.benchmarks import odds_to_probs

TRAIN = {"1516", "1617", "1718", "1819"}
matches = load_matches()


def run(agent):
    rows = []
    for match, result in walk_forward(matches):
        p = agent.predict(match)                       # forecast first
        rows.append((match["season"], rps(p, result["outcome"])))
        agent.update(match, result)                    # then learn
    return rows


def held(rows):
    return float(np.mean([s for season, s in rows if season not in TRAIN]))


odds = matches[["B365H", "B365D", "B365A"]].to_numpy(float)
book = rps_batch(odds_to_probs(odds), matches["outcome"].to_numpy())
book_rows = list(zip(matches["season"], book))

results = {
    "Naive (floor)": run(NaiveAgent()),
    "Elo": run(EloAgent(k=30, home_adv=50, cut=0.5, carryover=0.9)),
    "LogReg": run(LogRegAgent()),
    "Bookmaker (ceiling)": book_rows,
}
base = held(results["Naive (floor)"])
print("Held-out seasons (2019-20 to 2025-26). Lower RPS is better.")
print(f"{'Forecaster':<22}{'RPS':>8}{'Skill vs naive':>18}")
for name, rows in results.items():
    print(f"{name:<22}{held(rows):>8.4f}{1 - held(rows) / base:>17.1%}")
