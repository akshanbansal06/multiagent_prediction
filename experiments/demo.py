import numpy as np
from src.data.loader import load_matches
from src.data.feed import walk_forward
from src.agents.naive import NaiveAgent
from src.agents.elo import EloAgent
from src.evaluation.metrics import rps, rps_batch
from src.evaluation.benchmarks import odds_to_probs

TRAIN = {"1516", "1617", "1718", "1819"}      # used to tune Elo
LABEL = {0: "home win", 1: "draw", 2: "away win"}

matches = load_matches()
print(f"Loaded {len(matches)} Premier League matches, "
      f"{matches['season'].nunique()} seasons (2015-16 to 2025-26)\n")


def run(agent):
    rows = []
    for match, result in walk_forward(matches):      # date order, results hidden
        p = agent.predict(match)                     # 1. forecast
        rows.append((match, result["outcome"], p, rps(p, result["outcome"])))
        agent.update(match, result)                  # 2. only then learn
    return rows


naive = run(NaiveAgent())
elo = run(EloAgent(k=30, home_adv=50, cut=0.5, carryover=0.9))

odds = matches[["B365H", "B365D", "B365A"]].to_numpy(float)
book = rps_batch(odds_to_probs(odds), matches["outcome"].to_numpy())

held = np.array([r[0]["season"] not in TRAIN for r in naive])
scores = {
    "Naive (floor)": np.array([r[3] for r in naive])[held].mean(),
    "Elo": np.array([r[3] for r in elo])[held].mean(),
    "Bookmaker (ceiling)": book[held].mean(),
}
base = scores["Naive (floor)"]

print(f"Held-out seasons (2019-20 to 2025-26), {held.sum()} matches. Lower RPS is better.")
print(f"{'Forecaster':<22}{'RPS':>8}{'Skill vs naive':>18}")
for name, s in scores.items():
    print(f"{name:<22}{s:>8.4f}{(1 - s / base):>17.1%}")

print("\nElo forecasts for the last 5 matches in the data:")
for match, outcome, p, score in elo[-5:]:
    print(f"  {match['home']} v {match['away']}: "
          f"home {p[0]:.0%} / draw {p[1]:.0%} / away {p[2]:.0%}"
          f"  ->  actual: {LABEL[outcome]}  (RPS {score:.3f})")
