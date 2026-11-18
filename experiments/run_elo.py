import itertools
import numpy as np
from src.data.loader import load_matches
from src.data.feed import walk_forward
from src.agents.elo import EloAgent
from src.agents.naive import NaiveAgent
from src.evaluation.metrics import rps, rps_batch
from src.evaluation.benchmarks import odds_to_probs

TRAIN = {"1516", "1617", "1718", "1819"}

matches = load_matches()


def run(agent, df):
    """Walk-forward: predict, score, then learn. Returns a list of (season, RPS)."""
    rows = []
    for match, result in walk_forward(df):
        p = agent.predict(match)
        rows.append((match["season"], rps(p, result["outcome"])))
        agent.update(match, result)
    return rows


def mean_rps(rows, seasons):
    return float(np.nanmean([s for season, s in rows if season in seasons]))


# 1. Tune on the early seasons only
train_df = matches[matches["season"].isin(TRAIN)]
results = []
for k, hadv, cut, carry in itertools.product([10, 20, 30], [50, 80], [0.5, 0.7], [0.7, 0.9]):
    rows = run(EloAgent(k=k, home_adv=hadv, cut=cut, carryover=carry), train_df)
    results.append((mean_rps(rows, TRAIN), k, hadv, cut, carry))
results.sort()
print("Best 3 settings on training seasons (RPS, k, home_adv, cut, carryover):")
for r in results[:3]:
    print("  ", [round(x, 4) if isinstance(x, float) else x for x in r])

_, k, hadv, cut, carry = results[0]

# 2. Run chosen settings over all seasons, report held-out seasons only
elo_rows = run(EloAgent(k=k, home_adv=hadv, cut=cut, carryover=carry), matches)
naive_rows = run(NaiveAgent(), matches)

odds = matches[["B365H", "B365D", "B365A"]].to_numpy(float)
book = rps_batch(odds_to_probs(odds), matches["outcome"].to_numpy())
book_rows = list(zip(matches["season"], book))

test = set(matches["season"]) - TRAIN
print(f"\nHeld-out seasons ({len(test)} seasons), mean RPS:")
print(f"  naive      {mean_rps(naive_rows, test):.4f}")
print(f"  elo        {mean_rps(elo_rows, test):.4f}")
print(f"  bookmaker  {mean_rps(book_rows, test):.4f}")

print("\nPer season (naive / elo / bookmaker):")
for s in sorted(test):
    print(f"  {s}  {mean_rps(naive_rows, {s}):.4f}  {mean_rps(elo_rows, {s}):.4f}  {mean_rps(book_rows, {s}):.4f}")
