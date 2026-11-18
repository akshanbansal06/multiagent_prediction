import numpy as np
import pandas as pd
from src.data.loader import load_matches
from src.data.feed import walk_forward
from src.agents.naive import NaiveAgent
from src.agents.elo import EloAgent
from src.agents.logreg_agent import LogRegAgent
from src.agents.xgb_agent import XGBAgent
from src.agents.dixon_coles import DixonColesAgent
from src.evaluation.benchmarks import odds_to_probs

matches = load_matches()
agents = {
    "naive": NaiveAgent(),
    "elo": EloAgent(k=30, home_adv=50, cut=0.5, carryover=0.9),
    "logreg": LogRegAgent(),
    "xgb": XGBAgent(),
    "dixon_coles": DixonColesAgent(xi=0.003),
}

rows = []
for match, result in walk_forward(matches):
    row = {"match_id": match["match_id"], "date": match["date"],
           "season": match["season"], "home": match["home"], "away": match["away"],
           "outcome": result["outcome"]}
    for name, agent in agents.items():
        p = agent.predict(match)                 # forecast first
        row[f"{name}_H"], row[f"{name}_D"], row[f"{name}_A"] = p
        agent.update(match, result)              # then learn
    rows.append(row)

df = pd.DataFrame(rows)
book = odds_to_probs(matches[["B365H", "B365D", "B365A"]].to_numpy(float))
df["book_H"], df["book_D"], df["book_A"] = book[:, 0], book[:, 1], book[:, 2]
df.to_csv("data/forecasts.csv", index=False)
print(df.shape)
print(df.filter(regex="_H$|_D$|_A$").describe().loc[["min", "max"]].round(3))
