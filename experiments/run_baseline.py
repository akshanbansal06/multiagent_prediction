import numpy as np
from src.data.loader import load_matches
from src.data.feed import walk_forward
from src.agents.naive import NaiveAgent
from src.evaluation.metrics import rps

matches = load_matches()
agent = NaiveAgent()
scores = []
for match, result in walk_forward(matches):
    p = agent.predict(match)                      # forecast first
    scores.append(rps(p, result["outcome"]))      # then score it
    agent.update(match, result)                   # then learn from the result

print(f"matches: {len(scores)}  mean RPS: {np.mean(scores):.4f}")
