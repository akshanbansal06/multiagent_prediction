def walk_forward(matches):
    """Yield (match, result) in date order.
    The match dict has no result in it. Call agent.predict(match) before using result."""
    for row in matches.itertuples(index=False):
        match = {"match_id": row.match_id, "date": row.date, "season": row.season,
                 "home": row.home, "away": row.away}
        result = {"home_goals": row.home_goals, "away_goals": row.away_goals,
                  "outcome": int(row.outcome)}
        yield match, result
