from src.data.loader import load_matches
from src.data.feed import walk_forward

def test_feed_is_chronological_and_hides_results():
    matches = load_matches()
    last = None
    for match, result in walk_forward(matches):
        assert "outcome" not in match and "home_goals" not in match
        if last is not None:
            assert match["date"] >= last
        last = match["date"]

def test_loaded_data_looks_right():
    m = load_matches()
    assert len(m) == 11 * 380
    assert m["outcome"].isin([0, 1, 2]).all()
    assert m.groupby("season").size().eq(380).all()
