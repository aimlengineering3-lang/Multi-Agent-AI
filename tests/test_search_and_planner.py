from tests.helpers import FakeHTTPError  # noqa: F401  (ensures env defaults are set)
from Agents.planner import PlannerAgent
from search import SearchError, Source, WebSearch, format_sources


def fake_post(payload_by_query):
    calls = []

    def post(url, headers, payload, timeout):
        calls.append((url, headers, payload))
        return payload_by_query[payload["query"]]

    post.calls = calls
    return post


def test_search_parses_results_and_sends_bearer_key():
    post = fake_post({"q": {"results": [
        {"title": "A", "url": "https://a", "content": "alpha"},
        {"title": "B", "url": "https://b", "content": ""},      # dropped: no content
        {"title": "C", "url": "", "content": "gamma"},          # dropped: no url
    ]}})
    ws = WebSearch(api_key="tvly-x", http_post=post)
    out = ws.search("q")
    assert [s.url for s in out] == ["https://a"]
    assert post.calls[0][1]["Authorization"] == "Bearer tvly-x"
    assert post.calls[0][2]["max_results"] == ws.max_results


def test_search_many_dedupes_urls():
    post = fake_post({
        "q1": {"results": [{"title": "A", "url": "https://a", "content": "x"}]},
        "q2": {"results": [{"title": "A", "url": "https://a", "content": "x"},
                           {"title": "B", "url": "https://b", "content": "y"}]},
    })
    out = WebSearch(api_key="k", http_post=post).search_many(["q1", "q2"])
    assert [s.url for s in out] == ["https://a", "https://b"]


def test_search_without_key_raises():
    try:
        WebSearch(api_key="", http_post=fake_post({})).search("q")
        assert False
    except SearchError:
        pass


def test_search_many_raises_only_if_everything_failed():
    def post(url, headers, payload, timeout):
        if payload["query"] == "bad":
            raise SearchError("HTTP 500")
        return {"results": [{"title": "A", "url": "https://a", "content": "x"}]}

    ws = WebSearch(api_key="k", http_post=post)
    assert len(ws.search_many(["bad", "good"])) == 1
    try:
        ws.search_many(["bad"])
        assert False
    except SearchError:
        pass


def test_format_sources_numbers_and_truncates():
    text = format_sources([Source("T1", "https://1", "x" * 2000), Source("T2", "https://2", "short")], snippet_chars=50)
    assert text.startswith("[1] T1") and "[2] T2" in text and "..." in text and len(text) < 400


def test_planner_parses_json_plan():
    p = PlannerAgent.parse_plan('{"needs_search": true, "queries": ["a", "b", "c"], "reason": "x"}', "q", max_queries=2)
    assert p["needs_search"] and p["queries"] == ["a", "b"]


def test_planner_needs_search_without_queries_uses_question():
    p = PlannerAgent.parse_plan('{"needs_search": true, "queries": []}', "cheapest flight to Cape Town")
    assert p["queries"] == ["cheapest flight to Cape Town"]


def test_planner_garbage_falls_back_to_heuristic():
    assert PlannerAgent.parse_plan("lol idk", "Cricket World Cup 2027 schedule")["needs_search"] is True
    assert PlannerAgent.parse_plan("lol idk", "Explain recursion simply")["needs_search"] is False


def test_planner_timeless_returns_no_queries():
    p = PlannerAgent.parse_plan('{"needs_search": false, "queries": ["ignored"]}', "What is a REST API?")
    assert p["needs_search"] is False and p["queries"] == []
