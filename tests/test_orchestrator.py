import json

from tests.helpers import FakeLLM, FakeSearch
from llm import LLMError
from orchestrator import Orchestrator


def review(decision, score, feedback="fb", issues=None):
    return json.dumps({"score": score, "decision": decision, "issues": issues or [], "feedback": feedback})


NO_SEARCH = json.dumps({"needs_search": False, "queries": [], "reason": "timeless"})
NEEDS_SEARCH = json.dumps({"needs_search": True, "queries": ["icc 2027 world cup groups"], "reason": "schedule"})


def make(gen_replies, rev_replies, max_iterations=3, plan=NO_SEARCH, search=None):
    return Orchestrator(
        generator_llm=FakeLLM(gen_replies),
        reviewer_llm=FakeLLM(rev_replies),
        planner_llm=FakeLLM([plan]),
        search=search or FakeSearch(available=False),
        max_iterations=max_iterations,
        today="2026-10-06",
    )


def test_approved_first_round():
    o = make(["draft1"], [review("APPROVED", 9)])
    r = o.run("q")
    assert r["approved"] and r["final_answer"] == "draft1" and r["iterations_used"] == 1
    assert r["best_score"] == 9


def test_revision_then_approval():
    o = make(["draft1", "draft2"], [review("NEEDS_REVISION", 5, "add detail"), review("APPROVED", 9)])
    r = o.run("q")
    assert r["approved"] and r["final_answer"] == "draft2" and r["iterations_used"] == 2
    # feedback was passed to the solver on revision
    assert "add detail" in o.generator_llm.prompts[1]


def test_reviewer_sees_original_request():
    o = make(["d"], [review("APPROVED", 9)])
    o.run("What is DNS?")
    assert "What is DNS?" in o.reviewer_llm.prompts[0]


def test_limit_returns_best_scored_draft_not_last():
    o = make(
        ["d1", "d2", "d3"],
        [review("NEEDS_REVISION", 7), review("NEEDS_REVISION", 3), review("NEEDS_REVISION", 5)],
        max_iterations=3,
    )
    r = o.run("q")
    assert not r["approved"] and r["final_answer"] == "d1" and r["best_score"] == 7


def test_stagnation_stops_loop():
    o = make(["same", "same"], [review("NEEDS_REVISION", 5), review("APPROVED", 9)])
    r = o.run("q")
    assert r["iterations_used"] == 1 and not r["approved"]


def test_reviewer_outage_returns_draft_with_flag():
    o = make(["draft"], [LLMError("all down")])
    r = o.run("q")
    assert r["final_answer"] == "draft" and r["review_unavailable"] and not r["approved"]


def test_generator_outage_on_first_call_propagates():
    o = make([LLMError("down")], [])
    try:
        o.run("q")
        assert False
    except LLMError:
        pass


def test_generator_failure_during_revision_keeps_best_draft():
    o = make(["d1", LLMError("down")], [review("NEEDS_REVISION", 6)])
    r = o.run("q")
    assert r["final_answer"] == "d1" and not r["approved"]


def test_events_are_emitted_in_order():
    seen = []
    o = make(["d1", "d2"], [review("NEEDS_REVISION", 5), review("APPROVED", 9)])
    o.run("q", on_event=lambda stage, d: seen.append(stage))
    assert seen == ["planning", "drafting", "reviewing", "revising", "reviewing", "done"]


def test_unparseable_review_never_counts_as_approval():
    o = make(["d"] * 3, ["looks great!!"] * 3, max_iterations=2)
    # stagnation (identical drafts) ends it after round 1; either way: never approved
    assert o.run("q")["approved"] is False


# ---- grounding ---------------------------------------------------------------
SRC = [("ICC schedule", "https://icc.example/schedule", "Pakistan are in Group A.")]


def test_timeless_question_skips_search():
    search = FakeSearch(SRC)
    o = make(["d"], [review("APPROVED", 9)], search=search)
    r = o.run("What is DNS?")
    assert r["mode"] == "general" and search.queries == [] and r["sources"] == []


def test_grounded_run_gives_sources_to_solver_and_reviewer():
    search = FakeSearch(SRC)
    o = make(["d [1]"], [review("APPROVED", 9)], plan=NEEDS_SEARCH, search=search)
    r = o.run("Which group is Pakistan in?")
    assert r["mode"] == "grounded" and r["sources"][0]["url"] == "https://icc.example/schedule"
    assert search.queries == [["icc 2027 world cup groups"]]
    assert "Pakistan are in Group A." in o.generator_llm.prompts[0]
    assert "Pakistan are in Group A." in o.reviewer_llm.prompts[0]
    assert "2026-10-06" in o.generator_llm.prompts[0]


def test_no_search_key_marks_unverified_and_warns_solver():
    o = make(["d"], [review("APPROVED", 9)], plan=NEEDS_SEARCH, search=FakeSearch(available=False))
    r = o.run("Cheapest 2027 match?")
    assert r["mode"] == "unverified" and "no TAVILY_API_KEY" in r["search_error"]
    assert "live web search was unavailable" in o.generator_llm.prompts[0]
    assert "live web search was unavailable" in o.reviewer_llm.prompts[0]


def test_search_failure_degrades_to_unverified_not_crash():
    o = make(["d"], [review("APPROVED", 9)], plan=NEEDS_SEARCH, search=FakeSearch(error="HTTP 500"))
    r = o.run("q")
    assert r["mode"] == "unverified" and r["search_error"] == "HTTP 500" and r["final_answer"] == "d"


def test_empty_search_results_are_unverified():
    o = make(["d"], [review("APPROVED", 9)], plan=NEEDS_SEARCH, search=FakeSearch([]))
    assert o.run("q")["mode"] == "unverified"


def test_planner_outage_falls_back_to_heuristic():
    from llm import LLMError

    o = make(["d"], [review("APPROVED", 9)], search=FakeSearch(SRC))
    o.planner_llm.replies = [LLMError("down")]
    r = o.run("Cheapest ticket for the 2027 world cup?")  # heuristic: time-sensitive
    assert r["mode"] == "grounded"


def test_sources_reach_the_revision_prompt_too():
    o = make(["d1", "d2"], [review("NEEDS_REVISION", 5, "cite sources"), review("APPROVED", 9)],
             plan=NEEDS_SEARCH, search=FakeSearch(SRC))
    o.run("q")
    assert "Pakistan are in Group A." in o.generator_llm.prompts[1]
