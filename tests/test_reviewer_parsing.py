from Agents.reviewer import ReviewerAgent

parse = ReviewerAgent.parse_decision


def test_json_approved():
    r = parse('{"score": 9, "decision": "APPROVED", "issues": [], "feedback": "Clear and correct."}')
    assert r["decision"] == "APPROVED" and r["score"] == 9 and r["issues"] == []


def test_json_inside_markdown_fence():
    raw = '```json\n{"score": 5, "decision": "NEEDS_REVISION", "issues": ["no example"], "feedback": "Add one."}\n```'
    r = parse(raw)
    assert r["decision"] == "NEEDS_REVISION" and "no example" in r["feedback"]


def test_approved_but_low_score_is_downgraded():
    r = parse('{"score": 4, "decision": "APPROVED", "issues": [], "feedback": "ok"}')
    assert r["decision"] == "NEEDS_REVISION"


def test_score_is_clamped():
    assert parse('{"score": 99, "decision": "APPROVED", "feedback": "x"}')["score"] == 10


def test_legacy_format_still_works():
    r = parse("DECISION: APPROVED\nFEEDBACK: Looks good.")
    assert r["decision"] == "APPROVED" and r["feedback"] == "Looks good."


def test_garbage_defaults_to_needs_revision():
    r = parse("I think this is fine, honestly!")
    assert r["decision"] == "NEEDS_REVISION"


def test_legacy_ambiguous_decision_is_not_approval():
    assert parse("DECISION: NEEDS_REVISION (not APPROVED)\nFEEDBACK: x")["decision"] == "NEEDS_REVISION"


def test_malformed_json_falls_back_safely():
    assert parse('{"decision": "APPROVED", ')["decision"] == "NEEDS_REVISION"
