from tests.helpers import FakeClient, FakeHTTPError
import config
from llm import LLMService, LLMError


def make_service(clients, chain=("groq", "gemini")):
    LLMService._cooldown_until.clear()
    return LLMService(
        chain=list(chain),
        client_factory=lambda spec: clients[spec.name],
        sleep=lambda s: None,
    )


def test_success_on_first_provider():
    svc = make_service({"groq": FakeClient(["hello"]), "gemini": FakeClient(["unused"])})
    r = svc.complete("hi", system_prompt="be nice")
    assert r.text == "hello" and r.provider == "groq"
    assert r.prompt_tokens == 10


def test_system_and_user_roles_are_separate():
    groq = FakeClient(["ok"])
    make_service({"groq": groq, "gemini": FakeClient([])}).complete("task", system_prompt="rules")
    roles = [m["role"] for m in groq.calls[0]["messages"]]
    assert roles == ["system", "user"]


def test_rate_limit_fails_over_immediately():
    groq = FakeClient([FakeHTTPError(429)])
    gem = FakeClient(["from gemini"])
    r = make_service({"groq": groq, "gemini": gem}).complete("hi")
    assert r.provider == "gemini" and len(groq.calls) == 1  # no pointless retry on 429
    assert any("groq" in a for a in r.attempts)


def test_rate_limited_provider_is_cooled_down_for_next_call():
    groq = FakeClient([FakeHTTPError(429, retry_after=60), "should not be used"])
    gem = FakeClient(["one", "two"])
    svc = make_service({"groq": groq, "gemini": gem})
    svc.complete("a")
    r2 = svc.complete("b")
    assert r2.provider == "gemini" and len(groq.calls) == 1


def test_permanent_error_skips_provider_without_retry():
    groq = FakeClient([FakeHTTPError(401)])
    r = make_service({"groq": groq, "gemini": FakeClient(["ok"])}).complete("hi")
    assert r.provider == "gemini" and len(groq.calls) == 1


def test_transient_error_is_retried_then_succeeds():
    groq = FakeClient([FakeHTTPError(503), "recovered"])
    r = make_service({"groq": groq, "gemini": FakeClient([])}).complete("hi")
    assert r.text == "recovered" and len(groq.calls) == 2


def test_empty_response_triggers_failover():
    groq = FakeClient(["", ""])  # both retries empty
    r = make_service({"groq": groq, "gemini": FakeClient(["real"])}).complete("hi")
    assert r.provider == "gemini"


def test_think_blocks_are_stripped():
    r = make_service({"groq": FakeClient(["<think>secret</think>Answer."]), "gemini": FakeClient([])}).complete("q")
    assert r.text == "Answer."


def test_all_providers_fail_raises_llmerror():
    svc = make_service({"groq": FakeClient([FakeHTTPError(401)]), "gemini": FakeClient([FakeHTTPError(401)])})
    try:
        svc.complete("hi")
        assert False, "expected LLMError"
    except LLMError as e:
        assert "groq" in str(e) and "gemini" in str(e)


def test_no_configured_provider_gives_helpful_error():
    LLMService._cooldown_until.clear()
    svc = LLMService(chain=["nonexistent"], client_factory=lambda s: None)
    try:
        svc.complete("hi")
        assert False
    except LLMError as e:
        assert "API key" in str(e) or "provider" in str(e)
