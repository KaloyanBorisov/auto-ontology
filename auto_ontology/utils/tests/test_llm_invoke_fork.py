# Fork: OpenRouter / generic OpenAI-compatible gateways use ChatOpenAI.

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from auto_ontology.utils import llm_invoke

OPENROUTER = "https://openrouter.ai/api/v1"


def _build(**kw):
    return llm_invoke._build_client(
        model=kw.pop("model", "deepseek/deepseek-v4-flash-0731"),
        api_key="sk-or-test",
        base_url=kw.pop("base_url", OPENROUTER),
        temperature=0.0,
        max_tokens=100,
    )


def test_openrouter_gets_the_openai_client() -> None:
    client = _build()
    assert isinstance(client, ChatOpenAI)
    assert client.temperature == 0.0


def test_extra_body_reaches_the_client(monkeypatch) -> None:
    monkeypatch.setattr(llm_invoke, "_LLM_EXTRA_BODY", {"reasoning": {"enabled": False}})
    assert _build().extra_body == {"reasoning": {"enabled": False}}


def test_llm_client_can_force_openai_for_any_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(llm_invoke, "_LLM_CLIENT", "openai")
    assert llm_invoke._uses_openai_client("http://localhost:8000/v1")


def test_llm_client_nvidia_keeps_upstream_routing(monkeypatch) -> None:
    monkeypatch.setattr(llm_invoke, "_LLM_CLIENT", "nvidia")
    assert not llm_invoke._uses_openai_client(OPENROUTER)


def test_auto_leaves_nvidia_endpoints_alone() -> None:
    assert not llm_invoke._uses_openai_client("https://integrate.api.nvidia.com/v1")


@pytest.mark.parametrize("raw", ["not json", "[1, 2]"])
def test_bad_extra_body_is_ignored(monkeypatch, raw: str) -> None:
    monkeypatch.setenv("LLM_EXTRA_BODY", raw)
    assert llm_invoke._load_extra_body() is None


def test_extra_body_parses_json(monkeypatch) -> None:
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"provider": {"require_parameters": true}}')
    assert llm_invoke._load_extra_body() == {"provider": {"require_parameters": True}}
