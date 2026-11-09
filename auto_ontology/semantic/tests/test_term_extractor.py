# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for term extraction sanitization."""

from __future__ import annotations

from auto_ontology.semantic.models import (
    ColumnAttributeSpec,
    RawTableTermsResult,
    RawTermProposal,
    TermColumnRef,
)
from auto_ontology.semantic.term_extractor import _sanitize_result


def test_sanitize_uses_llm_display_names() -> None:
    specs = [
        ColumnAttributeSpec(
            source_column="amount",
            name="amount",
            datatype="numeric",
        ),
        ColumnAttributeSpec(
            source_column="status",
            name="status",
            datatype="text",
        ),
    ]
    raw = RawTableTermsResult(
        terms=[
            RawTermProposal(
                name="PurchaseOrder",
                description="A purchase order",
                attributes=[
                    TermColumnRef(
                        source_column="amount",
                        display_name="TotalAmount",
                    )
                ],
            )
        ]
    )

    result = _sanitize_result(raw, table={"name": "purchase_orders"}, specs=specs)

    assert len(result.terms) == 1
    assert result.terms[0].name == "Purchase Order"
    assigned = {a.source_column: a.display_name for a in result.terms[0].attributes}
    assert assigned == {"amount": "Total Amount", "status": "status"}
    assert specs[0].display_name == "Total Amount"
    assert specs[1].display_name == "status"


def test_sanitize_assigns_orphan_specs_to_primary_term() -> None:
    specs = [
        ColumnAttributeSpec(
            source_column="amount",
            name="amount",
            datatype="numeric",
        ),
        ColumnAttributeSpec(
            source_column="status",
            name="status",
            datatype="text",
        ),
    ]
    raw = RawTableTermsResult(
        terms=[
            RawTermProposal(
                name="Purchase Order",
                description="A purchase order",
                attributes=[
                    TermColumnRef(
                        source_column="amount",
                        display_name="Total Amount",
                    ),
                    TermColumnRef(
                        source_column="status",
                        display_name="Order Status",
                    ),
                ],
            )
        ]
    )

    result = _sanitize_result(raw, table={"name": "purchase_orders"}, specs=specs)

    assigned = {a.source_column: a.display_name for a in result.terms[0].attributes}
    assert assigned == {
        "amount": "Total Amount",
        "status": "Order Status",
    }
    assert specs[0].display_name == "Total Amount"
    assert specs[1].display_name == "Order Status"


# --- fork: output language + LLM synonyms ----------------------------------


def test_sanitize_keeps_llm_synonyms_cleaned() -> None:
    specs = [ColumnAttributeSpec(source_column="code", name="code", datatype="text")]
    raw = RawTableTermsResult(
        terms=[
            RawTermProposal(
                name="Tax Office",
                description="Териториална дирекция на НАП.",
                synonyms=["  данъчна служба ", "Tax Office", "ТД на НАП", "данъчна служба", ""],
                attributes=[TermColumnRef(source_column="code", display_name="Code")],
            )
        ]
    )

    result = _sanitize_result(raw, table={"name": "TS05_TAX_OFFICES"}, specs=specs)

    assert result.terms[0].description == "Териториална дирекция на НАП."
    assert result.terms[0].synonyms == ["данъчна служба", "ТД на НАП"]


def test_language_instruction_is_empty_by_default(monkeypatch) -> None:
    from auto_ontology.semantic.domain import output_language_instruction

    monkeypatch.delenv("SEMANTIC_OUTPUT_LANGUAGE", raising=False)
    monkeypatch.delenv("SEMANTIC_CONTEXT", raising=False)
    assert output_language_instruction() == ""


def test_language_instruction_asks_for_language_and_context(monkeypatch) -> None:
    from auto_ontology.semantic.domain import output_language_instruction

    monkeypatch.setenv("SEMANTIC_OUTPUT_LANGUAGE", "Bulgarian")
    monkeypatch.setenv("SEMANTIC_CONTEXT", "NRA registry exports; EGN is a personal ID.")
    text = output_language_instruction()
    assert "Write every description in Bulgarian" in text
    assert "Keep Term names and display_name labels in English" in text
    assert "NRA registry exports" in text


def test_extract_term_sends_the_instruction(monkeypatch) -> None:
    from auto_ontology.semantic import term_extractor

    monkeypatch.setenv("SEMANTIC_OUTPUT_LANGUAGE", "Bulgarian")
    seen: list[str] = []

    def _fake(_client, messages, model):
        seen.append(messages[0].content)
        return RawTableTermsResult(
            terms=[RawTermProposal(name="Tax Office", description="ТД", synonyms=["данъчна служба"])]
        )

    monkeypatch.setattr(term_extractor, "invoke_with_structured_output", _fake)
    monkeypatch.setattr(term_extractor, "get_llm_client", lambda **_: None)
    specs = [ColumnAttributeSpec(source_column="code", name="code", datatype="text")]

    result = term_extractor.extract_term({"name": "TS05_TAX_OFFICES"}, {}, specs)

    assert "Write every description in Bulgarian" in seen[0]
    assert result.terms[0].synonyms == ["данъчна служба"]
