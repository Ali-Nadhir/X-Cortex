"""Real grammar checks with a tiny local tokenizer; no model weights needed."""
from __future__ import annotations

import json
import sys

import pytest

from xcortex.report.constraints import build_report_constraint
from xcortex.report.schema import ClinicalReport, REPORT_SCHEMA, report_schema_for_options, validate_report_for_options


@pytest.fixture
def tokenizer():
    pytest.importorskip("lmformatenforcer")
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: i for i, token in enumerate(["<eos>", "<unk>", "<pad>", *alphabet])}
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>", pad_token="<pad>")


def prefix_ids(tokenizer, text):
    import torch
    return torch.tensor([tokenizer.pad_token_id, *tokenizer.encode(text, add_special_tokens=False)])


def walk_prefix(tokenizer, constraint, text):
    import torch
    ids = prefix_ids(tokenizer, "")
    for token_id in tokenizer.encode(text, add_special_tokens=False):
        assert token_id in constraint(0, ids), tokenizer.decode(ids)
        ids = torch.cat((ids, torch.tensor([token_id])))
    return ids


def complete_report():
    return {
        "study": {"region": "chest", "history": None, "view": None},
        "triage_summary": {"abnormality_probability": 0.2811, "threshold": 0.33, "band": "low", "calibrated": False, "score_type": "max_pathology_score", "method": "test classifier"},
        "findings": [], "impression": "Integration evidence reviewed.", "recommendations": [], "limitations": ["Integration only"],
    }


def test_report_grammar_rejects_thought_preamble_and_premature_eos(tokenizer):
    allowed = build_report_constraint(tokenizer)(0, prefix_ids(tokenizer, ""))
    assert tokenizer.convert_tokens_to_ids("{") in allowed
    assert tokenizer.convert_tokens_to_ids("<") not in allowed
    assert tokenizer.eos_token_id not in allowed


def test_report_grammar_accepts_complete_schema_with_nulls_and_provenance(tokenizer):
    report = complete_report()
    text = json.dumps(report, separators=(",", ":"))
    constraint = build_report_constraint(tokenizer)
    ids = walk_prefix(tokenizer, constraint, text)
    assert tokenizer.eos_token_id in constraint(0, ids)
    assert ClinicalReport.model_validate(json.loads(tokenizer.decode(ids[1:])))


def test_report_grammar_rejects_unknown_properties(tokenizer):
    constraint = build_report_constraint(tokenizer)
    # Required fields may appear in any order, but arbitrary extra fields cannot.
    ids = walk_prefix(tokenizer, constraint, '{"')
    assert tokenizer.convert_tokens_to_ids("x") not in constraint(0, ids)


def test_tokenizer_tree_is_cached_but_each_request_has_a_fresh_parser(tokenizer):
    first, second = build_report_constraint(tokenizer), build_report_constraint(tokenizer)
    assert first.token_enforcer.tokenizer_tree is second.token_enforcer.tokenizer_tree
    assert first.token_enforcer is not second.token_enforcer
    first_ids = walk_prefix(tokenizer, first, json.dumps(complete_report()))
    assert tokenizer.eos_token_id in first(0, first_ids)
    assert tokenizer.eos_token_id not in second(0, prefix_ids(tokenizer, ""))


def test_registered_gemma_end_of_turn_and_eos_are_only_allowed_after_json(tokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    constraint = build_report_constraint(tokenizer)
    initial = constraint(0, prefix_ids(tokenizer, ""))
    assert tokenizer.eos_token_id not in initial
    assert end_of_turn not in initial
    final_ids = walk_prefix(tokenizer, constraint, json.dumps(complete_report()))
    final = constraint(0, final_ids)
    assert tokenizer.eos_token_id in final
    assert end_of_turn in final


def test_escaped_unicode_and_newlines_preserve_json_and_byte_decoder(tokenizer):
    report = complete_report()
    report["impression"] = "Line one\nline two: \u03bcm\u00b2, \u03c0."
    text = json.dumps(report, ensure_ascii=True, separators=(",", ":"))
    constraint = build_report_constraint(tokenizer)
    ids = walk_prefix(tokenizer, constraint, text)
    assert tokenizer.eos_token_id in constraint(0, ids)
    assert tokenizer.decode(ids[1:], clean_up_tokenization_spaces=False) == text
    # Byte-level decoding still reconstructs multibyte characters exactly. JSON
    # Unicode escapes work even for a toy vocabulary with no full Unicode tokens.
    raw = report["impression"]
    assert constraint.token_enforcer.decoder(tokenizer.encode(raw, add_special_tokens=False)) == raw


def test_gemma_end_of_turn_registered_as_added_special_is_preserved(tokenizer):
    from tokenizers import AddedToken
    tokenizer.add_tokens([AddedToken("<end_of_turn>", special=True)])
    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    assert end_of_turn not in tokenizer.all_special_ids
    constraint = build_report_constraint(tokenizer)
    assert end_of_turn not in constraint(0, prefix_ids(tokenizer, ""))
    ids = walk_prefix(tokenizer, constraint, json.dumps(complete_report()))
    assert end_of_turn in constraint(0, ids)


def test_bridge_does_not_import_incompatible_lmfe_transformers_module(tokenizer, monkeypatch):
    monkeypatch.setitem(sys.modules, "lmformatenforcer.integrations.transformers", None)
    assert callable(build_report_constraint(tokenizer))


def test_missing_tokenizer_eos_fails_before_generation(tokenizer):
    tokenizer.eos_token = None
    with pytest.raises(ValueError, match="requires an EOS"):
        build_report_constraint(tokenizer)


def test_missing_grammar_dependency_is_actionable_without_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "lmformatenforcer", None)
    with pytest.raises(RuntimeError, match="models extra"):
        build_report_constraint(object())


def test_mcq_grammar_requires_supplied_choice_and_rejects_null_or_other_ids(tokenizer):
    schema = report_schema_for_options({"A": "Absent", "C": "Present"})
    constraint = build_report_constraint(tokenizer, schema)
    ids = walk_prefix(tokenizer, constraint, '{"answer_choice":')
    assert tokenizer.convert_tokens_to_ids('"') in constraint(0, ids)
    assert tokenizer.convert_tokens_to_ids("n") not in constraint(0, ids)
    ids = walk_prefix(tokenizer, constraint, '{"answer_choice":"')
    allowed = constraint(0, ids)
    assert tokenizer.convert_tokens_to_ids("A") in allowed
    assert tokenizer.convert_tokens_to_ids("C") in allowed
    assert tokenizer.convert_tokens_to_ids("B") not in allowed
    assert tokenizer.convert_tokens_to_ids("F") not in allowed

    base = complete_report()
    valid = {"answer_choice": "C", **{key: base[key] for key in ("study", "findings", "impression", "recommendations")},
             "limitations": base["limitations"], "triage_summary": base["triage_summary"]}
    final_ids = walk_prefix(tokenizer, constraint, json.dumps(valid))
    assert tokenizer.eos_token_id in constraint(0, final_ids)

    partial = build_report_constraint(tokenizer, schema)
    ids = walk_prefix(tokenizer, partial, '{"answer_choice":"C"')
    assert tokenizer.convert_tokens_to_ids("}") not in partial(0, ids)
    assert tokenizer.eos_token_id not in partial(0, ids)


def test_request_mcq_schema_does_not_mutate_generic_contract():
    generic = json.dumps(REPORT_SCHEMA, sort_keys=True)
    first = report_schema_for_options({"A": "First", "B": "Second"})
    second = report_schema_for_options({"D": "Other", "F": "Last"})
    assert first["properties"]["answer_choice"]["enum"] == ["A", "B"]
    assert second["properties"]["answer_choice"]["enum"] == ["D", "F"]
    assert json.dumps(REPORT_SCHEMA, sort_keys=True) == generic
    assert "answer_choice" not in report_schema_for_options()["required"]


def _compact_report(choice="A"):
    return {"answer_choice": choice, "study": {"region": "chest"}, "findings": [],
            "impression": "The selected option is supported by the query image.",
            "recommendations": [], "limitations": ["Development example only."]}


def test_actual_token_enforcer_requires_choice_first_and_preserves_whitespace_limit(tokenizer):
    constraint = build_report_constraint(tokenizer, report_schema_for_options({"A": "one", "B": "two"}))
    assert constraint.token_enforcer.root_parser.config.force_json_field_order is True
    assert constraint.token_enforcer.root_parser.config.max_consecutive_whitespaces == 2
    ids = walk_prefix(tokenizer, constraint, '{"')
    allowed = constraint(0, ids)
    assert tokenizer.convert_tokens_to_ids("a") in allowed
    assert tokenizer.convert_tokens_to_ids("s") not in allowed
    assert tokenizer.convert_tokens_to_ids("f") not in allowed

    space = tokenizer.encode(" ", add_special_tokens=False)[0]
    whitespace = build_report_constraint(tokenizer, report_schema_for_options({"A": "one", "B": "two"}))
    ids = walk_prefix(tokenizer, whitespace, "{  ")
    assert space not in whitespace(0, ids)

    generic = build_report_constraint(tokenizer)
    ids = walk_prefix(tokenizer, generic, '{"')
    assert tokenizer.convert_tokens_to_ids("s") in generic(0, ids)
    assert generic.token_enforcer.root_parser.config.force_json_field_order is False


def test_mcq_completion_waits_for_entire_object_and_does_not_accept_fallback(tokenizer):
    from lmformatenforcer.characterlevelparser import ForceStopParser

    schema = report_schema_for_options({"A": "one", "B": "two"})
    partial = build_report_constraint(tokenizer, schema)
    ids = walk_prefix(tokenizer, partial, '{"answer_choice":"A"')
    assert not partial.is_complete(ids)
    assert tokenizer.eos_token_id not in partial(0, ids)

    constraint = build_report_constraint(tokenizer, schema)
    ids = walk_prefix(tokenizer, constraint, json.dumps(_compact_report(), separators=(",", ":")))
    assert constraint.is_complete(ids)
    assert tokenizer.eos_token_id in constraint(0, ids)
    # This is LMFE's internal-error fallback. EOS alone must not masquerade as
    # a successfully closed report when the actual parser was discarded.
    constraint.token_enforcer.prefix_states[tuple(ids.tolist())].parser = ForceStopParser()
    assert not constraint.is_complete(ids)


def test_mcq_grammar_bounds_string_content_not_only_array_counts(tokenizer):
    schema = report_schema_for_options({"A": "one", "B": "two"})
    structure = build_report_constraint(tokenizer, schema)
    prefix = '{"answer_choice":"A","study":{},"findings":[{"structure":"' + "x" * 64
    ids = walk_prefix(tokenizer, structure, prefix)
    assert tokenizer.convert_tokens_to_ids('"') in structure(0, ids)
    assert tokenizer.convert_tokens_to_ids("x") not in structure(0, ids)

    observation = build_report_constraint(tokenizer, schema)
    prefix = '{"answer_choice":"A","study":{},"findings":[{"structure":"lung","observation":"' + "x" * 160
    ids = walk_prefix(tokenizer, observation, prefix)
    assert tokenizer.convert_tokens_to_ids('"') in observation(0, ids)
    assert tokenizer.convert_tokens_to_ids("x") not in observation(0, ids)


def test_mcq_measurement_property_is_excluded_instead_of_ineffective_zero_array_limit(tokenizer):
    schema = report_schema_for_options({"A": "one", "B": "two"})
    constraint = build_report_constraint(tokenizer, schema)
    ids = walk_prefix(tokenizer, constraint,
                      '{"answer_choice":"A","study":{},"findings":[{"structure":"lung","observation":"observed","')
    assert tokenizer.convert_tokens_to_ids("m") not in constraint(0, ids)
    data = _compact_report()
    data["findings"] = [{"structure": "lung", "observation": "observed", "evidence_ids": ["Q0"]}]
    report = validate_report_for_options(data, {"A": "one", "B": "two"})
    assert report.findings[0].measurements == []
    data["findings"][0]["measurements"] = []
    with pytest.raises(ValueError):
        validate_report_for_options(data, {"A": "one", "B": "two"})


@pytest.mark.parametrize("bad_field", ["observations", "findings", "impression", "limitations", "triage_range"])
def test_mcq_postvalidation_checks_the_same_bounded_contract(bad_field):
    data = _compact_report()
    finding = {"structure": "lung", "observation": "observed", "evidence_ids": ["Q0"]}
    if bad_field == "observations":
        data["findings"] = [{**finding, "observation": "x" * 161}]
    elif bad_field == "findings":
        data["findings"] = [finding] * 3
    elif bad_field == "impression":
        data["impression"] = "x" * 201
    elif bad_field == "limitations":
        data["limitations"] = ["one", "two", "three", "four"]
    else:
        data["triage_summary"] = {"abnormality_probability": 100, "threshold": 0.33, "band": "unvalidated"}
    with pytest.raises(ValueError):
        validate_report_for_options(data, {"A": "one", "B": "two"})


def test_ordinary_report_preserves_general_content_without_mcq_limits():
    report = complete_report()
    report["findings"] = [{"structure": "multiple structures", "observation": "x" * 400,
                           "evidence_ids": ["Q0"] * 7,
                           "measurements": [{"value": 1, "unit": "mm", "source": "Q0"}] * 4}] * 7
    report["recommendations"] = ["review"] * 5
    report["limitations"] = ["limitation"] * 5
    assert len(validate_report_for_options(report).findings) == 7
    assert "maxItems" not in report_schema_for_options()["properties"]["findings"]
