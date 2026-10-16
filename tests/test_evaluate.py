"""Tests for benchmarks/evaluate.py and the ask_model follow_up argument.

Nothing here needs Ollama or the network: urllib is blocked for every test,
and the "model" is a scripted stand-in. The scripted replies and hand-written
truths below are TEST FIXTURES for checking the maths. They are not benchmark
results and are never written to benchmarks/results/.
"""

import dataclasses
import inspect
import json
import urllib.request

import pytest

from benchmarks import evaluate as ev
from benchmarks.benchmark import CASES, DEBIT, Case
from groundcheck import Role, Verdict, demo


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted in a unit test")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)


class ScriptedModel:
    """Stands in for demo.ask_model: returns queued replies, records every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, sms, follow_up=None):
        self.calls.append({"sms": sms, "follow_up": follow_up})
        return self.replies.pop(0)


def reply(tx, bal):
    return json.dumps({"transaction_amount": tx, "available_balance": bal})


def item(id, source, tx, bal):
    return ev.EvalSms(id, source, tx, bal, "fixture")


NO_AMOUNTS = "Your account was debited. Please check the app for details."
AMBIGUOUS_WORDING = "Debited ₹1,250 and balance is ₹8,400."

# Six fixture SMS whose expected counts were worked out by hand (see the test).
SCENARIO_ITEMS = [
    item("t1", DEBIT, "1250", "8400"),
    item("t2", DEBIT, "1250", "8400"),
    item("t3", NO_AMOUNTS, None, None),
    item("t4", AMBIGUOUS_WORDING, "1250", "8400"),
    item("t5", DEBIT, "1300", "8400"),  # deliberately inconsistent truth: a false acceptance
    item("t6", DEBIT, "1250", "8400"),
]
SCENARIO_REPLIES = [
    reply(1250, "8,400"),          # t1 baseline: both right, both SUPPORTED
    reply(8400, 1250),             # t2 baseline: swapped, both UNSUPPORTED
    reply(1250, 8400),             # t2 retry: both corrected
    reply(1250, None),             # t3 baseline: invents a transaction amount
    reply(None, None),             # t3 retry: correct abstention
    reply(1250, 8400),             # t4 baseline: right values, but AMBIGUOUS wording
    reply(999, 8400),              # t4 retry: breaks the transaction amount
    reply(1250, 8400),             # t5 baseline: SUPPORTED but wrong against this truth
    "not json at all",             # t6 baseline: unusable output
    reply(1250, 8400),             # t6 retry: corrected
]


@pytest.fixture
def scenario():
    model = ScriptedModel(SCENARIO_REPLIES)
    records = ev.evaluate_all(SCENARIO_ITEMS, model)
    return records, model


def counts(rate_dict):
    return rate_dict["num"], rate_dict["den"]


# --- Dataset coverage ---------------------------------------------------------

def test_dataset_covers_every_benchmark_source_exactly_once():
    ev.check_dataset()
    covered = [i.source for i in ev.EVAL_SMS] + [s for s, _ in ev.EXCLUDED]
    assert sorted(covered) == sorted({c.source for c in CASES})


def test_dataset_has_18_sources_and_4_documented_exclusions():
    assert len(ev.EVAL_SMS) == 18 and len(ev.EXCLUDED) == 4
    assert all(len(reason) > 20 for _, reason in ev.EXCLUDED)


def test_dropping_a_source_is_detected(monkeypatch):
    monkeypatch.setattr(ev, "EVAL_SMS", ev.EVAL_SMS[:-1])
    with pytest.raises(ev.DatasetError, match="not covered"):
        ev.check_dataset()


def test_unknown_source_is_detected(monkeypatch):
    extra = item("s99", "Some SMS that is not in the benchmark.", "1", "2")
    monkeypatch.setattr(ev, "EVAL_SMS", ev.EVAL_SMS + (extra,))
    with pytest.raises(ev.DatasetError, match="not in benchmarks"):
        ev.check_dataset()


def test_duplicate_source_is_detected(monkeypatch):
    dup = item("s98", ev.EVAL_SMS[0].source, "1", "2")
    monkeypatch.setattr(ev, "EVAL_SMS", ev.EVAL_SMS + (dup,))
    with pytest.raises(ev.DatasetError, match="more than once"):
        ev.check_dataset()


def test_new_benchmark_case_with_a_new_source_is_detected(monkeypatch):
    new_case = Case("brand_new", "correct", "Brand new SMS ₹10.", "10", Role.TRANSACTION,
                    Verdict.AMBIGUOUS, "fixture")
    monkeypatch.setattr(ev, "CASES", CASES + (new_case,))
    with pytest.raises(ev.DatasetError, match="not covered"):
        ev.check_dataset()


# --- Intervals and rates --------------------------------------------------------

def test_wilson_matches_textbook_values():
    low, high = ev.wilson(5, 10)
    assert (round(low, 3), round(high, 3)) == (0.237, 0.763)
    low, high = ev.wilson(0, 10)
    assert low == 0.0 and round(high, 3) == 0.278
    assert ev.wilson(10, 10)[1] == 1.0


def test_zero_denominator_is_not_available_never_zero():
    assert ev.wilson(0, 0) is None
    r = ev.rate(0, 0)
    assert r["value"] is None and r["ci95"] is None and (r["num"], r["den"]) == (0, 0)
    assert ev.fmt_rate(r) == "n/a (0/0)"
    assert ev.rate(3, 4)["value"] == 0.75


def test_metrics_with_nothing_to_measure_report_na():
    model = ScriptedModel([reply(None, None)])
    records = ev.evaluate_all([item("z1", NO_AMOUNTS, None, None)], model)
    metrics = ev.compute_metrics(records)
    gc = metrics["groundcheck"]
    for r in (gc["accepted_precision"], gc["coverage"], gc["false_acceptances"],
              gc["detection_strict"]["precision"], gc["detection_conservative"]["recall"],
              metrics["correction"]["success"], metrics["pipeline"]["precision"]):
        assert r["value"] is None and r["den"] == 0
    assert metrics["baseline"]["accuracy"]["value"] == 1.0  # both abstentions are right
    assert any("n/a" in note for note in metrics["reliability_notes"])
    document = ev.build_document(records, None)
    assert "n/a (0/0)" in ev.format_report(document)


# --- Metrics against hand-computed counts ---------------------------------------

def test_scenario_metrics_match_hand_computed_counts(scenario):
    records, model = scenario
    m = ev.compute_metrics(records)
    assert len(model.calls) == 10 and model.replies == []  # 6 baseline + 4 retries, nothing else

    assert m["dataset"] == {"sms": 6, "fields": 12, "fields_with_numeric_truth": 10,
                            "fields_with_null_truth": 2}
    assert m["model_output"]["unusable_baseline_outputs"] == 1

    base = m["baseline"]
    assert counts(base["accuracy"]) == (6, 12)
    assert counts(base["released_precision"]) == (5, 11) and base["values_given"] == 11
    assert base["abstentions"] == {"correct": 1, "missed_value": 0}
    assert base["wrong_values"] == {"wrong_number": 3, "invented_number": 1, "unreadable": 2}

    gc = m["groundcheck"]
    assert gc["accepted"] == 4
    assert counts(gc["accepted_precision"]) == (3, 4)
    assert counts(gc["coverage"]) == (4, 11)
    assert counts(gc["true_values_released"]) == (3, 10)
    assert counts(gc["false_acceptances"]) == (1, 6)
    fr = gc["false_rejections"]
    assert (fr["num"], fr["den"]) == (2, 5)
    assert (fr["rejected_as_unsupported"], fr["rejected_as_ambiguous"]) == (0, 2)

    strict = gc["detection_strict"]
    assert (strict["tp"], strict["fp"], strict["fn"], strict["tn"]) == (3, 0, 3, 5)
    assert counts(strict["precision"]) == (3, 3) and counts(strict["recall"]) == (3, 6)
    cons = gc["detection_conservative"]
    assert (cons["tp"], cons["fp"], cons["fn"], cons["tn"]) == (5, 2, 1, 3)
    assert counts(cons["precision"]) == (5, 7) and counts(cons["recall"]) == (5, 6)

    cor = m["correction"]
    assert (cor["retried_sms"], cor["retried_fields"], cor["wrong_flagged_fields"]) == (4, 7, 5)
    assert counts(cor["success"]) == (5, 5)
    assert cor["outcomes_for_wrong_values"]["corrected"] == 5
    assert sum(cor["outcomes_for_wrong_values"].values()) == 5
    assert cor["outcomes_for_correct_but_rejected_values"] == {"recovered": 0, "still_withheld_correct": 1}
    reg = cor["regressions"]
    assert (reg["num"], reg["den"], reg["released_wrong"], reg["withheld"]) == (1, 2, 0, 1)

    pipe = m["pipeline"]
    assert pipe["released"] == 8 and pipe["released_wrong"] == 1
    assert counts(pipe["precision"]) == (7, 8) and counts(pipe["coverage"]) == (8, 11)


def test_pipeline_metrics_agree_with_the_released_values_in_the_records(scenario):
    records, _ = scenario
    released = right = 0
    for record in records:
        for name, final in record["final"].items():
            if final["origin"] == "withheld":
                assert final["value"] is None
                continue
            released += 1
            truth = record["truth"][name]
            right += truth is not None and ev.to_decimal(final["value"]) == ev.Decimal(truth)
    pipe = ev.compute_metrics(records)["pipeline"]
    assert (right, released) == (pipe["precision"]["num"], pipe["released"])


def test_small_samples_are_flagged_as_unreliable(scenario):
    notes = ev.compute_metrics(scenario[0])["reliability_notes"]
    assert any("anecdotal" in note for note in notes)
    assert not any(note.startswith("baseline.accuracy:") for note in notes)  # n = 12, no warning


@pytest.mark.parametrize("base_correct, kind, correct, released, expected", [
    (False, "number", True, True, "corrected"),
    (False, "null", True, False, "corrected"),               # right abstention (truth is null)
    (False, "number", False, True, "released_wrong"),
    (False, "number", True, False, "correct_but_withheld"),
    (False, "null", False, False, "abstained_missing"),
    (False, "invalid", False, False, "still_wrong_withheld"),
    (True, "number", True, True, "recovered"),
    (True, "number", True, False, "still_withheld_correct"),
    (True, "number", False, True, "regression_released"),
    (True, "number", False, False, "regression_withheld"),
    (True, "null", False, False, "regression_withheld"),
])
def test_retry_outcome_classification(base_correct, kind, correct, released, expected):
    assert ev.retry_outcome(base_correct, kind, correct, released) == expected


# --- Malformed model output --------------------------------------------------------

@pytest.mark.parametrize("raw, statuses", [
    ("not json at all", ("invalid", "invalid")),
    ("[1250, 8400]", ("invalid", "invalid")),
    ("{}", ("not_extracted", "not_extracted")),
    ('{"transaction_amount": [1250], "available_balance": true}', ("invalid", "invalid")),
    ('{"transaction_amount": "a lot", "available_balance": NaN}', ("invalid", "invalid")),
    ('{"transaction_amount": 1250, "available_balance": 8400, "extra": 1}', ("checked", "checked")),
])
def test_malformed_model_output_is_handled_without_crashing(raw, statuses):
    model = ScriptedModel([raw, reply(None, None)])
    record = ev.evaluate_sms(item("m1", DEBIT, "1250", "8400"), model)
    verifier = record["baseline"]["verifier"]
    assert (verifier["transaction_amount"]["status"], verifier["available_balance"]["status"]) == statuses
    document = ev.build_document([record], None)  # must be strictly serialisable (no NaN)
    json.dumps(document, allow_nan=False)


def test_unusable_output_is_recorded_and_retried_once():
    model = ScriptedModel(["not json at all", reply(1250, 8400)])
    record = ev.evaluate_sms(item("m2", DEBIT, "1250", "8400"), model)
    assert record["baseline"]["extraction"] is None and record["baseline"]["parse_error"]
    assert record["correction"]["attempted_fields"] == list(ev.FIELD_NAMES)
    assert all(v["origin"] == "correction" for v in record["final"].values())
    assert len(model.calls) == 2


# --- Retry safety --------------------------------------------------------------------

def test_at_most_one_retry_even_if_the_retry_is_still_wrong():
    model = ScriptedModel([reply(8400, 1250), reply(8400, 1250)])
    record = ev.evaluate_sms(item("r1", DEBIT, "1250", "8400"), model)
    assert len(model.calls) == 2 and model.replies == []
    assert record["correction"] is not None


def test_a_value_that_fails_verification_is_never_released():
    model = ScriptedModel([reply(8400, 1250), reply(999, 777)])
    record = ev.evaluate_sms(item("r2", DEBIT, "1250", "8400"), model)
    assert record["final"] == {
        "transaction_amount": {"value": None, "origin": "withheld"},
        "available_balance": {"value": None, "origin": "withheld"},
    }


def test_accepted_fields_are_never_replaced_by_the_retry():
    # tx is accepted at baseline; only the balance is flagged and retried.
    model = ScriptedModel([reply(1250, 1250), reply(8400, 8400)])
    record = ev.evaluate_sms(item("r3", DEBIT, "1250", "8400"), model)
    assert record["correction"]["attempted_fields"] == ["available_balance"]
    assert record["final"]["transaction_amount"] == {"value": 1250, "origin": "baseline"}
    assert record["final"]["available_balance"] == {"value": 8400, "origin": "correction"}


def test_follow_up_is_the_previous_answer_then_the_feedback():
    model = ScriptedModel([reply(8400, 1250), reply(1250, 8400)])
    record = ev.evaluate_sms(item("r4", DEBIT, "1250", "8400"), model)
    first, second = model.calls
    assert first["follow_up"] is None  # the baseline call is exactly what the UI sends
    roles = [m["role"] for m in second["follow_up"]]
    assert roles == ["assistant", "user"]
    assert second["follow_up"][0]["content"] == record["baseline"]["raw_output"]
    assert second["follow_up"][1]["content"] == record["correction"]["feedback"]


def test_feedback_uses_verdict_reason_and_evidence():
    model = ScriptedModel([reply(8400, 1250), reply(1250, 8400)])
    record = ev.evaluate_sms(item("r5", DEBIT, "1250", "8400"), model)
    feedback = record["correction"]["feedback"]
    v = record["baseline"]["verifier"]["transaction_amount"]
    assert "UNSUPPORTED" in feedback and v["reason"] in feedback
    assert v["evidence"][0] in feedback


def test_ground_truth_never_reaches_the_model():
    canary_tx, canary_bal = "424242", "737373"  # appear nowhere in the SMS
    model = ScriptedModel([reply(8400, 1250), reply(8400, 1250)])
    ev.evaluate_sms(item("r6", DEBIT, canary_tx, canary_bal), model)
    sent = json.dumps(model.calls, ensure_ascii=False)
    assert canary_tx not in sent and canary_bal not in sent
    # and the feedback builder cannot even receive a truth value:
    assert list(inspect.signature(ev.build_feedback).parameters) == ["flagged"]
    assert {f.name for f in dataclasses.fields(ev.FlaggedField)} == {
        "field", "claimed", "verdict", "reason", "evidence"}


def test_a_connection_failure_aborts_the_run():
    def failing(sms, follow_up=None):
        raise demo.DemoError("Could not reach Ollama")

    with pytest.raises(ev.EvaluationAborted, match="s01"):
        ev.evaluate_all(ev.EVAL_SMS, failing)


# --- Saving, replay, determinism -------------------------------------------------------

@pytest.fixture
def saved(scenario, tmp_path):
    records, _ = scenario
    document = ev.build_document(records, "test-version", created="2000-01-01T00:00:00+00:00")
    path = tmp_path / "results.json"
    ev.write_atomic(document, path)
    return document, path


def test_replay_is_deterministic_and_matches_the_stored_metrics(saved, capsys):
    document, path = saved
    first = json.dumps(ev.compute_metrics(document["records"]), sort_keys=True)
    second = json.dumps(ev.compute_metrics(ev.load_results(path)["records"]), sort_keys=True)
    assert first == second
    assert ev.main(["--replay", str(path)]) == 0
    out_one = capsys.readouterr().out
    assert "MATCH the file" in out_one and "SYNTHETIC AND LIMITED" in out_one
    assert ev.main(["--replay", str(path)]) == 0
    assert capsys.readouterr().out == out_one


def test_replay_detects_tampered_metrics(saved, capsys):
    document, path = saved
    document["metrics"]["baseline"]["accuracy"]["num"] = 99
    path.write_text(json.dumps(document), encoding="utf-8")
    assert ev.main(["--replay", str(path)]) == 1
    assert "DO NOT MATCH" in capsys.readouterr().out


def test_replay_warns_when_the_verifier_no_longer_agrees(saved, capsys):
    document, path = saved
    document["records"][0]["baseline"]["verifier"]["transaction_amount"]["verdict"] = "UNSUPPORTED"
    path.write_text(json.dumps(document), encoding="utf-8")
    ev.main(["--replay", str(path)])
    assert "WARNING verifier changed" in capsys.readouterr().out


@pytest.mark.parametrize("content", [
    "this is not json",
    "[1, 2, 3]",
    json.dumps({"kind": "something_else", "schema": 1, "records": [{}]}),
    json.dumps({"kind": ev.KIND, "schema": ev.SCHEMA, "records": []}),
    json.dumps({"kind": ev.KIND, "schema": ev.SCHEMA, "records": [{"id": "x", "source": "s", "truth": {}}]}),
])
def test_replay_rejects_malformed_files_cleanly(tmp_path, capsys, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    assert ev.main(["--replay", str(path)]) == 2
    assert "Error" in capsys.readouterr().err


def test_replay_of_a_missing_file_is_a_clean_error(tmp_path, capsys):
    assert ev.main(["--replay", str(tmp_path / "nope.json")]) == 2
    assert "Cannot read" in capsys.readouterr().err


def test_atomic_write_keeps_the_old_file_if_the_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "results.json"
    path.write_text("PREVIOUS RESULTS", encoding="utf-8")

    def broken_replace(src, dst):
        raise OSError("disk exploded")

    monkeypatch.setattr(ev.os, "replace", broken_replace)
    with pytest.raises(OSError):
        ev.write_atomic({"a": 1}, path)
    assert path.read_text(encoding="utf-8") == "PREVIOUS RESULTS"
    assert [p.name for p in tmp_path.iterdir()] == ["results.json"]  # no temp file left


def test_atomic_write_rejects_nan_before_touching_the_old_file(tmp_path):
    path = tmp_path / "results.json"
    path.write_text("PREVIOUS RESULTS", encoding="utf-8")
    with pytest.raises(ValueError):
        ev.write_atomic({"bad": float("nan")}, path)
    assert path.read_text(encoding="utf-8") == "PREVIOUS RESULTS"


def test_a_failed_run_preserves_previous_results(tmp_path, monkeypatch, capsys):
    out = tmp_path / "gemma_eval.json"
    out.write_text("PREVIOUS RESULTS", encoding="utf-8")

    def unreachable(sms, follow_up=None):
        raise demo.DemoError("Could not reach Ollama at http://localhost:11434/api/chat")

    monkeypatch.setattr(demo, "ask_model", unreachable)
    assert ev.main(["--out", str(out)]) == 2
    assert out.read_text(encoding="utf-8") == "PREVIOUS RESULTS"
    assert "Could not reach Ollama" in capsys.readouterr().err


def test_a_successful_run_saves_a_labelled_valid_file(tmp_path, monkeypatch, capsys):
    truth_by_source = {i.source: i for i in ev.EVAL_SMS}

    def stand_in_model(sms, follow_up=None):  # a TEST stand-in that answers with the truths
        i = truth_by_source[sms]
        return reply(i.transaction_amount, i.available_balance)

    monkeypatch.setattr(demo, "ask_model", stand_in_model)
    monkeypatch.setattr(ev, "get_ollama_version", lambda: "test-version")
    out = tmp_path / "sub" / "gemma_eval.json"
    assert ev.main(["--out", str(out)]) == 0
    saved_doc = json.loads(out.read_text(encoding="utf-8"))
    assert saved_doc["kind"] == ev.KIND and "SYNTHETIC" in saved_doc["label"]
    assert saved_doc["model"] == demo.MODEL and saved_doc["ollama_version"] == "test-version"
    assert len(saved_doc["records"]) == 18 and len(saved_doc["excluded_sources"]) == 4
    assert saved_doc["metrics"]["dataset"]["fields"] == 36
    assert ev.load_results(out)["metrics"] == saved_doc["metrics"]
    assert "Saved:" in capsys.readouterr().out


def test_an_invalid_dataset_stops_the_run_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(ev, "EVAL_SMS", ev.EVAL_SMS[:-1])
    assert ev.main(["--out", "unused.json"]) == 2
    assert "not covered" in capsys.readouterr().err


# --- demo.ask_model stays backward compatible --------------------------------------------

class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def capture_request(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["data"] = request.data
        seen["url"] = request.full_url
        return FakeResponse({"message": {"content": "{}"}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def old_style_body(sms):
    """The request body exactly as ask_model built it before follow_up existed."""
    return json.dumps({
        "model": demo.MODEL,
        "messages": [{"role": "system", "content": demo.SYSTEM_PROMPT},
                     {"role": "user", "content": sms}],
        "stream": False, "format": "json", "options": {"temperature": 0},
    }).encode("utf-8")


def test_ask_model_default_request_is_byte_identical_to_before(monkeypatch):
    seen = capture_request(monkeypatch)
    assert demo.ask_model("hello ₹5") == "{}"
    assert seen["data"] == old_style_body("hello ₹5")
    assert seen["url"] == demo.OLLAMA_URL


def test_an_empty_follow_up_changes_nothing(monkeypatch):
    seen = capture_request(monkeypatch)
    demo.ask_model("hello", follow_up=[])
    assert seen["data"] == old_style_body("hello")


def test_follow_up_only_appends_messages(monkeypatch):
    seen = capture_request(monkeypatch)
    extra = [{"role": "assistant", "content": "A"}, {"role": "user", "content": "B"}]
    demo.ask_model("hello", follow_up=extra)
    body = json.loads(seen["data"])
    base = json.loads(old_style_body("hello"))
    assert body["messages"] == base["messages"] + extra
    assert {k: v for k, v in body.items() if k != "messages"} == {
        k: v for k, v in base.items() if k != "messages"}
