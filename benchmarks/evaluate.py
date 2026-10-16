"""GroundCheck evaluation: raw Gemma extraction vs Gemma + GroundCheck gate.

SYNTHETIC AND LIMITED. 18 author-written bank-SMS sources (36 field
judgements) with author-assigned ground truth, derived from the sources in
benchmarks/benchmark.py. The verifier was built against these same wording
styles, so the gate's numbers are optimistic and say nothing about real-world
accuracy. This is NOT the verifier-only 30/30 in benchmarks/benchmark.py,
which is not model accuracy.

Commands (from the repository root)
    py -m benchmarks.evaluate                  real Ollama run (gemma3:1b)
    py -m benchmarks.evaluate --replay FILE    recompute metrics offline

Four kinds of data are kept apart in every record:
    truth       dataset ground truth (never sent to the model)
    baseline    raw model output, parsed extraction, verifier output
    correction  ONE retry for flagged fields, built from verifier output only
    final       what the gate would release (derived, never invented)

Exit codes: 0 ok, 1 replay metrics do not match the file, 2 operational
failure (Ollama unreachable, bad file, invalid dataset). A failed run never
overwrites earlier results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Sequence

if __package__ in (None, ""):  # started as "py benchmarks/evaluate.py"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark import CASES, COMMA, CREDIT, DEBIT, RS_STYLE
from groundcheck import demo, verify
from groundcheck.verifier import parse_amount

FIELD_NAMES = ("transaction_amount", "available_balance")
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "gemma_eval.json"
SCHEMA = 1
KIND = "gemma_eval_synthetic"
MIN_RELIABLE_N = 10  # rule of thumb for the notes, not a statistical test
LABEL = (
    "SYNTHETIC AND LIMITED: 18 author-written bank-SMS sources (36 field "
    "judgements) with author-assigned ground truth. The verifier was built "
    "against these same wording styles, so gate results are optimistic and say "
    "nothing about real-world accuracy. Not comparable with the verifier-only "
    "30/30 in benchmarks/benchmark.py, which is not model accuracy."
)


class DatasetError(ValueError):
    """The evaluation dataset no longer matches the benchmark sources."""


class ResultsFileError(ValueError):
    """A results file is unreadable or malformed."""


class EvaluationAborted(RuntimeError):
    """The model could not be reached; nothing is saved."""


# --- Dataset: ground truth lives ONLY here ------------------------------------

@dataclass(frozen=True)
class EvalSms:
    id: str
    source: str
    transaction_amount: str | None  # ground truth, Decimal-compatible text, or None
    available_balance: str | None
    note: str

    def truth(self) -> dict:
        return {"transaction_amount": self.transaction_amount,
                "available_balance": self.available_balance}


# None means "this field is NOT in the message": a model that gives a number
# there is inventing one. Truths are the author's judgement.
EVAL_SMS: tuple[EvalSms, ...] = (
    EvalSms("s01", DEBIT, "1250", "8400", "plain debit and balance"),
    EvalSms("s02", CREDIT, "2000", "9500", "credit wording"),
    EvalSms("s03", COMMA, "1250", "8400", "comma-separated clauses"),
    EvalSms("s04", "You paid ₹499 to ACME STORE. Balance: ₹1,501.", "499", "1501", "'paid' wording"),
    EvalSms("s05", RS_STYLE, "1250.00", "8400.50", "Rs. marker with decimals"),
    EvalSms("s06", "Your account was debited ₹1,250.50. Your available balance is ₹8,400.",
            "1250.50", "8400", "decimal transaction amount"),
    EvalSms("s07", "Your account was debited. Please check the app for details.",
            None, None, "no amounts at all: any number is invented"),
    EvalSms("s08", "Debited ₹500. Balance ₹500.", "500", "500", "same value in both roles"),
    EvalSms("s09", "₹8,400 is your balance after ₹1,250 was debited.", "1250", "8400",
            "balance mentioned first"),
    EvalSms("s10", "Your transaction limit is ₹5,000.", None, None,
            "a limit is neither a transaction nor a balance"),
    EvalSms("s11", "Debited ₹500. Limit ₹5,000.", "500", None, "a limit is not the balance"),
    EvalSms("s12", "Debited ₹1,250 and balance is ₹8,400.", "1250", "8400",
            "no punctuation between clauses"),
    EvalSms("s13", "Reference amount ₹8,400.", None, None, "an unlabelled reference amount"),
    EvalSms("s14", "Your balance after debit of ₹1,250 is ₹8,400.", "1250", "8400",
            "'balance after debit of' phrasing"),
    EvalSms("s15", "Your account was debited ₹1,25,000. Your available balance is ₹8,40,000.",
            "125000", "840000", "Indian lakh grouping"),
    EvalSms("s16", "INR 3200 was withdrawn\nBalance INR 10,000", "3200", "10000",
            "INR marker, newline-separated"),
    EvalSms("s17", "Debited Rs.1250.", "1250", None, "Rs. with no space, no balance given"),
    EvalSms("s18", "Debited ₹1,250; avl bal ₹8,400!", "1250", "8400", "semicolon and exclamation"),
)

# Benchmark sources with NO defensible single ground truth. Excluded on purpose.
EXCLUDED: tuple[tuple[str, str], ...] = (
    ("Your account was debited ₹1,2500. Your available balance is ₹8,400.",
     "Malformed number '1,2500' (invalid grouping): the transaction amount is "
     "undefined (1,250? 12,500?) and the verifier deliberately ignores it."),
    ("Debited ₹500. Credited ₹700.",
     "Two different transactions: there is no single transaction amount, and no balance."),
    ("Your balance was debited ₹1,250.",
     "Contradictory wording: it cannot be said whether 1,250 is a transaction or a balance."),
    ("₹500 and ₹700 were debited.",
     "Two debited amounts in one sentence: there is no single transaction amount."),
)


def check_dataset() -> None:
    """Fail loudly unless the dataset covers every benchmark source exactly once."""
    bench = {case.source for case in CASES}
    included = [item.source for item in EVAL_SMS]
    excluded = [source for source, _ in EXCLUDED]
    problems = []
    for source, count in Counter(included + excluded).items():
        if count > 1:
            problems.append(f"source listed more than once: {source!r}")
    for source in sorted(bench - set(included) - set(excluded)):
        problems.append(f"benchmark source not covered (add truth or exclude it): {source!r}")
    for source in sorted((set(included) | set(excluded)) - bench):
        problems.append(f"source is not in benchmarks/benchmark.py: {source!r}")
    ids = [item.id for item in EVAL_SMS]
    if len(set(ids)) != len(ids):
        problems.append("duplicate SMS ids")
    if any(not reason.strip() for _, reason in EXCLUDED):
        problems.append("every exclusion needs a reason")
    for item in EVAL_SMS:
        for name in FIELD_NAMES:
            value = item.truth()[name]
            try:
                if value is not None:
                    Decimal(value)
            except ArithmeticError:
                problems.append(f"{item.id}: truth for {name} is not a number: {value!r}")
    if tuple(demo.FIELDS) != FIELD_NAMES:
        problems.append("demo.FIELDS no longer matches the evaluated fields")
    if problems:
        raise DatasetError("; ".join(problems))


# --- Values and verification ----------------------------------------------------

def to_decimal(value: object) -> Decimal | None:
    """A model value as a Decimal, or None if it is not a readable finite amount."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        number = parse_amount(value)
    except (ValueError, ArithmeticError):
        return None
    return number if number.is_finite() else None


def _clean_value(value: object) -> object:
    """Keep JSON-safe scalars; anything else is stored as text (and so reads as invalid)."""
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    if isinstance(value, str):
        return value[:200]
    if isinstance(value, float) and math.isfinite(value):
        return value
    return repr(value)[:100]


def parse_model_output(raw: str) -> tuple[dict | None, str | None]:
    """Return (extraction, None), or (None, reason) if the output is unusable."""
    try:
        data = demo.parse_extraction(raw)
    except (demo.DemoError, TypeError) as exc:
        return None, (str(exc).splitlines() or ["unusable output"])[0][:200]
    return {name: _clean_value(data.get(name)) for name in FIELD_NAMES}, None


def verify_field(source: str, name: str, extraction: dict | None) -> dict:
    """Verifier output for one field of one extraction."""
    if extraction is None:
        return {"status": "invalid", "verdict": None,
                "reason": "The answer was not a usable JSON object.", "evidence": []}
    value = extraction.get(name)
    if value is None:
        return {"status": "not_extracted", "verdict": None,
                "reason": "No value was given.", "evidence": []}
    if to_decimal(value) is None:
        return {"status": "invalid", "verdict": None,
                "reason": "The value is not a readable amount.", "evidence": []}
    result = verify(source, value, demo.FIELDS[name])
    return {"status": "checked", "verdict": result.verdict.value,
            "reason": result.reason, "evidence": list(result.evidence)}


# --- Controlled correction: feedback is built from verifier output ONLY -----------

@dataclass(frozen=True)
class FlaggedField:
    field: str
    claimed: str  # what the model said, as text
    verdict: str  # UNSUPPORTED, AMBIGUOUS or UNVERIFIABLE
    reason: str
    evidence: tuple[str, ...]


FEEDBACK_HEADER = ("An automatic checker compared your previous answer with the SMS text. "
                   "These fields were not accepted:\n")
FEEDBACK_LINE = ("- {field}: you answered {claimed}. Checker verdict: {verdict}. "
                 "Reason: {reason} Evidence from the SMS: {evidence}\n")
FEEDBACK_FOOTER = ("Answer again with JSON only, in exactly the same shape. Use only amounts "
                   "that appear in the SMS, and use null for a field that is not in the message.")


def build_feedback(flagged: Sequence[FlaggedField]) -> str:
    """The follow-up text. It takes no ground truth, so it cannot contain any."""
    lines = []
    for f in flagged:
        evidence = "; ".join(f'"{e}"' for e in f.evidence) or "none found"
        lines.append(FEEDBACK_LINE.format(field=f.field, claimed=f.claimed, verdict=f.verdict,
                                          reason=f.reason, evidence=evidence))
    return FEEDBACK_HEADER + "".join(lines) + FEEDBACK_FOOTER


AskFn = Callable[..., str]


def evaluate_sms(item: EvalSms, ask: AskFn | None = None) -> dict:
    """Run one SMS: baseline extraction, verification, at most one correction."""
    ask = ask or demo.ask_model
    raw = ask(item.source)  # exactly what the UI sends
    extraction, parse_error = parse_model_output(raw)
    verifier = {name: verify_field(item.source, name, extraction) for name in FIELD_NAMES}
    record = {
        "id": item.id,
        "source": item.source,
        "truth": item.truth(),
        "baseline": {"raw_output": raw[:4000], "parse_error": parse_error,
                     "extraction": extraction, "verifier": verifier},
        "correction": None,
    }

    flagged = [name for name in FIELD_NAMES
               if verifier[name]["status"] != "not_extracted" and verifier[name]["verdict"] != "SUPPORTED"]
    if flagged:
        items = []
        for name in flagged:
            v = verifier[name]
            claimed = "no usable value" if extraction is None else json.dumps(extraction.get(name))
            items.append(FlaggedField(name, claimed, v["verdict"] or "UNVERIFIABLE",
                                      v["reason"], tuple(v["evidence"])))
        feedback = build_feedback(items)
        follow_up = [{"role": "assistant", "content": raw},
                     {"role": "user", "content": feedback}]
        raw2 = ask(item.source, follow_up=follow_up)  # the ONLY retry
        extraction2, parse_error2 = parse_model_output(raw2)
        record["correction"] = {
            "attempted_fields": flagged,
            "feedback": feedback,
            "raw_output": raw2[:4000],
            "parse_error": parse_error2,
            "extraction": extraction2,
            "verifier": {name: verify_field(item.source, name, extraction2) for name in flagged},
        }
    record["final"] = resolve_final(record)
    return record


def resolve_final(record: dict) -> dict:
    """What would be released. A corrected value is released only if it verified."""
    correction = record.get("correction")
    base_extraction = record["baseline"]["extraction"] or {}
    final = {}
    for name in FIELD_NAMES:
        if record["baseline"]["verifier"][name]["verdict"] == "SUPPORTED":
            final[name] = {"value": base_extraction.get(name), "origin": "baseline"}
        elif (correction and name in correction["attempted_fields"]
              and correction["verifier"][name]["verdict"] == "SUPPORTED"):
            final[name] = {"value": (correction["extraction"] or {}).get(name),
                           "origin": "correction"}
        else:
            final[name] = {"value": None, "origin": "withheld"}
    return final


def evaluate_all(items: Sequence[EvalSms], ask: AskFn | None = None,
                 progress: Callable[[int, int, EvalSms], None] | None = None) -> list[dict]:
    records = []
    for index, item in enumerate(items, start=1):
        if progress:
            progress(index, len(items), item)
        try:
            records.append(evaluate_sms(item, ask))
        except demo.DemoError as exc:  # only model/connection errors reach here
            raise EvaluationAborted(f"Stopped at {item.id}: {exc}") from exc
    return records


# --- Judging and metrics ---------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for k successes in n trials (None if n == 0)."""
    if n <= 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rate(k: int, n: int) -> dict:
    """A rate with its numerator and denominator. value is None when n == 0 ("n/a")."""
    return {"num": k, "den": n, "value": (k / n) if n else None, "ci95": wilson(k, n)}


def _field_state(extraction: dict | None, name: str) -> tuple[object, str]:
    if extraction is None:
        return None, "invalid"
    value = extraction.get(name)
    if value is None:
        return None, "null"
    return value, ("number" if to_decimal(value) is not None else "invalid")


def _is_correct(truth: Decimal | None, value: object, kind: str) -> bool:
    if truth is None:
        return kind == "null"  # nothing should be there
    return kind == "number" and to_decimal(value) == truth


def retry_outcome(base_correct: bool, kind: str, correct: bool, released: bool) -> str:
    """Classify a retried field. base_correct: the baseline value was right."""
    if base_correct:
        if correct:
            return "recovered" if released else "still_withheld_correct"
        return "regression_released" if released else "regression_withheld"
    if correct and (released or kind == "null"):
        return "corrected"  # right value verified, or right abstention (truth is null)
    if released:
        return "released_wrong"
    if correct:
        return "correct_but_withheld"
    if kind == "null":
        return "abstained_missing"
    return "still_wrong_withheld"


def field_judgments(record: dict) -> list[dict]:
    """One judgement per field, comparing truth, model value, verdict and retry."""
    correction = record.get("correction")
    out = []
    for name in FIELD_NAMES:
        raw_truth = record["truth"][name]
        truth = None if raw_truth is None else Decimal(raw_truth)
        value, kind = _field_state(record["baseline"]["extraction"], name)
        verdict = record["baseline"]["verifier"][name]["verdict"]
        judgement = {
            "sms_id": record["id"], "field": name, "truth": truth, "value": value,
            "kind": kind, "present": kind != "null", "verdict": verdict,
            "released": verdict == "SUPPORTED",
            "correct": _is_correct(truth, value, kind), "retry": None,
        }
        if (judgement["present"] and not judgement["released"]
                and correction and name in correction["attempted_fields"]):
            c_value, c_kind = _field_state(correction["extraction"], name)
            c_verdict = correction["verifier"][name]["verdict"]
            c_correct = _is_correct(truth, c_value, c_kind)
            c_released = c_verdict == "SUPPORTED"
            judgement["retry"] = {
                "value": c_value, "kind": c_kind, "verdict": c_verdict,
                "released": c_released, "correct": c_correct,
                "outcome": retry_outcome(judgement["correct"], c_kind, c_correct, c_released),
            }
        out.append(judgement)
    return out


def _detection(present: list[dict], flag: Callable[[dict], bool]) -> dict:
    """Positive = the model's value is wrong. Flagged = the gate objected."""
    tp = sum(1 for j in present if not j["correct"] and flag(j))
    fp = sum(1 for j in present if j["correct"] and flag(j))
    fn = sum(1 for j in present if not j["correct"] and not flag(j))
    tn = sum(1 for j in present if j["correct"] and not flag(j))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": rate(tp, tp + fp), "recall": rate(tp, tp + fn)}


def compute_metrics(records: Sequence[dict]) -> dict:
    """All metrics, computed only from the records. Deterministic."""
    judged = [j for record in records for j in field_judgments(record)]
    present = [j for j in judged if j["present"]]
    right = [j for j in present if j["correct"]]
    wrong = [j for j in present if not j["correct"]]
    accepted = [j for j in present if j["released"]]
    accepted_right = [j for j in accepted if j["correct"]]
    numeric_truth = [j for j in judged if j["truth"] is not None]
    retried = [j for j in judged if j["retry"]]
    outcomes = Counter(j["retry"]["outcome"] for j in retried)
    wrong_flagged = [j for j in retried if not j["correct"]]
    right_flagged = [j for j in retried if j["correct"]]
    regressions = outcomes["regression_released"] + outcomes["regression_withheld"]

    pipeline_released = pipeline_right = 0
    for j in present:
        if j["released"]:
            pipeline_released += 1
            pipeline_right += j["correct"]
        elif j["retry"] and j["retry"]["released"]:
            pipeline_released += 1
            pipeline_right += j["retry"]["correct"]

    metrics = {
        "dataset": {
            "sms": len(records), "fields": len(judged),
            "fields_with_numeric_truth": len(numeric_truth),
            "fields_with_null_truth": len(judged) - len(numeric_truth),
        },
        "model_output": {
            "unusable_baseline_outputs": sum(1 for r in records if r["baseline"]["parse_error"]),
        },
        "baseline": {
            "accuracy": rate(sum(j["correct"] for j in judged), len(judged)),
            "released_precision": rate(len(right), len(present)),
            "values_given": len(present),
            "abstentions": {
                "correct": sum(1 for j in judged if not j["present"] and j["correct"]),
                "missed_value": sum(1 for j in judged if not j["present"] and not j["correct"]),
            },
            "wrong_values": {
                "wrong_number": sum(1 for j in wrong if j["kind"] == "number" and j["truth"] is not None),
                "invented_number": sum(1 for j in wrong if j["kind"] == "number" and j["truth"] is None),
                "unreadable": sum(1 for j in wrong if j["kind"] == "invalid"),
            },
        },
        "groundcheck": {
            "accepted": len(accepted),
            "accepted_precision": rate(len(accepted_right), len(accepted)),
            "coverage": rate(len(accepted), len(present)),
            "true_values_released": rate(len(accepted_right), len(numeric_truth)),
            "false_acceptances": rate(sum(1 for j in wrong if j["released"]), len(wrong)),
            "false_rejections": {
                **rate(sum(1 for j in right if not j["released"]), len(right)),
                "rejected_as_unsupported": sum(1 for j in right if j["verdict"] == "UNSUPPORTED"),
                "rejected_as_ambiguous": sum(1 for j in right if j["verdict"] == "AMBIGUOUS"),
            },
            "detection_strict": _detection(present, lambda j: j["verdict"] == "UNSUPPORTED"),
            "detection_conservative": _detection(present, lambda j: not j["released"]),
        },
        "correction": {
            "retried_sms": sum(1 for r in records if r.get("correction")),
            "retried_fields": len(retried),
            "wrong_flagged_fields": len(wrong_flagged),
            "success": rate(outcomes["corrected"], len(wrong_flagged)),
            "outcomes_for_wrong_values": {
                key: outcomes[key] for key in ("corrected", "released_wrong", "correct_but_withheld",
                                               "abstained_missing", "still_wrong_withheld")},
            "outcomes_for_correct_but_rejected_values": {
                key: outcomes[key] for key in ("recovered", "still_withheld_correct")},
            "regressions": {
                **rate(regressions, len(right_flagged)),
                "released_wrong": outcomes["regression_released"],
                "withheld": outcomes["regression_withheld"],
            },
        },
        "pipeline": {
            "released": pipeline_released,
            "precision": rate(pipeline_right, pipeline_released),
            "coverage": rate(pipeline_released, len(present)),
            "released_wrong": pipeline_released - pipeline_right,
        },
    }
    metrics["reliability_notes"] = reliability_notes(metrics)
    return metrics


def reliability_notes(metrics: dict) -> list[str]:
    notes: list[str] = []

    def walk(node: object, path: str) -> None:
        if not isinstance(node, dict):
            return
        if {"num", "den", "ci95"} <= node.keys():
            if node["den"] == 0:
                notes.append(f"{path}: n/a, the denominator is 0 so there is nothing to measure")
            elif node["den"] < MIN_RELIABLE_N:
                notes.append(f"{path}: only {node['den']} cases, so the 95% interval is wide; "
                             "treat this estimate as anecdotal")
        for key, value in node.items():
            walk(value, f"{path}.{key}" if path else key)

    walk(metrics, "")
    return notes


# --- Saving, loading, replay -------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_ollama_version() -> str | None:
    """Best-effort Ollama version; None if it cannot be read (never invented)."""
    url = demo.OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/version"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return str(json.loads(response.read().decode("utf-8"))["version"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build_document(records: list[dict], ollama_version: str | None,
                   created: str | None = None) -> dict:
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "label": LABEL,
        "created_utc": created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": demo.MODEL,
        "ollama_version": ollama_version,
        "python": platform.python_version(),
        "hashes": {
            "dataset": _sha256(json.dumps(
                [[i.id, i.source, i.transaction_amount, i.available_balance] for i in EVAL_SMS],
                ensure_ascii=False)),
            "system_prompt": _sha256(demo.SYSTEM_PROMPT),
            "feedback_template": _sha256(FEEDBACK_HEADER + FEEDBACK_LINE + FEEDBACK_FOOTER),
        },
        "excluded_sources": [{"source": s, "reason": r} for s, r in EXCLUDED],
        "records": records,
        "metrics": compute_metrics(records),
    }


def write_atomic(document: dict, path: Path) -> None:
    """Write the file completely or not at all; an old file survives any failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_results(path: Path) -> dict:
    """Load and structurally validate a results file."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResultsFileError(f"Cannot read results file {path}: {exc}") from exc

    def need(condition: bool, what: str) -> None:
        if not condition:
            raise ResultsFileError(f"Malformed results file: {what}")

    need(isinstance(document, dict), "top level is not an object")
    need(document.get("kind") == KIND and document.get("schema") == SCHEMA,
         f"expected kind={KIND!r}, schema={SCHEMA}")
    records = document.get("records")
    need(isinstance(records, list) and records, "'records' must be a non-empty list")
    for record in records:
        need(isinstance(record, dict) and {"id", "source", "truth", "baseline"} <= record.keys(),
             "a record lacks id/source/truth/baseline")
        need(isinstance(record["truth"], dict) and set(FIELD_NAMES) <= record["truth"].keys(),
             f"record {record['id']}: bad truth")
        base = record["baseline"]
        need(isinstance(base, dict) and isinstance(base.get("verifier"), dict)
             and set(FIELD_NAMES) <= base["verifier"].keys()
             and (base.get("extraction") is None or isinstance(base["extraction"], dict)),
             f"record {record['id']}: bad baseline")
        correction = record.get("correction")
        need(correction is None or (isinstance(correction, dict)
                                    and isinstance(correction.get("attempted_fields"), list)
                                    and isinstance(correction.get("verifier"), dict)
                                    and (correction.get("extraction") is None
                                         or isinstance(correction["extraction"], dict))),
             f"record {record['id']}: bad correction")
    return document


def verifier_drift(records: Sequence[dict]) -> list[str]:
    """Re-run today's verifier on stored model outputs; list any changed verdicts."""
    drift = []
    for record in records:
        parts = [("baseline", record["baseline"], list(FIELD_NAMES))]
        if record.get("correction"):
            parts.append(("correction", record["correction"], record["correction"]["attempted_fields"]))
        for label, part, names in parts:
            for name in names:
                stored = part["verifier"][name]
                now = verify_field(record["source"], name, part["extraction"])
                if (now["status"], now["verdict"]) != (stored["status"], stored["verdict"]):
                    drift.append(f"{record['id']} {label} {name}: stored {stored['verdict']}, now {now['verdict']}")
    return drift


# --- Report --------------------------------------------------------------------------

def fmt_rate(r: dict) -> str:
    if r["den"] == 0:
        return "n/a (0/0)"
    low, high = r["ci95"]
    return (f"{r['num']}/{r['den']} ({100 * r['value']:.1f}%)  "
            f"95% CI {100 * low:.1f}-{100 * high:.1f}%")


def format_report(document: dict) -> str:
    m = document["metrics"]
    ds, base, gc = m["dataset"], m["baseline"], m["groundcheck"]
    cor, pipe = m["correction"], m["pipeline"]
    lines = [
        "GroundCheck evaluation: raw Gemma vs Gemma + GroundCheck gate",
        LABEL,
        f"model={document['model']}  ollama={document['ollama_version'] or 'unknown'}  "
        f"created={document['created_utc']}  python={document['python']}",
        "", "PER-FIELD RESULTS (truth is hidden from the model)",
    ]
    for record in document["records"]:
        for j in field_judgments(record):
            truth = "null" if j["truth"] is None else str(j["truth"])
            value = json.dumps(j["value"]) if j["present"] else "null"
            verdict = j["verdict"] or ("unreadable" if j["kind"] == "invalid" else "-")
            retry = ""
            if j["retry"]:
                retry = f"  retry: {json.dumps(j['retry']['value'])} {j['retry']['verdict'] or '-'} -> {j['retry']['outcome']}"
            lines.append(f"  {j['sms_id']} {j['field']:<18} truth={truth:<8} "
                         f"model={value[:12]:<12} {verdict:<11}{retry}")
    lines += [
        "", f"DATASET: {ds['sms']} SMS, {ds['fields']} fields "
            f"({ds['fields_with_numeric_truth']} with a numeric truth, {ds['fields_with_null_truth']} where nothing should be extracted)",
        f"  excluded sources: {len(document['excluded_sources'])} (no defensible ground truth)",
        "", "BASELINE (every model value released as given)",
        f"  field accuracy            {fmt_rate(base['accuracy'])}",
        f"  precision of given values {fmt_rate(base['released_precision'])}",
        f"  abstentions: {base['abstentions']['correct']} correct, {base['abstentions']['missed_value']} missed a value",
        f"  wrong values: {base['wrong_values']}",
        "", "GROUNDCHECK GATE (releases a value only when the verifier says SUPPORTED)",
        f"  precision of accepted     {fmt_rate(gc['accepted_precision'])}",
        f"  coverage of given values  {fmt_rate(gc['coverage'])}",
        f"  true values released      {fmt_rate(gc['true_values_released'])}",
        f"  false acceptances (wrong value accepted)   {fmt_rate(gc['false_acceptances'])}",
        f"  false rejections (correct value withheld)  {fmt_rate(gc['false_rejections'])}"
        f"  [unsupported {gc['false_rejections']['rejected_as_unsupported']}, ambiguous {gc['false_rejections']['rejected_as_ambiguous']}]",
    ]
    for key, title in (("detection_strict", "flag = UNSUPPORTED only"),
                       ("detection_conservative", "flag = anything not SUPPORTED")):
        d = gc[key]
        lines += [f"  wrong-value detection, {title}:",
                  f"    TP={d['tp']} FP={d['fp']} FN={d['fn']} TN={d['tn']}",
                  f"    precision {fmt_rate(d['precision'])}",
                  f"    recall    {fmt_rate(d['recall'])}"]
    lines += [
        "", "CORRECTION (one retry per SMS with flagged fields; feedback built from verifier output only)",
        f"  retried: {cor['retried_sms']} SMS, {cor['retried_fields']} fields",
        f"  success on wrong values   {fmt_rate(cor['success'])}",
        f"  outcomes for wrong values: {cor['outcomes_for_wrong_values']}",
        f"  outcomes for correct-but-rejected values: {cor['outcomes_for_correct_but_rejected_values']}",
        f"  regressions (retry broke a correct value) {fmt_rate(cor['regressions'])}"
        f"  [released wrong {cor['regressions']['released_wrong']}, withheld {cor['regressions']['withheld']}]",
        "", "PIPELINE (baseline accepted + verified corrections)",
        f"  precision of released     {fmt_rate(pipe['precision'])}",
        f"  coverage of given values  {fmt_rate(pipe['coverage'])}",
        f"  wrong values released     {pipe['released_wrong']}",
    ]
    if m["reliability_notes"]:
        lines += ["", "SMALL-SAMPLE WARNINGS (estimates below are not reliable)"]
        lines += [f"  - {note}" for note in m["reliability_notes"]]
    return "\n".join(lines)


# --- Command line ---------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="py -m benchmarks.evaluate",
        description="Evaluate raw Gemma extraction vs Gemma + GroundCheck on synthetic SMS.")
    parser.add_argument("--replay", metavar="FILE",
                        help="recompute metrics from a saved results file (offline, no Ollama)")
    parser.add_argument("--out", metavar="FILE", default=str(RESULTS_PATH),
                        help="where a real run saves its results")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    try:
        if args.replay:
            document = load_results(Path(args.replay))
            try:
                recomputed = json.loads(json.dumps(compute_metrics(document["records"])))
                drift = verifier_drift(document["records"])
                report = format_report({**document, "metrics": recomputed})
            except (KeyError, TypeError, ValueError, ArithmeticError, AttributeError) as exc:
                raise ResultsFileError(f"Malformed results file: {exc!r}") from exc
            print(report)
            for line in drift:
                print(f"WARNING verifier changed since this file was recorded: {line}")
            matches = recomputed == document.get("metrics")
            print("\nReplay: metrics recomputed from the stored records "
                  + ("MATCH the file." if matches else "DO NOT MATCH the file's stored metrics."))
            return 0 if matches else 1

        check_dataset()
        print(f"Evaluating {len(EVAL_SMS)} synthetic SMS with {demo.MODEL} through Ollama...")
        records = evaluate_all(
            EVAL_SMS, progress=lambda i, n, item: print(f"  [{i:2d}/{n}] {item.id}", flush=True))
        document = build_document(records, get_ollama_version())
        write_atomic(document, Path(args.out))
        print(format_report(document))
        print(f"\nSaved: {args.out}")
        return 0
    except (DatasetError, ResultsFileError, EvaluationAborted) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
