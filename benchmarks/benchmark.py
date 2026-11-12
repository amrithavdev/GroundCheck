"""GroundCheck synthetic benchmark: small, offline, deterministic.

WHAT THIS IS
    30 hand-written SMS-style cases (6 categories x 5 cases) run through
    groundcheck.verify(). Each case states the verdict we INTEND, based on
    the safety policy below. It does not call Ollama or any model.

WHAT THIS IS NOT
    It is not a measure of real-world accuracy. The cases were written by the
    project author, in English, in a few bank-SMS styles, and the verifier was
    built with these same styles in mind. A 100% score here means "the
    verifier does what its author intended on these 30 examples", nothing more.

SAFETY POLICY (how expected verdicts were chosen)
    SUPPORTED    only when the amount is in the source AND its role is clear.
    UNSUPPORTED  the amount is absent, or clearly has a different role.
    AMBIGUOUS    the role is unclear or conflicting. Never accept it.
    The costly mistake is a FALSE ACCEPTANCE: SUPPORTED when the expected
    verdict is UNSUPPORTED or AMBIGUOUS. That count should be zero.

RUN (from the repository root)
    py -m benchmarks.benchmark
    py benchmarks/benchmark.py        # also works
Exit code: 0 if every case passes, 1 if any case fails, 2 if the case list
itself is invalid (duplicate cases).
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):  # started as "py benchmarks/benchmark.py"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groundcheck import Role, Verdict, verify

# Short names so the case table below stays readable.
T, B = Role.TRANSACTION, Role.BALANCE
SUPPORTED, UNSUPPORTED, AMBIGUOUS = (
    Verdict.SUPPORTED,
    Verdict.UNSUPPORTED,
    Verdict.AMBIGUOUS,
)


@dataclass(frozen=True)
class Case:
    name: str
    category: str
    source: str
    amount: str  # the claim being checked, e.g. "₹8,400"
    role: Role  # the role the claim gives that amount
    expected: Verdict
    why: str  # why this verdict is the intended one


@dataclass(frozen=True)
class Outcome:
    case: Case
    actual: str  # a Verdict value, or "ERROR" if verify() raised
    reason: str
    evidence_ok: bool  # every evidence snippet really occurs in the source

    @property
    def passed(self) -> bool:
        return self.actual == self.case.expected.value and self.evidence_ok


# --- Shared messages ----------------------------------------------------------

DEBIT = "Your account was debited ₹1,250. Your available balance is ₹8,400."
CREDIT = "Your account was credited ₹2,000. Your available balance is ₹9,500."
RS_STYLE = "Rs. 1,250.00 was debited from your account. Avl bal Rs. 8,400.50."
COMMA = "Debited ₹1,250, balance is ₹8,400."

# --- The cases ----------------------------------------------------------------

CASES: tuple[Case, ...] = (
    # 1. Correct claims: clear wording, right role -> SUPPORTED
    Case("correct_transaction_rupee_symbol", "correct", DEBIT, "₹1,250", T, SUPPORTED,
         "1,250 is next to 'debited' and nothing else competes."),
    Case("correct_balance_rupee_symbol", "correct", DEBIT, "₹8,400", B, SUPPORTED,
         "8,400 is next to 'balance'."),
    Case("correct_credit_transaction", "correct", CREDIT, "2000", T, SUPPORTED,
         "'credited' is a transaction cue."),
    Case("correct_comma_separated_clauses", "correct", COMMA, "8400", B, SUPPORTED,
         "The comma separates the two clauses, so each amount has one cue."),
    Case("correct_paid_wording", "correct", "You paid ₹499 to ACME STORE. Balance: ₹1,501.",
         "499", T, SUPPORTED, "'paid' is a transaction cue; the other amount is a balance."),

    # 2. Role swaps: real amount, wrong role -> UNSUPPORTED
    Case("swap_balance_as_transaction", "swap", DEBIT, "₹8,400", T, UNSUPPORTED,
         "The Gemma failure: the balance is claimed as the transaction."),
    Case("swap_transaction_as_balance", "swap", DEBIT, "₹1,250", B, UNSUPPORTED,
         "The debited amount is claimed as the balance."),
    Case("swap_credit_message", "swap", CREDIT, "9500", T, UNSUPPORTED,
         "9,500 is the balance, not the credited amount."),
    Case("swap_comma_separated_clauses", "swap", COMMA, "1250", B, UNSUPPORTED,
         "1,250 is clearly the debit; claiming it as the balance is wrong."),
    Case("swap_rs_marker_with_decimals", "swap", RS_STYLE, "8400.50", T, UNSUPPORTED,
         "'Avl bal' marks 8,400.50 as the balance."),

    # 3. Absent values: claimed amount is not in the source -> UNSUPPORTED
    Case("absent_unrelated_amount", "absent", DEBIT, "1300", T, UNSUPPORTED,
         "1,300 appears nowhere."),
    Case("absent_transposed_digits", "absent", DEBIT, "1520", T, UNSUPPORTED,
         "1,520 is a digit-swapped 1,250, not the same amount."),
    Case("absent_decimals_differ", "absent", "Your account was debited ₹1,250.50. "
         "Your available balance is ₹8,400.", "1250", T, UNSUPPORTED,
         "1,250.50 and 1,250 are different amounts."),
    Case("absent_no_amounts_in_source", "absent",
         "Your account was debited. Please check the app for details.", "1250", T,
         UNSUPPORTED, "The source contains no amount at all."),
    Case("absent_malformed_number_not_truncated", "absent",
         "Your account was debited ₹1,2500. Your available balance is ₹8,400.", "1250", T,
         UNSUPPORTED, "'1,2500' must not be cut short and read as 1,250."),

    # 4. Conflicting wording: cues disagree or compete -> AMBIGUOUS
    Case("conflict_two_transaction_amounts", "conflict", "Debited ₹500. Credited ₹700.",
         "500", T, AMBIGUOUS, "Two different transaction amounts; which one is 'the' one?"),
    Case("conflict_same_amount_two_roles", "conflict", "Debited ₹500. Balance ₹500.",
         "500", T, AMBIGUOUS, "The same value is both a debit and a balance."),
    Case("conflict_both_cues_in_one_clause", "conflict", "Your balance was debited ₹1,250.",
         "1250", T, AMBIGUOUS, "'balance' and 'debited' both sit next to the amount."),
    Case("conflict_neighbour_cue_leak", "conflict",
         "₹8,400 is your balance after ₹1,250 was debited.", "1250", T, AMBIGUOUS,
         "'balance' belongs to 8,400, but it is also near 1,250; do not guess."),
    Case("conflict_shared_cue_over_list", "conflict", "₹500 and ₹700 were debited.",
         "700", T, AMBIGUOUS, "One cue covers a list; ₹500's role is not established."),

    # 5. Unclear roles: not enough evidence to accept -> AMBIGUOUS
    Case("unclear_no_cue_words", "unclear", "Your transaction limit is ₹5,000.", "5000", T,
         AMBIGUOUS, "A limit is not a debit; no movement word is present."),
    Case("unclear_extra_amount_unknown_role", "unclear", "Debited ₹500. Limit ₹5,000.",
         "500", T, AMBIGUOUS, "The unexplained ₹5,000 could also be a transaction."),
    Case("unclear_and_without_comma", "unclear", "Debited ₹1,250 and balance is ₹8,400.",
         "1250", T, AMBIGUOUS, "No punctuation between clauses; the verifier refuses to guess."),
    Case("unclear_balance_claim_no_cue", "unclear", "Reference amount ₹8,400.", "8400", B,
         AMBIGUOUS, "Nothing says this amount is a balance."),
    Case("unclear_balance_after_debit_phrase", "unclear",
         "Your balance after debit of ₹1,250 is ₹8,400.", "1250", T, AMBIGUOUS,
         "Both cue types sit before 1,250 and 8,400 has no cue."),

    # 6. Formatting, punctuation, several amounts: should still be recognised
    Case("format_indian_lakh_grouping", "format",
         "Your account was debited ₹1,25,000. Your available balance is ₹8,40,000.",
         "125000", T, SUPPORTED, "1,25,000 is the Indian grouping of 125000."),
    Case("format_claim_with_rs_and_decimals", "format", DEBIT, "Rs. 1,250.00", T, SUPPORTED,
         "'Rs. 1,250.00' is numerically the same as ₹1,250."),
    Case("format_inr_marker_newline_separated", "format",
         "INR 3200 was withdrawn\nBalance INR 10,000", "10000", B, SUPPORTED,
         "A newline separates the clauses; INR is a currency marker."),
    Case("format_rs_no_space_trailing_period", "format", "Debited Rs.1250.", "1,250", T,
         SUPPORTED, "'Rs.1250.' must not mistake the periods for another amount."),
    Case("format_semicolon_and_exclamation", "format", "Debited ₹1,250; avl bal ₹8,400!",
         "8400", B, SUPPORTED, "';' and '!' end the clauses."),
)


# --- Running and reporting ------------------------------------------------------

def check_no_duplicates(cases: tuple[Case, ...]) -> list[str]:
    """Return a list of problems (empty if the case list is clean)."""
    problems = []
    names = Counter(c.name for c in cases)
    problems += [f"duplicate case name: {n}" for n, k in names.items() if k > 1]
    keys = Counter((c.source, c.amount, c.role) for c in cases)
    problems += [f"duplicate case (same source, amount, role): {k[0]!r} {k[1]!r}"
                 for k, n in keys.items() if n > 1]
    return problems


def run_case(case: Case) -> Outcome:
    try:
        result = verify(case.source, case.amount, case.role)
    except Exception as exc:  # a crash is a failed case, not a crashed benchmark
        return Outcome(case, "ERROR", f"verify() raised {type(exc).__name__}: {exc}", False)
    evidence_ok = all(snippet in case.source for snippet in result.evidence)
    return Outcome(case, result.verdict.value, result.reason, evidence_ok)


def print_case(number: int, outcome: Outcome) -> None:
    status = "PASS" if outcome.passed else "FAIL"
    case = outcome.case
    print(f"[{status}] {number:02d} {case.name:<40} "
          f"expected={case.expected.value:<11} actual={outcome.actual}")
    print(f"       reason: {outcome.reason}")
    if not outcome.passed:
        print(f"       intent: {case.why}")
        if not outcome.evidence_ok:
            print("       problem: evidence contains text that is not in the source")


def print_summary(outcomes: list[Outcome]) -> None:
    total = len(outcomes)
    passed = sum(o.passed for o in outcomes)
    expected = Counter(o.case.expected.value for o in outcomes)
    actual = Counter(o.actual for o in outcomes)
    false_accepts = [o for o in outcomes
                     if o.case.expected != SUPPORTED and o.actual == SUPPORTED.value]
    over_cautious = [o for o in outcomes
                     if o.case.expected == SUPPORTED and o.actual != SUPPORTED.value]

    def counts(counter: Counter) -> str:
        return "  ".join(f"{k}={counter.get(k, 0)}" for k in
                         ("SUPPORTED", "UNSUPPORTED", "AMBIGUOUS", "ERROR") if counter.get(k, 0)
                         or k != "ERROR")

    print("\n" + "=" * 70)
    print("SUMMARY (synthetic, limited benchmark; see the notes in benchmark.py)")
    print(f"Total cases            : {total}")
    print(f"Exact-match accuracy   : {passed}/{total} ({100 * passed / total:.1f}%)")
    print(f"Expected verdicts      : {counts(expected)}")
    print(f"Actual verdicts        : {counts(actual)}")
    print(f"False acceptances      : {len(false_accepts)}   "
          "(expected UNSUPPORTED/AMBIGUOUS, got SUPPORTED)")
    for o in false_accepts:
        print(f"    - {o.case.name}")
    print(f"Over-cautious cases    : {len(over_cautious)}   "
          "(expected SUPPORTED, got something else)")
    for o in over_cautious:
        print(f"    - {o.case.name}")
    failed = total - passed
    print("RESULT: " + ("ALL CASES PASSED" if failed == 0 else f"{failed} CASE(S) FAILED"))


def main(cases: tuple[Case, ...] = CASES) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # keep ₹ printable on any console
        sys.stdout.reconfigure(errors="replace")

    problems = check_no_duplicates(cases)
    if problems:
        for problem in problems:
            print(f"Invalid case list: {problem}", file=sys.stderr)
        return 2

    print("GroundCheck SYNTHETIC benchmark: hand-written cases, offline, no model.\n")
    outcomes = [run_case(case) for case in cases]
    for number, outcome in enumerate(outcomes, start=1):
        print_case(number, outcome)
    print_summary(outcomes)
    return 0 if all(o.passed for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())