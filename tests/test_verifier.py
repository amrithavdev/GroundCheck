from decimal import Decimal

import pytest

from groundcheck import Role, Verdict, verify

MSG = "Your account was debited ₹1,250. Your available balance is ₹8,400."


# --- The four required behaviours -------------------------------------------

def test_correct_transaction_amount_is_supported():
    result = verify(MSG, "₹1,250", Role.TRANSACTION)
    assert result.verdict == Verdict.SUPPORTED
    assert result.evidence == ("Your account was debited ₹1,250",)


def test_balance_amount_claimed_as_transaction_is_unsupported():
    # The exact mistake Gemma 3 1B made in the manual test.
    result = verify(MSG, "8400", Role.TRANSACTION)
    assert result.verdict == Verdict.UNSUPPORTED
    assert "balance" in result.reason
    assert result.evidence == ("Your available balance is ₹8,400",)


def test_missing_amount_is_unsupported():
    result = verify(MSG, "₹1,300", Role.TRANSACTION)
    assert result.verdict == Verdict.UNSUPPORTED
    assert "does not appear" in result.reason
    assert result.evidence == ()


def test_both_cue_types_in_one_clause_is_ambiguous():
    source = "Your balance was debited ₹1,250."
    result = verify(source, "1250", Role.TRANSACTION)
    assert result.verdict == Verdict.AMBIGUOUS
    assert "unclear" in result.reason
    assert result.evidence == ("Your balance was debited ₹1,250",)


# --- Roles and several amounts ----------------------------------------------

def test_balance_amount_claimed_as_balance_is_supported():
    assert verify(MSG, "₹8,400", Role.BALANCE).verdict == Verdict.SUPPORTED

def test_swapped_transaction_and_balance_amounts_are_not_supported():
    # Each amount is real, but each is claimed in the other amount's role.
    balance_as_transaction = verify(MSG, "₹8,400", Role.TRANSACTION)
    transaction_as_balance = verify(MSG, "₹1,250", Role.BALANCE)
    assert balance_as_transaction.verdict != Verdict.SUPPORTED
    assert transaction_as_balance.verdict != Verdict.SUPPORTED


def test_cue_from_neighbouring_amount_does_not_cause_wrong_verdict():
    # "balance" belongs to 8,400, not 1,250. We must not say UNSUPPORTED.
    source = "₹8,400 is your balance after ₹1,250 was debited."
    assert verify(source, "1250", Role.TRANSACTION).verdict == Verdict.AMBIGUOUS


def test_amounts_separated_by_comma_are_classified_separately():
    source = "Debited ₹1,250, balance is ₹8,400."
    assert verify(source, "1250", Role.TRANSACTION).verdict == Verdict.SUPPORTED
    assert verify(source, "8400", Role.BALANCE).verdict == Verdict.SUPPORTED


def test_amounts_joined_by_and_are_ambiguous():
    # No punctuation separates the clauses, so we refuse to guess.
    source = "Debited ₹1,250 and balance is ₹8,400."
    assert verify(source, "1250", Role.TRANSACTION).verdict == Verdict.AMBIGUOUS


def test_shared_cue_after_a_list_of_amounts_is_ambiguous():
    source = "₹500 and ₹700 were debited."
    assert verify(source, "500", Role.TRANSACTION).verdict == Verdict.AMBIGUOUS
    assert verify(source, "700", Role.TRANSACTION).verdict == Verdict.AMBIGUOUS


def test_two_different_transaction_amounts_are_ambiguous():
    source = "Debited ₹500. Credited ₹700."
    assert verify(source, "500", Role.TRANSACTION).verdict == Verdict.AMBIGUOUS


def test_extra_amount_with_unclear_role_prevents_supported():
    source = "Debited ₹500. Limit ₹5,000."
    assert verify(source, "500", Role.TRANSACTION).verdict == Verdict.AMBIGUOUS


def test_same_amount_with_two_roles_is_ambiguous():
    source = "Debited ₹500. Balance ₹500."
    assert verify(source, "500", Role.TRANSACTION).verdict == Verdict.AMBIGUOUS


def test_amount_without_cue_words_is_ambiguous():
    source = "Your transaction limit is ₹5,000."
    assert verify(source, "5000", Role.TRANSACTION).verdict == Verdict.AMBIGUOUS


# --- Amount normalisation ----------------------------------------------------

@pytest.mark.parametrize(
    "candidate",
    ["₹1,250", "1,250", "1250", "1250.00", "Rs. 1,250.00", "INR 1250",
     1250, 1250.0, Decimal("1250.00")],
)
def test_candidate_formats_all_match(candidate):
    source = "Your account was debited ₹1,250."
    assert verify(source, candidate, Role.TRANSACTION).verdict == Verdict.SUPPORTED


@pytest.mark.parametrize(
    "source",
    ["₹1,250 was debited.", "Rs. 1,250.00 was debited.",
     "Debited Rs.1250.", "INR 1250 debited."],
)
def test_source_formats_all_match(source):
    assert verify(source, "1250", Role.TRANSACTION).verdict == Verdict.SUPPORTED


def test_malformed_source_number_is_not_cut_short():
    # "1,2500" must NOT be read as 1,250.
    source = "Your account was debited ₹1,2500."
    result = verify(source, "1250", Role.TRANSACTION)
    assert result.verdict == Verdict.UNSUPPORTED
    assert "does not appear" in result.reason


def test_different_decimal_amount_is_not_equal():
    source = "Your account was debited ₹1,250.50."
    assert verify(source, "1250", Role.TRANSACTION).verdict == Verdict.UNSUPPORTED


# --- Input checks and evidence ------------------------------------------------

def test_unreadable_candidate_raises():
    with pytest.raises(ValueError):
        verify(MSG, "a lot")


def test_unknown_role_cannot_be_claimed():
    with pytest.raises(ValueError):
        verify(MSG, "1250", Role.UNKNOWN)


@pytest.mark.parametrize("amount", ["1250", "8400", "500"])
def test_evidence_is_always_copied_from_source(amount):
    result = verify(MSG, amount, Role.TRANSACTION)
    assert all(snippet in MSG for snippet in result.evidence)