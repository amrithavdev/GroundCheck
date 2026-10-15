"""GroundCheck verifier prototype.

Question answered: "Does this source text support the claim that
<amount> plays <role> (e.g. the transaction amount)?"

No Ollama, no UI, standard library only.

Design rule: when the role of an amount cannot be established reliably,
the answer is AMBIGUOUS, never SUPPORTED.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"


class Role(str, Enum):
    TRANSACTION = "transaction"
    BALANCE = "balance"
    UNKNOWN = "unknown"  # internal: an amount whose role we can't tell


@dataclass(frozen=True)
class VerificationResult:
    verdict: Verdict
    reason: str
    evidence: tuple[str, ...]  # exact snippets copied from the source


@dataclass(frozen=True)
class _Mention:
    value: Decimal
    role: Role
    clause: str


# --- Patterns ---------------------------------------------------------------

# A number such as 1250, 1,250, 12,50,000 or 1,250.50
_NUMBER = r"\d+(?:,\d{2,3})*(?:\.\d+)?"

# In the source, an amount must carry a currency marker: ₹, Rs, Rs. or INR.
# The lookahead stops a malformed number such as "1,2500" from being
# silently cut short and read as 1250.
_AMOUNT_IN_TEXT = re.compile(
    rf"(?:₹|\b(?:Rs\.?|INR))\s*({_NUMBER})(?!\d|,\d)", re.IGNORECASE
)

# A candidate amount supplied by the caller (currency marker optional).
_CANDIDATE = re.compile(
    rf"\s*(?:₹|Rs\.?|INR)?\s*({_NUMBER})\s*", re.IGNORECASE
)

# Cue words. Deliberately small and limited to words that describe money
# MOVING. Nouns like "transaction limit" or "purchase limit" are not cues.
_TRANSACTION_CUES = re.compile(
    r"\b(?:debited|credited|withdrawn|spent|paid|received|transferred|"
    r"deposited|purchased|(?:debit|credit|purchase|transaction|txn) of)\b",
    re.IGNORECASE,
)
_BALANCE_CUES = re.compile(r"\b(?:balance|bal)\b", re.IGNORECASE)

# A clause ends at . ; ! ? , or a newline.
_CLAUSE_END = re.compile(r"[.;!?,\n]")


# --- Helpers ----------------------------------------------------------------

def parse_amount(value: str | int | float | Decimal) -> Decimal:
    """Turn '₹1,250', '1250.00', 1250 or Decimal('1250.00') into a Decimal."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    match = _CANDIDATE.fullmatch(str(value))
    if not match:
        raise ValueError(f"Could not read an amount from {value!r}")
    try:
        return Decimal(match.group(1).replace(",", ""))
    except InvalidOperation as exc:  # pragma: no cover - regex prevents this
        raise ValueError(f"Could not read an amount from {value!r}") from exc


def _roles_in(text: str) -> set[Role]:
    found: set[Role] = set()
    if _TRANSACTION_CUES.search(text):
        found.add(Role.TRANSACTION)
    if _BALANCE_CUES.search(text):
        found.add(Role.BALANCE)
    return found


def _classify(before: str, after: str) -> Role:
    """Decide an amount's role from the words on BOTH sides of it.

    Exactly one kind of cue -> that role.
    No cues, or cues for both roles -> UNKNOWN (we refuse to guess).
    """
    roles = _roles_in(before) | _roles_in(after)
    if len(roles) == 1:
        return next(iter(roles))
    return Role.UNKNOWN


def find_mentions(source: str) -> list[_Mention]:
    """Find every currency amount in the source and classify its role."""
    matches = list(_AMOUNT_IN_TEXT.finditer(source))

    # Hide the amounts so the '.' in "Rs." and the ',' in "1,250" are not
    # mistaken for the end of a clause.
    masked = source
    for m in matches:
        masked = masked[: m.start()] + "#" * (m.end() - m.start()) + masked[m.end():]
    breaks = [m.start() for m in _CLAUSE_END.finditer(masked)]

    mentions: list[_Mention] = []
    for i, m in enumerate(matches):
        clause_start = max((b + 1 for b in breaks if b < m.start()), default=0)
        clause_end = min((b for b in breaks if b >= m.end()), default=len(source))

        # If another amount is in the same clause, only look at the text
        # between the two amounts.
        prev_end = matches[i - 1].end() if i > 0 else 0
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(source)

        before = source[max(clause_start, prev_end): m.start()]
        after = source[m.end(): min(clause_end, next_start)]

        mentions.append(
            _Mention(
                value=Decimal(m.group(1).replace(",", "")),
                role=_classify(before, after),
                clause=source[clause_start:clause_end].strip(),
            )
        )
    return mentions


# --- Public API -------------------------------------------------------------

def verify(
    source: str,
    amount: str | int | float | Decimal,
    role: Role = Role.TRANSACTION,
) -> VerificationResult:
    """Check that `amount` appears in `source` AND plays `role` there."""
    if role not in (Role.TRANSACTION, Role.BALANCE):
        raise ValueError("role must be Role.TRANSACTION or Role.BALANCE")

    target = parse_amount(amount)
    mentions = find_mentions(source)
    matches = [m for m in mentions if m.value == target]

    # 1. Is the amount in the source at all?
    if not matches:
        return VerificationResult(
            Verdict.UNSUPPORTED,
            f"The amount {target} does not appear in the source.",
            (),
        )

    evidence = tuple(dict.fromkeys(m.clause for m in matches))
    roles_found = {m.role for m in matches}

    # 2. Does it play the claimed role (and only that role)?
    if roles_found == {role}:
        # Any OTHER amount with the same or an unclear role could be the
        # real one, so SUPPORTED would be an over-claim.
        rivals = [
            m for m in mentions
            if m.value != target and m.role in (role, Role.UNKNOWN)
        ]
        if rivals:
            rival_evidence = tuple(dict.fromkeys(m.clause for m in rivals))
            return VerificationResult(
                Verdict.AMBIGUOUS,
                f"The amount looks like a {role.value} amount, but the source "
                f"has other amounts that are also {role.value} amounts or "
                f"whose role is unclear.",
                tuple(dict.fromkeys(evidence + rival_evidence)),
            )
        return VerificationResult(
            Verdict.SUPPORTED,
            f"The amount appears in the source as the {role.value} amount.",
            evidence,
        )

    # 3. Same amount shows up with conflicting roles.
    if role in roles_found:
        return VerificationResult(
            Verdict.AMBIGUOUS,
            "The amount appears more than once with conflicting roles.",
            evidence,
        )

    # 4. We can't tell what the amount is.
    if Role.UNKNOWN in roles_found:
        return VerificationResult(
            Verdict.AMBIGUOUS,
            "The amount is in the source, but its role is unclear.",
            evidence,
        )

    # 5. We can tell, and it is a different role than claimed.
    actual = ", ".join(sorted(r.value for r in roles_found))
    return VerificationResult(
        Verdict.UNSUPPORTED,
        f"The amount is in the source, but as the {actual} amount, "
        f"not the {role.value} amount.",
        evidence,
    )
    