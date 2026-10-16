"""GroundCheck demo: extract with a local model, then verify independently.

Run from the project root (the folder containing the `groundcheck` folder):

    py -3.13 -m groundcheck.demo

Step 1: ask Gemma (through Ollama) to extract two fields from an SMS.
Step 2: ignore everything the model says about WHY, and check each extracted
        value against the original SMS using the GroundCheck verifier.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from groundcheck import Role, verify

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:1b"
TIMEOUT_SECONDS = 120  # the first request can be slow while the model loads

SMS = "Your account was debited ₹1,250. Your available balance is ₹8,400."

# Which verifier role each extracted field must have in the SMS.
FIELDS = {
    "transaction_amount": Role.TRANSACTION,
    "available_balance": Role.BALANCE,
}

SYSTEM_PROMPT = (
    "Extract two fields from a bank SMS. "
    'Reply with JSON only, in exactly this shape: '
    '{"transaction_amount": <amount>, "available_balance": <amount>}. '
    "transaction_amount is the money that moved in this transaction. "
    "available_balance is the money left in the account. "
    "Use null for a field that is not in the message."
)


class DemoError(Exception):
    """A problem we can explain to the user in one or two lines."""


def ask_model(sms: str) -> str:
    """Send the SMS to Ollama and return the model's text reply."""
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sms},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # server answered with an error
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        hint = f" Try: ollama pull {MODEL}" if exc.code == 404 else ""
        raise DemoError(f"Ollama returned HTTP {exc.code}: {detail}{hint}") from exc
    except OSError as exc:  # connection refused, timeout, reset, ...
        raise DemoError(
            f"Could not reach Ollama at {OLLAMA_URL} ({exc}). "
            "Is Ollama running? You can check with: ollama list"
        ) from exc

    try:
        return json.loads(raw)["message"]["content"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DemoError(f"Unexpected reply format from Ollama: {raw[:200]}") from exc


def parse_extraction(content: str) -> dict:
    """Turn the model's text into a dict, or explain why we cannot."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DemoError(f"The model did not return valid JSON. Raw output:\n{content}") from exc
    if not isinstance(data, dict):
        raise DemoError(f"The model returned JSON, but not an object. Raw output:\n{content}")
    return data


def check_field(name: str, role: Role, value: object) -> None:
    """Verify one extracted value against the SMS and print the result."""
    print(f"\n{name} = {value!r}   (must be the {role.value} amount)")

    if value is None:
        print("  Not extracted by the model, so there is nothing to verify.")
        return
    # bool is a subclass of int in Python, so rule it out explicitly.
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        print("  Cannot verify: the model gave something that is not an amount.")
        return

    try:
        result = verify(SMS, value, role)
    except ValueError as exc:  # e.g. the model answered "a lot"
        print(f"  Cannot verify: {exc}")
        return

    print(f"  Verdict : {result.verdict.value}")
    print(f"  Reason  : {result.reason}")
    if result.evidence:
        for snippet in result.evidence:
            print(f"  Evidence: {snippet!r}")
    else:
        print("  Evidence: (none)")


def main() -> int:
    # Let the rupee sign print even if the console or a file is not UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    print(f"Source SMS:\n  {SMS}")
    try:
        content = ask_model(SMS)
        print(f"\nRaw output from {MODEL}:\n  {content.strip()}")
        data = parse_extraction(content)
    except DemoError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

    for name, role in FIELDS.items():
        check_field(name, role, data.get(name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())