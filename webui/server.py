"""GroundCheck web UI server (standard library only, no Flask).

Run from the repository root:   py -m webui.server
Then open http://127.0.0.1:8000

Endpoints
    GET  /                  the page (and /app.css, /app.js: a fixed whitelist)
    GET  /api/benchmark     runs the real benchmark cases, returns the results
    POST /api/verify        manual claims, verifier only (no model, works offline)
    POST /api/analyze       Gemma extraction through Ollama, then verification
"""

from __future__ import annotations

import http.client
import json
import sys
import traceback
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__ in (None, ""):  # started as "py webui/server.py"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark import CASES, run_case
from groundcheck import demo, verify

from benchmarks.evaluate import (
    RESULTS_PATH,
    load_results,
    compute_metrics,
    verifier_drift,
)

HOST, PORT = "127.0.0.1", 8000
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
STATIC = Path(__file__).resolve().parent / "static"
FILES = {  # the ONLY files this server will ever serve
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_BODY = 8 * 1024  # bytes
MAX_SMS = 1000  # characters
MAX_CLAIM = 40  # characters
HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    ),
}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def is_amount(value: object) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def check_value(sms: str, name: str, role, value: object) -> dict:
    """Verify one claimed value against the SMS. Returns plain JSON data.

    (demo.check_field is not used: it prints, and it always checks the
    built-in demo SMS instead of the SMS passed in.)
    """
    shown = None if value is None else (str(value) if is_amount(value) else repr(value)[:MAX_CLAIM])
    base = {"field": name, "role": role.value, "claimed": shown, "evidence": []}
    if value is None:
        return {**base, "status": "not_extracted", "verdict": None,
                "reason": "No value was given, so there is nothing to verify."}
    if not is_amount(value):
        return {**base, "status": "invalid", "verdict": None,
                "reason": "The value is not an amount, so it cannot be verified."}
    try:
        result = verify(sms, value, role)
    except (ValueError, ArithmeticError) as exc:
        return {**base, "status": "invalid", "verdict": None, "reason": str(exc)}
    return {**base, "status": "checked", "verdict": result.verdict.value,
            "reason": result.reason, "evidence": list(result.evidence)}


def get_sms(body: dict) -> str:
    sms = body.get("sms")
    if not isinstance(sms, str) or not sms.strip():
        raise ApiError(400, "Field 'sms' must be a non-empty string.")
    if len(sms) > MAX_SMS:
        raise ApiError(400, f"The SMS is too long (maximum {MAX_SMS} characters).")
    return sms.strip()


def analyze(body: dict) -> dict:
    sms = get_sms(body)
    try:
        content = demo.ask_model(sms)
        extracted = demo.parse_extraction(content)
    except demo.DemoError as exc:
        raise ApiError(502, str(exc)[:600])
    except (http.client.HTTPException, UnicodeDecodeError, TypeError) as exc:
        raise ApiError(502, f"Unexpected reply from Ollama: {exc}"[:600])
    return {
        "mode": "gemma",
        "model": demo.MODEL,
        "raw": content[:2000],
        "fields": [check_value(sms, name, role, extracted.get(name))
                   for name, role in demo.FIELDS.items()],
    }


def verify_manual(body: dict) -> dict:
    sms = get_sms(body)
    claims = body.get("claims")
    if not isinstance(claims, dict):
        raise ApiError(400, "Field 'claims' must be an object.")
    values = {}
    for name in demo.FIELDS:
        value = claims.get(name)
        if isinstance(value, str):
            value = value.strip() or None
        if value is not None and not is_amount(value):
            raise ApiError(400, f"Claim '{name}' must be text or a number.")
        if isinstance(value, str) and len(value) > MAX_CLAIM:
            raise ApiError(400, f"Claim '{name}' is too long.")
        values[name] = value
    if all(v is None for v in values.values()):
        raise ApiError(400, "Enter at least one claim to verify.")
    return {
        "mode": "manual",
        "fields": [check_value(sms, name, role, values[name])
                   for name, role in demo.FIELDS.items()],
    }


def benchmark() -> dict:
    outcomes = [run_case(case) for case in CASES]
    expected = lambda o: o.case.expected.value  # noqa: E731
    return {
        "summary": {
            "total": len(outcomes),
            "passed": sum(o.passed for o in outcomes),
            "expected": dict(Counter(expected(o) for o in outcomes)),
            "actual": dict(Counter(o.actual for o in outcomes)),
            "false_acceptances": sum(
                expected(o) != "SUPPORTED" and o.actual == "SUPPORTED" for o in outcomes),
            "over_cautious": sum(
                expected(o) == "SUPPORTED" and o.actual != "SUPPORTED" for o in outcomes),
        },
        "cases": [
            {"name": o.case.name, "category": o.case.category, "source": o.case.source,
             "amount": o.case.amount, "role": o.case.role.value, "expected": expected(o),
             "actual": o.actual, "passed": o.passed, "reason": o.reason}
            for o in outcomes
        ],
    }

def evaluation() -> dict:
    """Return aggregate metrics from the saved synthetic evaluation."""
    try:
        document = load_results(RESULTS_PATH)
        records = document["records"]
        metrics = compute_metrics(records)

        return {
            "available": True,
            "model": document.get("model", "Unknown"),
            "label": document.get("label", "Synthetic evaluation; limited evidence."),
            "dataset": metrics["dataset"],
            "baseline": metrics["baseline"],
            "groundcheck": metrics["groundcheck"],
            "correction": metrics["correction"],
            "pipeline": metrics["pipeline"],
            "drift_count": len(verifier_drift(records)),
            "reliability_notes": metrics.get("reliability_notes", []),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {
            "available": False,
            "message": "Evaluation results are unavailable or invalid.",
        }


ROUTES = {"/api/analyze": analyze, "/api/verify": verify_manual}


class Handler(BaseHTTPRequestHandler):
    server_version = "GroundCheckUI"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in HEADERS.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        if self.headers.get_content_type() != "application/json":
            raise ApiError(415, "Content-Type must be application/json.")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise ApiError(411, "A Content-Length header is required.")
        if length <= 0 or length > MAX_BODY:
            raise ApiError(413, f"Request body must be 1 to {MAX_BODY} bytes.")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "Request body is not valid JSON.")
        if not isinstance(data, dict):
            raise ApiError(400, "Request body must be a JSON object.")
        return data

    def _handle(self, action) -> None:
        try:
            if self.headers.get("Host") not in ALLOWED_HOSTS:
                raise ApiError(403, "Unexpected Host header.")
            action()
        except ApiError as exc:
            self._json(exc.status, {"error": exc.message})
        except Exception:  # never leak internals to the browser
            traceback.print_exc()
            self._json(500, {"error": "Internal server error."})


    def _get(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in FILES:
            name, content_type = FILES[path]
            try:
                self._send(200, (STATIC / name).read_bytes(), content_type)
            except OSError:
                raise ApiError(404, "File not found.")
        elif path == "/api/benchmark":
            self._json(200, benchmark())
        elif path == "/api/evaluation":
            self._json(200, evaluation())
        else:
            raise ApiError(404, "Not found.")



    def _post(self) -> None:
        action = ROUTES.get(self.path.split("?", 1)[0])
        if action is None:
            raise ApiError(404, "Not found.")
        self._json(200, action(self._read_json()))

    def do_GET(self) -> None:
        self._handle(self._get)

    def do_POST(self) -> None:
        self._handle(self._post)


def main() -> int:
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        print(f"Cannot start on {HOST}:{PORT}: {exc}", file=sys.stderr)
        return 1
    print(f"GroundCheck UI running at http://{HOST}:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
