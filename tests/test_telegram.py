import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from precision_tap.params import TelegramConfig
from precision_tap.telegram import TelegramClient, split_message


class Handler(BaseHTTPRequestHandler):
    requests = []
    fail_first = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode()
        type(self).requests.append((self.path, raw))
        if len(self.requests) <= self.fail_first:
            body = json.dumps({"ok": False, "error_code": 429, "description": "Too Many Requests",
                               "parameters": {"retry_after": 0.05}}).encode()
            self.send_response(429)
        else:
            body = json.dumps({"ok": True, "result": {"message_id": 42, "chat": {"id": 1}}}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    Handler.requests = []
    Handler.fail_first = 0
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _cfg(base, **kw):
    d = dict(enabled=True, bot_token="123:abc", chat_ids=["1"], parse_mode="HTML",
             min_seconds_between_messages=0.0, timeout_sec=5, max_retries=2, api_base=base,
             send_charts=True)
    d.update(kw)
    return TelegramConfig(**{k: v for k, v in d.items() if k != "send_charts"})


def test_send_text_posts_html(server):
    tg = TelegramClient(_cfg(server))
    res = tg.send_text("<b>TAP 1</b>\nRELIANCE.NS")
    assert all(r.ok for r in res) and res[0].message_id == 42
    path, body = Handler.requests[0]
    assert path.endswith("/bot123:abc/sendMessage")
    assert "parse_mode=HTML" in body and "chat_id=1" in body


def test_rate_limit_retry(server):
    Handler.fail_first = 1
    tg = TelegramClient(_cfg(server, max_retries=3))
    res = tg.send_text("hello")
    assert all(r.ok for r in res), res[0].error
    assert len(Handler.requests) == 2          # one 429, then the successful retry


def test_long_message_is_split(server):
    tg = TelegramClient(_cfg(server))
    tg.send_text("line\n" * 2000)
    assert len(Handler.requests) > 1


def test_split_boundaries():
    chunks = split_message("a" * 5000)
    assert all(len(c) <= 4096 for c in chunks)
    assert "".join(chunks).count("a") == 5000
    text = "\n".join(["x" * 400] * 30)
    chunks = split_message(text, limit=1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert not any(c.startswith("\n") for c in chunks)


def test_dry_run_makes_no_request(server):
    tg = TelegramClient(_cfg(server), dry_run=True)
    assert all(r.ok for r in tg.send_text("nothing sent"))
    assert Handler.requests == []


def test_missing_token_raises(server):
    tg = TelegramClient(_cfg(server, bot_token=""))
    with pytest.raises(Exception):
        tg.send_text("x")


# ─────────────────────────────────────────────────────────────────────────────
# the sample credentials from `.env.example` are not credentials
# ─────────────────────────────────────────────────────────────────────────────
def test_sample_credentials_are_not_configured():
    """`init` copies the template verbatim, so "token is set" must not mean "ready".

    The sample token is well-formed (`<digits>:<secret>`), every check that only
    tests for presence passes, and the only symptom is a 401 on the first alert.
    A client holding it must report itself unconfigured, so alerts take the
    documented log-only path instead of being marked *given up* after a failed
    send.
    """
    from precision_tap.telegram import SAMPLE_BOT_TOKEN, SAMPLE_CHAT_ID, is_sample_credential

    tg = TelegramClient(_cfg("http://127.0.0.1:1", bot_token=SAMPLE_BOT_TOKEN,
                             chat_ids=[SAMPLE_CHAT_ID]))
    assert tg.configured is False
    assert is_sample_credential(SAMPLE_BOT_TOKEN, [SAMPLE_CHAT_ID]) is True

    # a real-looking token is untouched, and a *missing* token is a different
    # problem (missing ≠ sample — they are reported differently by `doctor`)
    assert TelegramClient(_cfg("http://127.0.0.1:1")).configured is True
    assert is_sample_credential("", []) is False
    assert is_sample_credential("123:abc", ["42"]) is False
    # the sample *token* cannot authenticate whatever chat id sits next to it ...
    assert is_sample_credential(SAMPLE_BOT_TOKEN, ["555"]) is True
    # ... while replacing it is enough to count as configured (a wrong chat id
    # then fails honestly, with Telegram's own "chat not found")
    assert is_sample_credential("999:realtoken", [SAMPLE_CHAT_ID]) is False


def test_the_sample_constants_match_the_shipped_template():
    """One source of truth: `.env.example` is what `init` writes into `.env`."""
    from pathlib import Path

    from precision_tap.telegram import SAMPLE_BOT_TOKEN, SAMPLE_CHAT_ID

    env = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    assert f"TELEGRAM_BOT_TOKEN={SAMPLE_BOT_TOKEN}" in env
    assert f"TELEGRAM_CHAT_ID={SAMPLE_CHAT_ID}" in env
