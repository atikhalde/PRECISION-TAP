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
