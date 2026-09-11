#!/usr/bin/env python3
"""Local stand-in for the Telegram Bot API — lets you test delivery without internet.

    python tools/mock_telegram_server.py 8081        # prints every received message
    python -m precision_tap scan --set telegram.api_base=http://127.0.0.1:8081 \\
           --set telegram.bot_token=123:local --set telegram.chat_ids=1

Responds ok to sendMessage / sendPhoto / getMe / getUpdates and echoes the decoded
payload, so you can eyeball exact formatting and inline buttons.
"""
import json
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
SEQ = [0]


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw_bytes = self.rfile.read(n)
        raw = raw_bytes.decode("utf-8", "ignore")
        method = self.path.rsplit("/", 1)[-1]
        ctype = self.headers.get("Content-Type") or ""
        fields = {}
        photo = 0
        if ctype.startswith("multipart/form-data"):
            import email
            msg = email.message_from_bytes(
                b"Content-Type: " + ctype.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw_bytes)
            for part in msg.walk():
                name = part.get_param('name') or ''
                if not name:
                    continue
                if part.get_filename():
                    photo += len(part.get_payload(decode=True) or b"")
                    fields[name] = f"<{photo} bytes uploaded>"
                else:
                    fields[name] = (part.get_payload(decode=True) or b"").decode("utf-8", "ignore")
        else:
            fields = {k: v[0] for k, v in urllib.parse.parse_qs(raw, keep_blank_values=True).items()}
        SEQ[0] += 1
        if method == "getMe":
            self._json({"ok": True, "result": {"id": 1, "is_bot": True, "username": "mock_bot",
                                               "first_name": "Precision Tap (mock)"}})
            return
        if method == "getUpdates":
            self._json({"ok": True, "result": [{"update_id": 1, "message": {
                "chat": {"id": 111888, "type": "private", "first_name": "You"}}}]})
            return
        print(f"\n─── {method} ─────────────────────────────────────────── chat={fields.get('chat_id')}")
        for key in ("parse_mode", "reply_markup", "message_thread_id"):
            if fields.get(key):
                print(f"[{key}] {fields[key]}")
        text = fields.get("text") or fields.get("caption") or ""
        print(text.replace("</b>", "**").replace("<b>", "**").replace("<i>", "_").replace("</i>", "_")
                 .replace("<code>", "`").replace("</code>", "`"))
        if fields.get("photo"):
            print(f"[attachment] {fields['photo']}")
        try:
            chat_id = int(fields.get("chat_id") or 0)
        except ValueError:
            chat_id = 0
        self._json({"ok": True, "result": {"message_id": SEQ[0], "date": 0,
                                           "chat": {"id": chat_id, "type": "private"}}})

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


print(f"mock Telegram API on http://127.0.0.1:{PORT}  (ctrl-c to stop)")
HTTPServer(("0.0.0.0", PORT), H).serve_forever()
