"""Telegram Bot API transport.

Plain HTTP (``requests`` when available, ``urllib`` otherwise) — no heavy SDK.
Handles the things that actually bite in production:

* 4096-character message cap -> split on line boundaries
* 429 rate limiting -> honour ``parameters.retry_after``, plus a per-chat minimum
  gap so a 60-symbol scan does not get the bot throttled
* transient 5xx / connection errors -> exponential backoff with jitter
* MarkdownV2 reserved characters -> escaped
* ``getUpdates`` based chat-id discovery (you message the bot, the CLI finds it)
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .params import TelegramConfig

log = logging.getLogger("precision_tap.telegram")

TG_MAX_LEN = 4096
_MDV2_ESCAPES = r"_*[]()~`>#+-=|{}.!\\"


def escape_markdownv2(text: str) -> str:
    return re.sub(rf"([{re.escape(_MDV2_ESCAPES)}])", r"\\\1", str(text))


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def split_message(text: str, limit: int = TG_MAX_LEN) -> List[str]:
    """Split on newline boundaries, falling back to hard cuts."""
    text = text or ""
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    buf = ""
    for line in text.split("\n"):
        while len(line) > limit:                     # pathological single line
            chunks.append(line[:limit])
            line = line[limit:]
        cand = line if not buf else buf + "\n" + line
        if len(cand) > limit:
            chunks.append(buf)
            buf = line
        else:
            buf = cand
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


class TelegramError(RuntimeError):
    pass


class TelegramPermanentError(TelegramError):
    """A rejection that retrying cannot fix (401 bad token, 400 chat not found,
    403 bot blocked / not an admin).

    These are the errors that make a scanner look healthy while nothing is ever
    delivered, so they are surfaced instead of being parked in the retry queue.
    """

    status_code = 0


@dataclass
class SendResult:
    ok: bool
    chat_id: str = ""
    message_id: int = 0
    error: str = ""
    chunks: int = 1
    method: str = "sendMessage"
    permanent: bool = False

    def __bool__(self) -> bool:
        return self.ok


class TelegramClient:
    """Minimal, dependency-light Telegram bot client."""

    def __init__(self, cfg: TelegramConfig, *, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run or not cfg.enabled
        self.base = (cfg.api_base or "https://api.telegram.org").rstrip("/")
        self._last_sent: Dict[str, float] = {}
        self._session = None
        if not dry_run:
            try:
                import requests
                self._session = requests.Session()
                if cfg.proxy:
                    self._session.proxies.update({"http": cfg.proxy, "https": cfg.proxy})
            except Exception:
                self._session = None

    # ── low level ────────────────────────────────────────────────────────
    @property
    def configured(self) -> bool:
        return bool(self.cfg.bot_token) and bool(self.cfg.chat_ids)

    def _url(self, method: str) -> str:
        return f"{self.base}/bot{self.cfg.bot_token}/{method}"

    def _post(self, method: str, fields: Dict[str, Any], files: Optional[Dict[str, Tuple[str, bytes, str]]] = None) -> Dict[str, Any]:
        if self.dry_run:
            log.info("[dry-run] %s %s", method, {k: (v if k != "text" else str(v)[:120] + "…")
                                                 for k, v in fields.items() if v not in (None, "")})
            return {"ok": True, "result": {"message_id": int(time.time()) % 10**6}, "dry_run": True}
        last: Optional[Exception] = None
        for attempt in range(max(1, self.cfg.max_retries + 1)):
            try:
                if self._session is not None:
                    return self._post_requests(method, fields, files)
                return self._post_urllib(method, fields, files)
            except TelegramRateLimited as rl:
                wait = rl.retry_after
                log.warning("telegram 429: backing off %.1fs", wait)
                time.sleep(wait)
                last = rl
            except TelegramPermanentError:
                raise                               # a 401/400/403 never heals by waiting
            except Exception as exc:                # network hiccup / 5xx
                last = exc
                if attempt >= self.cfg.max_retries:
                    break
                time.sleep(min(30.0, (self.cfg.timeout_sec * 0.3) * (1.6 ** attempt)
                               + random.uniform(0, 0.4)))
        raise TelegramError(f"telegram {method} failed: {last}")

    def _handle(self, resp_status: int, body: str, method: str) -> Dict[str, Any]:
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise TelegramError(f"non-JSON response from {method} (HTTP {resp_status}): {body[:200]}") from exc
        if resp_status == 429 or data.get("error_code") == 429:
            ra = ((data.get("parameters") or {}).get("retry_after")) or 3
            err = TelegramRateLimited(f"rate limited: {data.get('description')}")
            err.retry_after = float(ra)
            raise err
        if resp_status >= 400 or not data.get("ok", False):
            msg = f"{method} failed (HTTP {resp_status}): {data.get('description', body[:200])}"
            if 400 <= resp_status < 500 and resp_status != 429:
                err = TelegramPermanentError(msg)
                err.status_code = resp_status
                raise err
            raise TelegramError(msg)
        return data

    def _post_requests(self, method: str, fields: Dict[str, Any], files) -> Dict[str, Any]:
        assert self._session is not None
        clean = {k: v for k, v in fields.items() if v not in (None, "")}
        r = self._session.post(self._url(method), data=clean,
                               files=files, timeout=self.cfg.timeout_sec)
        return self._handle(r.status_code, r.text, method)

    def _post_urllib(self, method: str, fields: Dict[str, Any], files) -> Dict[str, Any]:
        if files:
            raise TelegramError("sending attachments needs the `requests` package")
        data = urlencode({k: v for k, v in fields.items() if v not in (None, "")}).encode()
        req = Request(self._url(method), data=data,
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urlopen(req, timeout=self.cfg.timeout_sec) as fh:   # noqa: S310 (https by config)
            return self._handle(getattr(fh, "status", 200), fh.read().decode("utf-8"), method)

    def _throttle(self, chat_id: str) -> None:
        gap = self.cfg.min_seconds_between_messages
        if gap <= 0:
            return
        prev = self._last_sent.get(chat_id, 0.0)
        delta = time.monotonic() - prev
        if delta < gap:
            time.sleep(gap - delta)
        self._last_sent[chat_id] = time.monotonic()

    # ── public API ───────────────────────────────────────────────────────
    def _require_config(self) -> None:
        if self.dry_run:
            return
        if not self.cfg.bot_token:
            raise TelegramError("TELEGRAM_BOT_TOKEN is not set (see .env / config.yaml)")
        if not self.cfg.chat_ids:
            raise TelegramError("no chat ids configured (TELEGRAM_CHAT_ID)")

    def send_text(self, text: str, *, chat_id: Optional[str] = None,
                  reply_markup: Optional[Dict[str, Any]] = None,
                  disable_preview: bool = True) -> List[SendResult]:
        self._require_config()
        chats = [str(chat_id)] if chat_id else [str(c) for c in self.cfg.chat_ids]
        if not chats:
            raise TelegramError("no chat ids configured (TELEGRAM_CHAT_ID)")
        results: List[SendResult] = []
        for chat in chats:
            for chunk in split_message(text):
                self._throttle(chat)
                fields = {
                    "chat_id": chat,
                    "text": chunk,
                    "disable_web_page_preview": 1 if disable_preview else 0,
                    "disable_notification": 1 if self.cfg.disable_notification else 0,
                    "protect_content": 1 if self.cfg.protect_content else 0,
                }
                pm = self.cfg.parse_mode
                if pm in ("HTML", "MARKDOWNV2", "MARKDOWN"):
                    fields["parse_mode"] = {"MARKDOWNV2": "MarkdownV2", "MARKDOWN": "Markdown"}.get(pm, "HTML")
                if reply_markup:
                    fields["reply_markup"] = json.dumps(reply_markup)
                if self.cfg.message_thread_id:
                    fields["message_thread_id"] = self.cfg.message_thread_id
                try:
                    resp = self._post("sendMessage", fields)
                    results.append(SendResult(ok=True, chat_id=chat,
                                             message_id=(resp.get("result") or {}).get("message_id", 0)))
                except TelegramError as exc:
                    if fields.get("parse_mode") in ("MarkdownV2", "Markdown"):
                        # a stray reserved char should not cost us the alert
                        fields["parse_mode"] = "HTML"
                        fields["text"] = strip_html(chunk)
                        try:
                            resp = self._post("sendMessage", fields)
                            results.append(SendResult(ok=True, chat_id=chat,
                                                     message_id=(resp.get("result") or {}).get("message_id", 0)))
                            continue
                        except TelegramError as exc2:
                            exc = exc2
                    log.error("telegram send failed (%s): %s", chat, exc)
                    results.append(SendResult(ok=False, chat_id=chat, error=str(exc),
                                              permanent=isinstance(exc, TelegramPermanentError)))
        return results

    def send_photo(self, caption: str, photo_path: str, *, chat_id: Optional[str] = None,
                   reply_markup: Optional[Dict[str, Any]] = None) -> List[SendResult]:
        self._require_config()
        chats = [str(chat_id)] if chat_id else [str(c) for c in self.cfg.chat_ids]
        results: List[SendResult] = []
        if len(caption or "") > 1024:
            # A photo caption is capped at 1024 chars by the Bot API — silently
            # truncating would eat the stop/targets/defence lines.  The alert
            # text matters more than the picture, so send the full text instead.
            log.debug("caption is %d chars (>1024) — sending text instead of photo",
                      len(caption))
            for chat in chats:
                results.extend(self.send_text(caption, chat_id=chat, reply_markup=reply_markup))
            return results
        for chat in chats:
            self._throttle(chat)
            fields = {
                "chat_id": chat, "caption": caption,
                "disable_notification": 1 if self.cfg.disable_notification else 0,
            }
            if self.cfg.parse_mode in ("HTML", "MARKDOWNV2", "MARKDOWN"):
                fields["parse_mode"] = {"MARKDOWNV2": "MarkdownV2", "MARKDOWN": "Markdown"}.get(
                    self.cfg.parse_mode, "HTML")
            if reply_markup:
                fields["reply_markup"] = json.dumps(reply_markup)
            if self.cfg.message_thread_id:
                fields["message_thread_id"] = self.cfg.message_thread_id
            if self.dry_run:
                log.info("[dry-run] sendPhoto -> %s (%s)", chat, photo_path)
                results.append(SendResult(ok=True, chat_id=chat, method="sendPhoto", message_id=0))
                continue
            try:
                with open(photo_path, "rb") as fh:
                    blob = fh.read()
                resp = self._post("sendPhoto", fields,
                                  files={"photo": (f"{chat}.png", blob, "image/png")})
                results.append(SendResult(ok=True, chat_id=chat, method="sendPhoto",
                                          message_id=(resp.get("result") or {}).get("message_id", 0)))
            except Exception as exc:
                log.warning("sendPhoto failed for %s (%s) — sending text only", chat, exc)
                results.extend(SendResult(ok=r.ok, chat_id=chat, error=r.error or str(exc),
                                          method="sendMessage+photo-failed")
                               for r in self.send_text(caption, chat_id=chat,
                                                       reply_markup=reply_markup))
        return results

    def get_me(self) -> Dict[str, Any]:
        return self._post("getMe", {}).get("result", {})

    def get_chat(self, chat_id: str) -> Dict[str, Any]:
        """Describe a chat without sending anything (validates token + chat id + membership).

        ``getMe`` only proves the token is real; a mistyped ``TELEGRAM_CHAT_ID``
        — or a bot that was never started / added to the group — fails later at
        *send* time, when the alert is already built.  ``getChat`` surfaces that
        misconfiguration up front: it raises ``TelegramPermanentError`` (400)
        for an unknown chat and 403 when the bot cannot see it.
        """
        return self._post("getChat", {"chat_id": str(chat_id)}).get("result", {})

    def validate_chats(self) -> Dict[str, Any]:
        """``getMe`` + ``getChat`` per configured chat. Returns ``{chat_id: info}``."""
        self._require_config()
        me = self.get_me()
        chats: Dict[str, Any] = {}
        for chat in [str(c) for c in self.cfg.chat_ids]:
            chats[chat] = self.get_chat(chat)
        return {"bot": me, "chats": chats}

    def get_updates(self, offset: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        fields: Dict[str, Any] = {"limit": limit, "allowed_updates": json.dumps(["message", "channel_post"])}
        if offset:
            fields["offset"] = offset
        return self._post("getUpdates", fields).get("result", [])

    def discover_chats(self, timeout_sec: float = 12.0) -> List[Dict[str, Any]]:
        """Poll getUpdates for a moment and return any chats that messaged the bot."""
        found: Dict[str, Dict[str, Any]] = {}
        deadline = time.time() + timeout_sec
        offset = None
        while time.time() < deadline:
            try:
                for upd in self.get_updates(offset=offset):
                    offset = upd.get("update_id", 0) + 1
                    for key in ("message", "channel_post", "edited_message"):
                        msg = upd.get(key) or {}
                        chat = msg.get("chat") or {}
                        if chat.get("id"):
                            found[str(chat["id"])] = {
                                "chat_id": chat["id"], "type": chat.get("type"),
                                "title": chat.get("title") or chat.get("username")
                                         or f"{chat.get('first_name','')} {chat.get('last_name','')}".strip(),
                            }
            except Exception as exc:
                log.debug("getUpdates: %s", exc)
            if found:
                break
            time.sleep(1.0)
        return list(found.values())


class TelegramRateLimited(TelegramError):
    retry_after = 3.0
