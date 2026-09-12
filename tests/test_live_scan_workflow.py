"""`.github/workflows/live-scan.yml` — regressions that made a green run say nothing.

The hosted workflow is the zero-maintenance way to run the scanner, and it has
already failed in one specific shape: every run reported success, the chat stayed
empty, and the one line on the run page that said what the job had decided was
blank.  A runner is not available offline, so these tests parse the workflow
instead of executing it and pin exactly the parts whose failure is silent —
a step that cannot read its own outputs, a digest gated out on the days people
actually go looking, and embedded python whose syntax error would only surface on
the runner.
"""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github/workflows/live-scan.yml"

#: `${{ … }}` is substituted by the runner before the script is written out;
#: a bare word keeps the surrounding python parseable for the syntax check.
GH_EXPR = re.compile(r"\$\{\{.*?\}\}", re.S)


@pytest.fixture(scope="module")
def steps():
    doc = yaml.safe_load(WF.read_text(encoding="utf-8"))
    return doc["jobs"]["live-scan"]["steps"]


def _step(steps, name: str) -> dict:
    found = [s for s in steps if s.get("name") == name]
    assert found, f"no step named {name!r}; have: {[s.get('name') for s in steps]}"
    return found[0]


def _index(steps, name: str) -> int:
    return [s.get("name") for s in steps].index(name)


def _heredocs(text: str):
    """Every `python - <<'PY' … PY` body, de-indented to column 0."""
    out = []
    for body, indent in re.findall(r"python - <<'PY'[^\n]*\n(.*?)\n(\s*)PY\n", text, re.S):
        out.append("\n".join(ln[len(indent):] if ln.startswith(indent) else ln
                             for ln in body.split("\n")))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# the run page must say what the job decided
# ─────────────────────────────────────────────────────────────────────────────
def test_the_plan_step_does_not_read_its_own_outputs(steps):
    """A step's `run` script is expanded *before* it executes, so
    `${{ steps.plan.outputs.* }}` inside the step with `id: plan` is always
    empty.  The plan notice used to live there and every run annotated itself
    with `plan= · now  IST · window closes  IST · budget s` — the single line
    that says whether the job ran a session loop or one catch-up cycle, blank on
    exactly the runs where the user was asking why nothing arrived."""
    plan = _step(steps, "Plan this job (session budget)")
    assert plan.get("id") == "plan"
    assert "steps.plan.outputs" not in plan["run"], \
        "this step writes those outputs; it cannot also read them"
    assert "GITHUB_OUTPUT" in plan["run"]


def test_the_plan_is_announced_by_a_later_step(steps):
    announce = _step(steps, "Announce the plan")
    assert _index(steps, "Announce the plan") > _index(steps, "Plan this job (session budget)")
    run = announce["run"]
    for output in ("plan", "ist_now", "window_end", "budget"):
        assert f"steps.plan.outputs.{output}" in run, output
    assert "::notice::" in run


def test_a_non_trading_day_is_flagged_on_the_run_page(steps):
    """A weekend dispatch can only re-read the last completed bar.  That has to
    be visible without opening the log, otherwise two green Saturday runs look
    like two attempts the scanner silently failed."""
    plan = _step(steps, "Plan this job (session budget)")
    assert "trading_day=" in plan["run"]
    announce = _step(steps, "Announce the plan")
    assert "steps.plan.outputs.trading_day" in announce["run"]
    assert "::warning::" in announce["run"]
    assert "not an NSE session" in announce["run"]


def test_the_summary_names_the_session_that_was_evaluated(steps):
    """`alerts.recent_bars: 1` means one cycle looks at one bar, so "which bar"
    is the first thing a silent run has to answer."""
    run = _step(steps, "Summarise scan result")["run"]
    assert "bar_session" in run and "session_now" in run and "market_state" in run
    # …but only an exchange-tracked feed can be behind today's session
    assert 'state in ("open", "closed")' in run
    for prefix in ("MARKET CLOSED", "NO FRESH SESSION", "FEED BEHIND"):
        assert prefix in run, f"the {prefix} note would never reach the step summary"


# ─────────────────────────────────────────────────────────────────────────────
# the digest is the "silence ≠ no signals" channel — it must not be gated out
# ─────────────────────────────────────────────────────────────────────────────
def test_the_digest_still_posts_on_a_non_trading_day(steps):
    """The 16:40 gate exists so a morning kick cannot consume the day's one
    digest slot.  On a Saturday there is no slot to protect and nothing left to
    wait for — and a manual weekend dispatch is precisely when the chat is
    silent *and* unexplained."""
    run = _step(steps, "Post end-of-day digest")["run"]
    assert re.search(r"if now\.weekday\(\) < 5 and \(now\.hour, now\.minute\) < \(16, 40\)", run), \
        "the early exit must apply to trading days only"


def test_the_digest_reports_the_session_it_evaluated(steps):
    run = _step(steps, "Post end-of-day digest")["run"]
    assert "bar_session" in run
    assert "market closed" in run
    # alerts are counted by the bar they fired on, not only by when the row was
    # written — on a weekend nothing was written "today"
    assert "bar_date" in run


def test_the_digest_stays_idempotent_through_the_ledger(steps):
    """A weekend dispatch must not turn into a daily spam channel: one digest
    per date, guarded by the same ledger that dedupes alerts."""
    run = _step(steps, "Post end-of-day digest")["run"]
    assert "__digest__" in run
    assert "record_alert" in run
    assert "digest already posted today" in run


# ─────────────────────────────────────────────────────────────────────────────
# the runner is the only place a syntax error would otherwise surface
# ─────────────────────────────────────────────────────────────────────────────
def test_every_embedded_python_block_parses():
    text = WF.read_text(encoding="utf-8")
    blocks = _heredocs(text)
    assert len(blocks) >= 5, "heredoc extraction drifted from the workflow"
    for i, src in enumerate(blocks, 1):
        try:
            ast.parse(GH_EXPR.sub("GH_EXPR", src))
        except SyntaxError as exc:                        # pragma: no cover - the point of the test
            pytest.fail(f"embedded python block {i} does not parse: line {exc.lineno}: {exc.msg}")


def test_every_run_block_is_valid_bash():
    bash = shutil.which("bash")
    if not bash:                                          # pragma: no cover
        pytest.skip("bash unavailable")
    doc = yaml.safe_load(WF.read_text(encoding="utf-8"))
    tmp = Path("/tmp") / "precision-tap-workflow-step.sh"
    for step in doc["jobs"]["live-scan"]["steps"]:
        if not step.get("run"):
            continue
        tmp.write_text(step["run"], encoding="utf-8")
        proc = subprocess.run([bash, "-n", str(tmp)], capture_output=True, text=True)
        assert proc.returncode == 0, f"step {step.get('name')!r}: {proc.stderr}"


def test_the_scan_step_still_owns_the_cadence(steps):
    """Cron only kicks; the job runs the loop.  If this ever regresses to
    `scan --once` per cron slot the intraday tap stream disappears again while
    every run still reports success."""
    run = _step(steps, "Run live scanner (yfinance → Telegram)")["run"]
    assert "steps.plan.outputs.plan" in run
    assert "run --duration" in run
    assert "steps.plan.outputs.budget" in run
    # …and a cycle that found signals but delivered none must still fail the job
    assert "exit \"$rc\"" in run
