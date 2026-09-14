"""Feedback helpers for the SGC 2026 hands-on notebooks (`aircheck` module).

No score, no ranking, no leaderboard. This module exists to do three things:

  1. let you commit an answer before you see the real one,
  2. tell you immediately whether you're right, without ever telling you off,
  3. hand you the "why this matters" once you've committed — right or wrong.

Zero dependencies beyond the standard library and IPython (which you already have
if you are reading this in a notebook). Feedback is rendered as Markdown via
`display()`, so it works in JupyterLab, VS Code, Colab and nbviewer, survives being
saved, and needs no JavaScript, no widgets, and no network.

    import aircheck

    aircheck.predict("b1_middle", 400)   # commit a guess first
    aircheck.check("b1_middle", answer)  # grade it, then explain why it matters
    aircheck.hint("b1_middle")           # nudge, no penalty
    aircheck.reveal("b1_middle")         # just tell me, no penalty either
    aircheck.mcq("b1_label", "c")        # concept check with per-option feedback
    aircheck.progress()                  # what you've worked through so far

The grading logic here is meant to be read. Only the answers themselves live in
`answers.key`, so that scrolling past a cell doesn't spoil the block you haven't
reached yet.
"""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_STATE = {"committed": {}, "answered": set(), "hints": {}, "seen": set()}

try:
    from IPython.display import Markdown, display

    def _show(text):
        display(Markdown(text))

except ImportError:  # running outside a notebook, e.g. under pytest
    def _show(text):
        print(text.replace("**", "").replace("> ", "   "))


def _load_key():
    path = _HERE / "answers.key"
    if not path.exists():
        raise FileNotFoundError(
            "answers.key is missing next to aircheck.py. Regenerate it with: python make_answers.py")
    return json.loads(zlib.decompress(base64.b64decode(path.read_bytes())))


_KEY = _load_key()
_Q = _KEY["questions"]
_BLOCKS = _KEY["blocks"]


# ----------------------------------------------------------------- normalising
class BlankNotFilled(Exception):
    """Raised when a `...` placeholder reaches the checker."""


def _reject_placeholder(qid, value):
    if value is Ellipsis:
        raise BlankNotFilled(
            f"'{qid}' still has a `...` in it — fill in the blank above first. "
            "If you'd rather move on, run aircheck.reveal('" + qid + "').")


def _scalar(value):
    """Unwrap the many things a notebook hands you that are almost a number."""
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(value, "iloc") and len(value) == 1:
        return _scalar(value.iloc[0])
    return value


def _as_set(value):
    if isinstance(value, str):
        value = [value]
    return {str(_scalar(v)).strip().upper() for v in value}


def _fmt(question, answer):
    if question.get("show"):
        return question["show"]
    kind = question["kind"]
    if kind == "set":
        return ", ".join(f"`{a}`" for a in sorted(answer))
    if kind == "int":
        return f"**{answer:,}**"
    if kind == "num":
        return f"**{answer:,.4g}**"
    return f"**{answer}**"


# -------------------------------------------------------------------- grading
def _grade(question, value):
    """Return (passed, note). The note is never a reprimand."""
    kind, answer = question["kind"], question["a"]

    if kind in ("int", "num"):
        try:
            got = float(_scalar(value))
        except (TypeError, ValueError):
            return False, f"that came through as a `{type(_scalar(value)).__name__}` rather than a number"
        if kind == "int":
            if int(round(got)) == int(answer):
                return True, ""
        else:
            if abs(got - answer) <= question.get("rel", 0.05) * abs(answer):
                return True, ""
        ratio = got / answer if answer else float("inf")
        if 0.5 < ratio < 2:
            return False, f"you have {got:,.4g}, which is the right ballpark but not the number"
        if 900 < ratio < 1100 or 0.0009 < ratio < 0.0011:
            return False, f"you have {got:,.4g} — that looks like a units slip (MB vs GB?)"
        if 90 < ratio < 110 or 0.009 < ratio < 0.011:
            return False, f"you have {got:,.4g} — off by about 100, so maybe a percentage vs a fraction"
        return False, f"you have {got:,.4g}"

    if kind == "set":
        got, want = _as_set(value), _as_set(answer)
        if got == want:
            return True, ""
        missing, extra = want - got, got - want
        bits = []
        if missing:
            bits.append(f"{len(missing)} still to find")
        if extra:
            bits.append(f"{', '.join(sorted(extra))} shouldn't be in there")
        return False, f"you have {len(got)} — " + ", and ".join(bits)

    if kind == "str":
        if str(_scalar(value)).strip().casefold() == str(answer).strip().casefold():
            return True, ""
        return False, f"you have `{_scalar(value)}`"

    raise ValueError(f"unknown question kind {kind!r}")


# ----------------------------------------------------------------- public API
def predict(qid, value):
    """Commit a guess before you compute the real answer.

    Being wrong here is not just tolerated, it is the mechanism. Committing to a
    number and then finding out you were off is what makes the real number stick;
    reading it passively does almost nothing.
    """
    if qid not in _Q:
        raise KeyError(f"no such question: {qid}")
    _reject_placeholder(qid, value)
    _STATE["committed"][qid] = _scalar(value)
    _touch(qid)
    _show(f"🔮 **Locked in: {_scalar(value)}** — now go and find out.")


def check(qid, value):
    """Check an answer, then explain why it matters."""
    if qid not in _Q:
        raise KeyError(
            f"no such question: {qid}. Known: {', '.join(sorted(_Q))}")
    _reject_placeholder(qid, value)
    question = _Q[qid]
    passed, note = _grade(question, value)
    _touch(qid)

    if not passed:
        lines = [f"🔸 **Not quite** — {note}." if note else "🔸 **Not quite.**"]
        remaining = len(question["hints"]) - _STATE["hints"].get(qid, 0)
        if remaining > 0:
            lines.append(
                f"\nHave another go, or run `aircheck.hint(\"{qid}\")` for a nudge.")
        else:
            lines.append(f"\nRun `aircheck.reveal(\"{qid}\")` and keep moving — "
                         "there's no penalty and the next part doesn't depend on it.")
        _show("\n".join(lines))
        return

    _STATE["answered"].add(qid)
    lines = ["✅ **That's it.**"]
    guess = _STATE["committed"].get(qid)
    if guess is not None:
        lines[0] += f"  *(you predicted {guess})*"
    lines.append("")
    lines.append(f"> {question['why']}")
    _show("\n".join(lines))


def mcq(qid, choice):
    """Answer a concept check. Every option gets its own feedback, right or wrong."""
    if qid not in _Q or "options" not in _Q[qid]:
        raise KeyError(f"no such concept check: {qid}")
    _reject_placeholder(qid, choice)
    question = _Q[qid]
    key = str(_scalar(choice)).strip().lower()
    _touch(qid)

    if key not in question["options"]:
        valid = ", ".join(f"`{k}`" for k in sorted(question["options"]))
        _show(f"🔸 `{key}` isn't one of the options. Pick one of {valid}.")
        return

    option = question["options"][key]
    if option["correct"]:
        _STATE["answered"].add(qid)
        _show(f"✅ **Yes.** {option['feedback']}\n\n> {question['why']}")
    else:
        _show(f"🔸 **Not this one.** {option['feedback']}\n\nTry another.")


def hint(qid):
    """Get the next nudge. Costs nothing — there is no score here."""
    if qid not in _Q:
        raise KeyError(f"no such question: {qid}")
    _touch(qid)
    hints = _Q[qid]["hints"]
    used = _STATE["hints"].get(qid, 0)
    if used >= len(hints):
        _show(
            f"💡 That's all the hints. `aircheck.reveal(\"{qid}\")` will just tell you.")
        return
    _STATE["hints"][qid] = used + 1
    _show(f"💡 **Hint {used + 1} of {len(hints)}** — {hints[used]}")


def reveal(qid):
    """Show the answer and why it matters. No penalty; getting unstuck is the point."""
    if qid not in _Q:
        raise KeyError(f"no such question: {qid}")
    question = _Q[qid]
    _touch(qid)
    _STATE["answered"].add(qid)
    if "options" in question:
        correct = next(
            k for k, v in question["options"].items() if v["correct"])
        answer = f"**{correct}** — {question['options'][correct]['feedback']}"
    else:
        answer = _fmt(question, question["a"])
    guess = _STATE["committed"].get(qid)
    lines = [f"🔓 **Answer:** {answer}"]
    if guess is not None:
        lines[0] += f"  *(you predicted {guess})*"
    lines += ["", f"> {question['why']}"]
    _show("\n".join(lines))


def explain(qid):
    """Re-read the 'why it matters' for something you've already done."""
    if qid not in _Q:
        raise KeyError(f"no such question: {qid}")
    _show(f"> {_Q[qid]['why']}")


def _touch(qid):
    _STATE["seen"].add(qid)


def progress(notebook=None):
    """What you've worked through. Not a score — just a place to pick up from.

    Only the blocks of the notebook you are in are listed: by default that is every
    notebook containing a question you have touched in this kernel (predicted, checked,
    hinted or revealed). Pass `notebook=1` or `notebook=2` to be explicit.
    """
    if notebook is None:
        touched = {b["notebook"] for b in _BLOCKS if set(b["qids"]) & _STATE["seen"]}
    else:
        touched = {int(notebook)}
    blocks = [b for b in _BLOCKS if not touched or b["notebook"] in touched]
    lines = ["### Where you are", ""]
    for block in blocks:
        title, qids = block["title"], block["qids"]
        done = sum(1 for q in qids if q in _STATE["answered"])
        if done == len(qids):
            mark, detail = "✅", "done"
        elif done:
            mark, detail = "◐", f"{done} of {len(qids)}"
        else:
            mark, detail = "○", "not started"
        lines.append(f"- {mark} **{title}** — {detail}")
    lines += ["", "*Finishing everything is optional. Finishing a block is a good place to stop.*"]
    _show("\n".join(lines))


def checkpoint(message="Caught up — everything below will work whether or not you solved that one."):
    """Print the reassurance that goes with a checkpoint cell."""
    _show(f"🧭 {message}")
