"""Evaluate multi-turn conversations with the dialog manager.

Usage: python evaluate_dialog.py [Student-Handbook-2022.pdf] [dialog_tests.json] [--fails]
Each turn passes when every "expect" item appears (a|b alternatives, re:<regex>), no "not"
item appears, and, when "options": true, the reply offers clarification options.
Without dialog_manager.py (baseline) the single-turn engine is used with the previous question.
"""
import json
import re
import sys
from pathlib import Path

from handbook_qa import HandbookQA, build_kb, load_kb

args = [a for a in sys.argv[1:] if not a.startswith("--")]
pdf = args[0] if args else "Student-Handbook-2022.pdf"
tests = json.loads(Path(args[1] if len(args) > 1 else "dialog_tests.json").read_text(encoding="utf-8"))
only_fails = "--fails" in sys.argv

qa = HandbookQA(load_kb("handbook_kb.json") or build_kb(pdf, "handbook_kb.json"))
try:
    from dialog_manager import DialogManager
    dm = DialogManager(qa)
except ImportError:
    dm = None


def check(expect, text):
    if expect.startswith("re:"):
        return re.search(expect[3:], text, re.I) is not None
    return any(a.lower() in text.lower() for a in expect.split("|"))


turn_ok = turn_n = conv_ok = 0
for conv in tests:
    sid = conv["name"]
    prev = None
    all_ok = True
    for t in conv["turns"]:
        if dm:
            res = dm.handle(sid, t["user"])
        else:
            res = qa.answer(t["user"], previous=prev)
            prev = t["user"]
        text = res.get("answer", "")
        ok = all(check(e, text) for e in t.get("expect", []))
        ok = ok and not any(check(n, text) for n in t.get("not", []))
        if t.get("options"):
            ok = ok and bool(res.get("options"))
        if t.get("kind"):
            ok = ok and res.get("kind", "").startswith(t["kind"])
        turn_ok += ok
        turn_n += 1
        all_ok &= ok
        if not only_fails or not ok:
            opts = f"  options={res.get('options')}" if res.get("options") else ""
            print(("PASS" if ok else "FAIL"), f"[{sid}] {t['user']}\n     -> {text[:230]}{opts}")
    conv_ok += all_ok
print(f"\nTurns: {turn_ok}/{turn_n} = {100 * turn_ok / turn_n:.1f}%   Conversations fully correct: "
      f"{conv_ok}/{len(tests)}")
