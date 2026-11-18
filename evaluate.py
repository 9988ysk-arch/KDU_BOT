"""Evaluate the non-LLM handbook QA engine on a gold question set.

Usage:  python evaluate.py [Student-Handbook-2022.pdf] [questions.json] [--fails] [--session]

An answer counts as correct when every expected key fact appears in it:
  "a|b"        either substring a or b (case-insensitive)
  "re:<regex>" the regular expression must match
Items with a "prev" field are follow-up questions: "prev" is sent as the previous question.
--session    also re-runs the whole set with each question receiving the previous one as
             context, to check that follow-up handling never breaks standalone questions.
Per-category accuracy is printed when items carry "cat". Failures are written to
<questions.json>.failures.json.
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from handbook_qa import HandbookQA, build_kb, load_kb

args = [a for a in sys.argv[1:] if not a.startswith("--")]
only_fails = "--fails" in sys.argv
pdf = args[0] if len(args) > 0 else "Student-Handbook-2022.pdf"
gold_file = args[1] if len(args) > 1 else "gold_questions.json"

kb = load_kb("handbook_kb.json") or build_kb(pdf, "handbook_kb.json")
qa = HandbookQA(kb)
gold = json.loads(Path(gold_file).read_text(encoding="utf-8"))


def check(expect: str, answer: str) -> bool:
    if expect.startswith("re:"):
        return re.search(expect[3:], answer, re.I) is not None
    return any(alt.lower() in answer.lower() for alt in expect.split("|"))


def judge(item: dict, answer: str) -> bool:
    return all(check(e, answer) for e in item["expect"])


correct, times, failures = 0, [], []
per_cat = defaultdict(lambda: [0, 0])
for item in gold:
    t0 = time.perf_counter()
    res = qa.answer(item["q"], previous=item.get("prev"))
    times.append((time.perf_counter() - t0) * 1000)
    ans = res["answer"]
    ok = judge(item, ans)
    correct += ok
    cat = item.get("cat", "all")
    per_cat[cat][0] += ok
    per_cat[cat][1] += 1
    if not ok:
        failures.append({**item, "answer": ans, "kind": res["kind"]})
    if not only_fails or not ok:
        prev = f"  (after: {item['prev']})" if item.get("prev") else ""
        print(("PASS" if ok else "FAIL"), f"[{res['kind']}]", item["q"] + prev, "\n     ->", ans[:260])

n = len(gold)
Path(gold_file + ".failures.json").write_text(json.dumps(failures, indent=1, ensure_ascii=False), encoding="utf-8")
print()
if len(per_cat) > 1:
    for cat, (c, t) in per_cat.items():
        print(f"  {cat:<20} {c:>4}/{t:<4} = {100 * c / t:5.1f}%")
print(f"\nAccuracy: {correct}/{n} = {100 * correct / n:.1f}%")
print(f"Mean latency: {sum(times) / n:.1f} ms  |  max: {max(times):.1f} ms  (CPU only, no LLM)")

if "--session" in sys.argv:
    prev, ok2 = None, 0
    for item in gold:
        if item.get("prev"):
            prev = item["prev"]
        ans = qa.answer(item["q"], previous=prev)["answer"]
        good = judge(item, ans)
        ok2 += good
        if not good:
            print("SESSION-FAIL", item["q"], "| prev:", prev, "\n     ->", ans[:200])
        prev = item["q"]
    print(f"Session-mode accuracy (previous question passed as context): {ok2}/{n} = {100 * ok2 / n:.1f}%")
