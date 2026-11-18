"""Single-message small-talk intent test: python evaluate_smalltalk.py smalltalk_tests.json [--fails]"""
import json, re, sys
from handbook_qa import HandbookQA, load_kb
from dialog_manager import DialogManager
qa = HandbookQA(load_kb("handbook_kb.json")); dm = DialogManager(qa)
tests = json.load(open(sys.argv[1], encoding="utf-8"))
ok = 0
for i, t in enumerate(tests):
    r = dm.handle(f"st{i}", t["u"])
    good = r["kind"] in ["small_talk_" + i for i in t["intent"].split("|")]
    if t.get("reply"):
        good = good and any(a.lower() in r["answer"].lower() for a in t["reply"].split("|"))
    ok += good
    if not good and "--fails" in sys.argv:
        print("FAIL", repr(t["u"]), "expected", t["intent"], "got", r["kind"], "|", r["answer"][:90])
print(f"small talk intents: {ok}/{len(tests)} = {100*ok/len(tests):.1f}%")
