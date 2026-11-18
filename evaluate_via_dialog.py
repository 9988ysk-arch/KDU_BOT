"""Run single-question sets through the DialogManager (fresh session per question)."""
import json, re, sys
from handbook_qa import HandbookQA, load_kb
from dialog_manager import DialogManager
qa = HandbookQA(load_kb("handbook_kb.json")); dm = DialogManager(qa)
def check(e, a):
    return re.search(e[3:], a, re.I) is not None if e.startswith("re:") else any(x.lower() in a.lower() for x in e.split("|"))
tot = ok = 0
for f in sys.argv[1:]:
    for i, x in enumerate(json.load(open(f))):
        sid = f"{f}-{i}"
        if x.get("prev"):
            dm.handle(sid, x["prev"])
        r = dm.handle(sid, x["q"])
        good = all(check(e, r["answer"]) for e in x["expect"])
        tot += 1; ok += good
        if not good:
            print("FAIL", f, "|", x["q"], "->", r["kind"], "|", r["answer"][:150])
print(f"via dialog manager: {ok}/{tot} = {100*ok/tot:.1f}%")
