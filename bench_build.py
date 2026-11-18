"""Build a ~500-question benchmark for the handbook QA engine.

Ground truth is parsed here, directly from the PDF, with code that is independent
of handbook_qa.py (fixed page numbers, simple row rules). Questions are produced
from several phrasing templates; ~15% get a realistic typo in a template word.
Hand-written questions (rules, general facts, out-of-scope) are appended from
bench_handwritten.json.

Usage: python bench_build.py Student-Handbook-2022.pdf  ->  bench500_questions.json
"""
import json
import os
import random
import re
import sys
from collections import defaultdict

import pdfplumber

BLIND = "--blind" in sys.argv
random.seed(777 if BLIND else 2026)
PDF = next((a for a in sys.argv[1:] if not a.startswith("--")), "Student-Handbook-2022.pdf")

# Course summary tables: page -> programmes (read manually from the handbook headings)
COURSE_PAGES = {17: ["CS", "SE"], 22: ["CS", "SE"], 28: ["CS"], 35: ["CS"], 41: ["SE"], 48: ["SE"]}
PROG_WORDS = {"CS": ["CS", "computer science"], "SE": ["SE", "software engineering"]}
CAT = {"C": "compulsory", "E": "elective", "M": "military"}


def clean(c):
    return re.sub(r"\s+", " ", (c or "").replace("\n", " ")).strip()


def typo(word):
    """Swap two inner letters (a common keyboard typo)."""
    if len(word) < 6:
        return word
    i = random.randint(1, len(word) - 3)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def maybe_typo(template):
    if random.random() > 0.15:
        return template
    words = [w for w in re.findall(r"[a-z]{6,}", template) if "{" not in w]
    if not words:
        return template
    w = random.choice(words)
    return template.replace(w, typo(w), 1)


Q = []


def add(cat, q, expect, src="template"):
    Q.append({"cat": cat, "q": q, "expect": expect, "src": src})


with pdfplumber.open(PDF) as pdf:
    pages = {i + 1: p for i, p in enumerate(pdf.pages)}

    # ------------------------------------------------------------ courses
    offerings = defaultdict(list)   # code -> [(prog, sem, cat, gpa, ngpa, mgpa, name)]
    sem_rows = defaultdict(list)    # (prog, sem) -> [code]
    sem_totals = {}                 # (prog, sem) -> (gpa, ngpa, mgpa)
    for pg, progs in COURSE_PAGES.items():
        sem = None
        for t in pages[pg].extract_tables():
            for r in t:
                r = [clean(c) for c in r]
                j = " ".join(r)
                m = re.search(r"Total for Semester\s*(\d)", j)
                if m:
                    nums = [c for c in r if re.fullmatch(r"\d+", c)]
                    for p in progs:
                        sem_totals[(p, int(m.group(1)))] = tuple(int(x) for x in nums[-3:])
                    continue
                m = re.match(r"\s*Semester\s*0?(\d)", j)
                if m and not re.search(r"[A-Z]{2,3}\d{4}", j):
                    sem = int(m.group(1))
                    continue
                if r and re.fullmatch(r"[A-Z]{2,3}\d{4}", r[0]):
                    name = re.sub(r"\s*\(.*?\)\s*$", "", r[2])
                    g, n, mg = (r[4:7] + ["", "", ""])[:3]
                    for p in progs:
                        offerings[r[0]].append((p, sem, CAT.get(r[3], ""), g, n, mg, name))
                        sem_rows[(p, sem)].append(r[0])

    title_codes = defaultdict(set)
    for code, offs in offerings.items():
        title_codes[offs[0][6].lower()].add(code)

    def credit_of(offs):
        vals = set()
        for (_, _, _, g, n, mg, _) in offs:
            for v, typ in ((g, "GPA"), (n, "NGPA"), (mg, "MGPA")):
                if re.fullmatch(r"\d+", v or ""):
                    vals.add((int(v), typ))
        return vals

    codes = sorted(offerings)
    random.shuffle(codes)
    credit_t = ["How many credits is {x}?", "credits for {x}", "What is the credit value of {x}?",
                "{x} credits", "how many credits does {x} carry", "credit rating of {x}"]
    name_t = ["What is {c}?", "What is the name of module {c}?", "{c} module name", "Which course has the code {c}?",
              "what does {c} stand for", "tell me the title of {c}"]
    code_t = ["What is the module code of {t}?", "course code for {t}", "What is the code of the {t} module?",
              "{t} module code"]
    sem_t = ["Which semester is {t} offered?", "In which semester do we study {t}?", "When is {t} taught?",
             "{t} is in which semester"]
    cat_t = ["Is {t} compulsory or elective?", "Is {t} an elective module?", "{t} compulsory?",
             "Is {t} optional?"]


    if BLIND:  # unseen wording, used once to check generalisation
        credit_t = ["tell me the credits of {x}", "{x} is worth how many credits?", "how much credit for {x}",
                    "credit count of {x}", "what credit does {x} have"]
        name_t = ["{c} is which subject?", "full name of {c}", "{c} means what module?", "give the module title for {c}"]
        code_t = ["what code is used for {t}?", "{t} has which module code?", "find the code for {t}"]
        sem_t = ["{t} comes in what semester?", "during which semester can I take {t}?", "semester of {t}?"]
        cat_t = ["is {t} mandatory?", "do I have to take {t} or is it elective?", "{t} elective or compulsory"]
    n_credit_code = n_credit_title = n_name = n_code = n_sem = n_cat = 0
    for code in codes:
        offs = offerings[code]
        name = offs[0][6]
        creds = credit_of(offs)
        unique_title = len(title_codes[name.lower()]) == 1 and len(name) > 6
        # credits by code
        if len(creds) == 1 and n_credit_code < 60:
            v, typ = next(iter(creds))
            add("module_credits", maybe_typo(random.choice(credit_t)).format(x=code), [f"{v} {typ}"])
            n_credit_code += 1
        # credits by title
        if len(creds) == 1 and unique_title and n_credit_title < 40:
            v, typ = next(iter(creds))
            add("module_credits", maybe_typo(random.choice(credit_t)).format(x=name), [f"{v} {typ}"])
            n_credit_title += 1
        # name from code
        if n_name < 40:
            words = [w for w in re.findall(r"[A-Za-z]{4,}", name)][:2] or [name]
            add("module_name", maybe_typo(random.choice(name_t)).format(c=code), words)
            n_name += 1
        # code from title
        if unique_title and n_code < 30:
            add("module_code", maybe_typo(random.choice(code_t)).format(t=name), [code])
            n_code += 1
        # semester
        sems = sorted({o[1] for o in offs if o[1]})
        if unique_title and n_sem < 30:
            add("module_semester", maybe_typo(random.choice(sem_t)).format(t=name), [f"Semester {s}" for s in sems])
            n_sem += 1
        # category (only when the same for every programme)
        cats = {o[2] for o in offs}
        if unique_title and len(cats) == 1 and n_cat < 25 and "military" not in cats and "" not in cats:
            add("module_category", maybe_typo(random.choice(cat_t)).format(t=name), [cats.pop()])
            n_cat += 1

    # semester module counts and credit totals
    for (p, s), lst in sorted(sem_rows.items()):
        if s <= 4 and p == "SE":
            continue  # levels 1-2 are common to CS and SE
        pw = random.choice(PROG_WORDS[p]) if s > 4 else ""
        suffix = f" for {pw}" if pw else ""
        add("semester_count", maybe_typo(random.choice(["How many modules are in semester {s}{x}?",
                                                         "number of modules in semester {s}{x}",
                                                         "How many courses do students take in semester {s}{x}?"])
                                         ).format(s=s, x=suffix), [f"{len(lst)} "])
        add("semester_list", maybe_typo(random.choice(["List the modules in semester {s}{x}",
                                                        "What modules are in semester {s}{x}?",
                                                        "semester {s} subjects{x}"])).format(s=s, x=suffix),
            random.sample(lst, min(3, len(lst))))
        tot = sem_totals.get((p, s))
        if tot:
            for typ, v in zip(("GPA", "NGPA", "MGPA"), tot):
                if typ == "MGPA" and random.random() < 0.5:
                    continue
                add("semester_credits", maybe_typo(random.choice(["How many {t} credits in semester {s}{x}?",
                                                                  "{t} credits for semester {s}{x}",
                                                                  "total {t} credits in semester {s}{x}"])
                                                   ).format(t=typ, s=s, x=suffix), [f"{v} {typ}"])

    # ------------------------------------------------------------ grades (Table 12, page 54)
    grade_rows = []
    for t in pages[54].extract_tables():
        for r in t[1:]:
            r = [clean(c) for c in r if clean(c)]
            m = re.match(r"(\d+)\s*[\u2013-]\s*(\d+)", r[0]) if r else None
            if m and len(r) >= 3:
                grade_rows.append((int(m.group(1)), int(m.group(2)), r[1], r[2]))
    mark_t = ["got {m} in the exam, which grade?", "grade for a final mark of {m}", "{m}% gives what grade?",
              "which letter grade is {m}"] if BLIND else ["What grade do I get for {m} marks?", "{m} marks grade", "I scored {m}, what is my grade?",
              "what grade is {m} marks", "if I get {m} marks what grade"]
    for m in random.sample(range(35, 101), 26):
        lo, hi, g, gpv = next(x for x in grade_rows if x[0] <= m <= x[1])
        add("grade_marks", maybe_typo(random.choice(mark_t)).format(m=m), [f"grade {g}", gpv])
    for lo, hi, g, gpv in grade_rows:
        add("grade_gpv", random.choice(["What is the GPV of {g}?", "GPV for grade {g}", "grade point value of {g}"]
                                       ).format(g=g), [gpv])
        add("grade_range", random.choice(["What marks are needed for grade {g}?", "marks range for grade {g}"]
                                         ).format(g=g), [str(lo)])

    # ------------------------------------------------------------ classes (Table 13, page 58)
    class_t = ["with an FGPA of {f} what is my result", "final result for FGPA {f}", "FGPA={f} which class honours"] if BLIND else ["What class do I get with FGPA {f}?", "FGPA {f} class", "my FGPA is {f}, which class?",
               "Which class is awarded for an FGPA of {f}?"]
    for f in [2.0, 2.15, 2.5, 2.99, 3.0, 3.1, 3.29, 3.3, 3.45, 3.55, 3.69, 3.7, 3.8, 3.95, 4.0]:
        exp = "First" if f >= 3.7 else "Upper" if f >= 3.3 else "Lower" if f >= 3.0 else "Pass"
        add("fgpa_class", random.choice(class_t).format(f=f"{f:.2f}" if random.random() < .5 else f), [exp])

    # ------------------------------------------------------------ heights (Table 2, page 11)
    for t in pages[11].extract_tables():
        hdr = [clean(c) for c in t[0]]
        for r in t[1:]:
            r = [clean(c) for c in r]
            for ci in range(1, len(hdr)):
                add("height", random.choice(["minimum height for {g} {s}", "What is the height requirement for a "
                                             "{g} applicant to the {s}?"]).format(g=r[0].lower(), s=hdr[ci].lower()),
                    [r[ci]])

    # ------------------------------------------------------------ staff (pages 60-86)
    honor = re.compile(r"^(Dr\.|Mr\.|Mrs\.|Ms\.|Miss|Maj\.|Eng\.)")
    desig = re.compile(r"(Lecturer|Instructor|Professor)")
    staff = {}
    for pg in range(60, 87):
        lines = (pages[pg].extract_text() or "").splitlines()
        cur = None
        for i, ln in enumerate(lines):
            ln = ln.strip()
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if honor.match(ln) and desig.search(nxt):
                nm = re.sub(r"\((?:Mrs?|Ms|Dr|Eng|Prof)\.?\)\s*", "", ln)       # "Dr. (Mrs.) PG Wijendra"
                nm = re.sub(r"\s*\(.*$", "", nm).strip()                            # drop "(Dean ...)"
                cur = (nm, re.sub(r"\s*[–(-]?\s*\(?Head.*$", "", nxt).strip())
            m = re.match(r"e-mail\s+(\S+@\S+)", ln)
            if m and cur and cur[0] not in staff:
                staff[cur[0]] = (cur[1], m.group(1))
    email_t = ["give me {n}'s e-mail", "e-mail for {n}", "mail address of {n}"] if BLIND else ["What is the email of {n}?", "{n} email address", "how can I email {n}", "contact email of {n}"]
    des_t = ["what post does {n} hold?", "job title of {n}", "{n} is a lecturer or instructor?"] if BLIND else ["What is the designation of {n}?", "What is {n}'s position?", "{n} designation"]
    names = list(staff)
    for n in names:
        d, e = staff[n]
        add("staff_email", random.choice(email_t).format(n=n), [e])
    for n in random.sample(names, 30):
        d, e = staff[n]
        core = re.sub(r"\s*\(.*\)", "", d).strip(" –-")
        add("staff_designation", random.choice(des_t).format(n=n), [core])

# ------------------------------------------------------------ hand-written questions
try:
    hw_file = next((p for p in ("bench_handwritten.json", "tests/bench_handwritten.json") if os.path.exists(p)),
                   "bench_handwritten.json")
    for item in ([] if BLIND else json.load(open(hw_file, encoding="utf-8"))):
        item["src"] = "handwritten"
        Q.append(item)
except FileNotFoundError:
    pass

if BLIND:  # keep a 150-question random sample
    Q = random.sample(Q, min(150, len(Q)))
json.dump(Q, open("blind150_questions.json" if BLIND else "bench500_questions.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
cats = defaultdict(int)
for q in Q:
    cats[q["cat"]] += 1
print(len(Q), "questions")
for c, n in sorted(cats.items()):
    print(f"  {c:<18}{n}")
