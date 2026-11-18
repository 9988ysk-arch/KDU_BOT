"""
make_benchmark.py - builds an 800+ question benchmark for the handbook QA system.

Gold answers are read straight from the PDF with code that is independent of
handbook_qa.py (raw pdfplumber table rows and raw page text), so the benchmark
checks the whole pipeline: extraction + question understanding + answer building.

Expectation syntax used by evaluate.py:
  "a|b"     either substring a or b must appear (case-insensitive)
  "re:..."  a regular expression must match
Usage:  python make_benchmark.py Student-Handbook-2022.pdf benchmark800.json
"""
import json
import random
import re
import sys

import pdfplumber

PDF = sys.argv[1] if len(sys.argv) > 1 else "Student-Handbook-2022.pdf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "benchmark800.json"
rng = random.Random(2026)

# Course tables: page -> programmes (read manually from the section headings)
COURSE_PAGES = {17: ["CS", "SE"], 22: ["CS", "SE"], 28: ["CS"], 35: ["CS"], 41: ["SE"], 48: ["SE"]}
FIRST_SEM = {17: 1, 22: 3, 28: 5, 35: 7, 41: 5, 48: 7}
PROG_NAME = {"CS": "computer science", "SE": "software engineering"}

pages_text = []
rows = []  # (code, title, cat, gpa, ngpa, mgpa, semester, program)
sem_totals = {}  # (prog, sem) -> (gpa, ngpa, mgpa)
grade_rows, class_rows = [], []
with pdfplumber.open(PDF) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        pages_text.append(page.extract_text() or "")
        if i in COURSE_PAGES:
            for t in page.extract_tables():
                sem = FIRST_SEM[i]
                for r in t:
                    r = [(c or "").replace("\n", " ").strip() for c in r]
                    if r and r[0].startswith("Total for Semester"):
                        nums = [c for c in r if re.fullmatch(r"\d+", c)]
                        for prog in COURSE_PAGES[i]:
                            sem_totals[(prog, sem)] = tuple(int(x) for x in nums[-3:])
                        sem += 1
                        continue
                    if r and re.fullmatch(r"[A-Z]{2,3}\d{4}", r[0]):
                        title = re.sub(r"\s*\(.*?\)", "", r[2]).strip()
                        for prog in COURSE_PAGES[i]:
                            rows.append((r[0], title, r[3], r[4], r[5], r[6], sem, prog))
        for t in page.extract_tables():
            head = " ".join((c or "") for c in t[0]).lower()
            if "grade" in head and "gpv" in head:
                for r in t[1:]:
                    v = [c for c in r if c]
                    if len(v) >= 2:
                        grade_rows.append(v)
            if "fgpa" in head and "result" in head:
                class_rows = [[c for c in r if c] for r in t[1:]]

bench = []


def add(cat, q, expect, src="template"):
    bench.append({"cat": cat, "q": q, "expect": expect, "src": src})


def typo(q):
    """Introduce one realistic typo (swap two inner letters) in a word of 6+ letters."""
    words = q.split()
    idx = [i for i, w in enumerate(words) if len(w) >= 6 and w.isalpha()]
    if not idx:
        return q
    i = rng.choice(idx)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


# ---------------------------------------------------------------- modules
by_code = {}
for code, title, cat, g, n, m, sem, prog in rows:
    by_code.setdefault(code, []).append((title, cat, g, n, m, sem, prog))
title_count = {}
for code, offs in by_code.items():
    title_count[offs[0][0].lower()] = title_count.get(offs[0][0].lower(), 0) + 1

CAT = {"C": "compulsory", "E": "elective", "M": "military"}
for code, offs in sorted(by_code.items()):
    title = offs[0][0]
    credit_res = []
    for _, _, g, n, m, *_ in offs:
        for val, typ in ((g, "GPA"), (n, "NGPA"), (m, "MGPA")):
            if re.fullmatch(r"\d+", val):
                credit_res.append(rf"\b{int(val)} {typ} credits")
    credit_re = "re:" + "|".join(sorted(set(credit_res))) if credit_res else None
    sems = sorted({f"Semester {o[5]}" for o in offs})
    cats = sorted({CAT.get(o[1], "") for o in offs} - {""})
    unique_title = title_count[title.lower()] == 1 and len(title) > 6
    templates = []
    if credit_re:
        templates += [("module_credits", f"How many credits is {code}?", [credit_re]),
                      ("module_credits", f"{code} eke credits keeyada", [credit_re])]
        if unique_title:
            templates += [("module_credits", f"How many credits does {title} have?", [credit_re]),
                          ("module_credits", f"credit value of the {title} module", [credit_re])]
    templates += [("module_name", f"What is the name of module {code}?", [title]),
                  ("module_name", f"{code} kiyanne mokakda", [title])]
    if unique_title:
        templates += [("module_code", f"What is the module code of {title}?", [code]),
                      ("module_semester", f"Which semester is {title} offered in?", ["|".join(sems)]),
                      ("module_category", f"Is {title} compulsory or elective?", ["|".join(cats)])]
    for cat, q, exp in rng.sample(templates, k=min(3, len(templates))):
        add(cat, typo(q) if rng.random() < 0.12 else q, exp)

# ---------------------------------------------------------------- semesters
for sem in range(1, 9):
    progs = ["CS", "SE"]
    lists = {p: sorted({c for c, _, _, _, _, _, s, pp in rows if s == sem and pp == p}) for p in progs}
    for p in progs:
        codes = lists[p]
        if not codes:
            continue
        pn = PROG_NAME[p]
        add("semester_count", f"How many modules are there in semester {sem} for {pn}?", [f"re:\\b{len(codes)}\\b"])
        two = rng.sample(codes, k=min(2, len(codes)))
        add("semester_list", f"List the modules in semester {sem} for {pn}", two)
        el = sorted({c for c, _, cat, _, _, _, s, pp in rows if s == sem and pp == p and cat == "E"})
        if el:
            add("semester_electives", f"How many elective modules in semester {sem} for {pn}?",
                [f"re:\\b{len(el)}\\b"])
            add("semester_electives", f"what are the electives in semester {sem} for {pn}", rng.sample(el, k=min(2, len(el))))
        tot = sem_totals.get((p, sem))
        if tot:
            add("semester_credits", f"How many GPA credits are in semester {sem} for {pn}?", [f"re:\\b{tot[0]} GPA"])
            add("semester_credits", f"semester {sem} {pn} MGPA credits", [f"re:\\b{tot[2]} MGPA"])
    if lists["CS"] == lists["SE"]:
        add("semester_count", f"how many subjects in semester {sem}", [f"re:\\b{len(lists['CS'])}\\b"])

# ---------------------------------------------------------------- grades / GPV / classes
grade_tab = []
for v in grade_rows:
    m = re.match(r"(\d+)\s*-\s*(\d+)", v[0].replace("–", "-"))
    if m:
        grade_tab.append((int(m.group(1)), int(m.group(2)), v[1], v[2] if len(v) > 2 else ""))
for mark in rng.sample(range(35, 101), 50):
    g = next(x for x in grade_tab if x[0] <= mark <= x[1])
    q = rng.choice([f"What grade do I get for {mark} marks?", f"I got {mark} marks, what is my grade?",
                    f"{mark} marks grade", f"if i score {mark} what grade"])
    add("grade_from_marks", q, [f"grade {g[2]} "])
for lo, hi, g, gpv in grade_tab:
    add("gpv", rng.choice([f"What is the GPV of {g}?", f"grade point value for {g} grade"]), [gpv])
    add("grade_range", f"What marks are needed for grade {g}?", [f"{lo}"])
cls = []
for v in class_rows:
    nums = [float(x) for x in re.findall(r"\d+\.\d+", v[0])]
    cls.append((nums[0], nums[1] if len(nums) > 1 else 9.0, v[1]))
for _ in range(35):
    f = round(rng.uniform(2.0, 4.0), 2)
    res = next(c[2] for c in cls if c[0] <= f < c[1] or (f >= c[0] and c[1] == 9.0))
    key = "Upper" if "Upper" in res else "Lower" if "Lower" in res else res.split()[0]
    add("class_from_fgpa", rng.choice([f"What class do I get with an FGPA of {f}?", f"FGPA {f} class",
                                       f"my fgpa is {f} which class"]), [key])

# ---------------------------------------------------------------- staff (raw text of section 7)
staff_text = "\n".join(pages_text[59:86])
lines = staff_text.splitlines()
honor = re.compile(r"^(Dr\.|Mr\.|Mrs\.|Ms\.|Miss|Maj\.|Eng\.)\s?(.+)")
seen = set()
dept_marks = [(k, re.search(r"Department of ([A-Za-z ]+)", l).group(1).strip())
              for k, l in enumerate(lines) if re.match(r"^7\.\d+\s+Department of", l.strip())]
for i, ln in enumerate(lines[:-1]):
    m = honor.match(ln.strip())
    if not m or not re.search(r"Lecturer|Instructor|Professor", lines[i + 1]):
        continue
    name = re.sub(r"\(.*", "", ln).strip()
    surname = re.sub(r"[^A-Za-z]", " ", name).split()[-1]
    if surname in seen or surname in ("Silva",) or len(surname) < 3 or surname in ("Dr", "Mr", "Mrs", "Ms", "Eng"):
        continue
    seen.add(surname)
    email = ""
    research = ""
    for j in range(i + 2, min(i + 25, len(lines))):
        if honor.match(lines[j].strip()) and j + 1 < len(lines) and re.search(r"Lecturer|Instructor", lines[j + 1]):
            break
        em = re.search(r"[\w.\-]+@kdu\.ac\.lk", lines[j])
        if em:
            email = em.group(0)
        if lines[j].startswith("Research Area"):
            same = re.sub(r"^Research Areas?\s*", "", lines[j]).strip()
            research = re.split(r",", same or lines[j + 1])[0].strip()
    honor_word = m.group(1)
    if email:
        add("staff_email", rng.choice([f"What is the email of {honor_word} {surname}?", f"{surname} email address",
                                       f"email of {surname}"]), [email])
    if research and len(research) > 4:
        add("staff_research", rng.choice([f"What are the research areas of {surname}?",
                                          f"{surname} research interests"]), [research[:28].strip()])
    desig = lines[i + 1].split("(")[0].replace("Head", "").strip(" –-")
    add("staff_designation", f"What is the designation of {honor_word} {surname}?", [desig.split()[0]])
    dept = next((d for k, d in reversed(dept_marks) if k <= i), None)
    if dept:
        add("staff_department", rng.choice([f"Which department does {honor_word} {surname} belong to?",
                                            f"{surname} works in which department"]), [dept])

# ---------------------------------------------------------------- departments / programmes (manual gold)
for q, e in [
    ("What degrees does the department of computer science offer?", ["Computer Science", "Software Engineering"]),
    ("degrees offered by the computer science department", ["Software Engineering"]),
    ("Which programmes are offered by the IT department?", ["Information Technology", "Information Systems"]),
    ("list degrees of department of information technology", ["Information Systems"]),
    ("What degree is offered by the computer engineering department?", ["Computer Engineering"]),
    ("Which department offers the software engineering degree?", ["Department of Computer Science"]),
    ("Which department offers the computer science degree?", ["Department of Computer Science"]),
    ("Which department offers the IT degree?", ["Department of Information Technology"]),
    ("Which department offers the information systems degree?", ["Department of Information Technology"]),
    ("Which department offers computer engineering?", ["Department of Computer Engineering"]),
    ("Does computational mathematics department offer a degree?", ["Mathematics"]),
    ("How many degrees does the computer science department offer?", ["2"]),
    ("how many departments are in the faculty of computing", ["4"]),
    ("list the departments", ["Computer Science", "Computational Mathematics"]),
    ("Who is the head of the department of computer science?", ["Sirisuriya"]),
    ("head of the information technology department", ["Kathriarachchi"]),
    ("Who heads computer engineering department?", ["Hettige"]),
    ("Head of department of computational mathematics", ["Kottegoda"]),
    ("How long is the computer engineering degree?", ["4"]),
    ("duration of information systems degree", ["4"]),
    ("How many years does the software engineering degree take for military students?", ["4½"]),
    ("A/L stream for computer engineering", ["Maths"]),
    ("A/L stream for information systems", ["Biology"]),
    ("A/L stream required for computer science", ["Maths"]),
    ("Which stream for software engineering?", ["Maths"]),
    ("Can commerce students apply for IT?", ["Commerce"]),
    ("What is the maximum duration for computer science?", ["8"]),
    ("How many degree programmes are offered by the faculty?", ["5"]),
    ("What degrees are offered by FOC?", ["Information Systems"]),
    ("What are the courses offered by the faculty of computing?", ["Computer Engineering"]),
]:
    add("department_programme", q, e, "manual")

# ---------------------------------------------------------------- people / appointments (manual gold)
for q, e in [
    ("Who is the dean?", ["Gunasekara"]), ("dean of faculty of computing email", ["asela@kdu.ac.lk"]),
    ("Who is the vice chancellor of KDU?", ["Milinda Peiris"]), ("chancellor of the university", ["Gerard Hector de Silva"]),
    ("Who is the registrar of KDU?", ["Kithsiri"]), ("deputy registrar name", ["Dissanayke"]),
    ("Who is the rector of the southern campus?", ["Edirisinghe"]),
    ("DVC academic", ["Dhammika"]), ("Who is the DVC for defence and administration?", ["Chandrasiri"]),
    ("Dean of graduate studies", ["Rajapakshe"]), ("dean of defence and strategic studies", ["Herath"]),
    ("foreign relations officer of KDU", ["Udapamunuwa"]), ("vice chancellor telephone", ["0112634274"]),
    ("registrar tel number", ["0710219248|0112635268"]), ("rector phone", ["0473624895"]),
    ("Who researches machine translation?", ["Hettige"]), ("Who works on image processing?", ["Samaraweera|Kulasekara"]),
    ("Who works on blockchain?", ["Upeksha"]),
    ("who researches data mining", ["Data Mining|Kumara|Priyadarshani|Vidanagama|Ilmini"]),
    ("How many lecturers in the computational mathematics department?", ["4"]),
]:
    add("people_manual", q, e, "manual")

# ---------------------------------------------------------------- rules / general (manual gold)
MANUAL_RULES = [
    ("What are the two assessment components?", ["Continuous Assessment"]),
    ("What is the weight of the written examination?", ["70%"]),
    ("weight of continuous assessment", ["30%"]),
    ("Can the weightage of CA change?", ["nature of the course"]),
    ("What does CA include?", ["tutorials"]),
    ("Are mid semester exams part of CA?", ["mid semester"]),
    ("minimum mark for CA and ES components", ["35%"]),
    ("What is the pass mark for a GPA course?", ["45%"]),
    ("What is the minimum attendance to sit for the exam?", ["80%"]),
    ("attendance needed on medical grounds", ["70%"]),
    ("Who considers attendance below 80% on medical grounds?", ["Faculty Board"]),
    ("How is SGPA calculated?", ["formula|Credit value|credit"]),
    ("What is SGPA?", ["Semester Grade Point Average"]),
    ("What is YGPA?", ["Year|Grade Point Average"]),
    ("What does FGPA mean?", ["Final Grade Point Average"]),
    ("What does MGPA mean?", ["Military"]),
    ("What is NGPA?", ["Non"]),
    ("Are NGPA credits counted in the GPA?", ["GPA credits|not|NGPA"]),
    ("Which credits are used to calculate GPA for civil students?", ["GPA credits"]),
    ("What grade is given for absence?", ["Ab"]),
    ("What does grade Ex mean?", ["Excused"]),
    ("What does grade Ie mean?", ["ES"]),
    ("What is grade Ne?", ["Not eligible"]),
    ("How to complete a course with grade Ie?", ["ES component"]),
    ("How do I clear grade Ia?", ["CA/PBCA"]),
    ("What should I do with grade Ib?", ["both ES and CA"]),
    ("Is CA mark carried over when re-sitting?", ["carried over"]),
    ("Is there a fee to re-sit a course?", ["charge"]),
    ("Who approves the re-sit charge?", ["BOM"]),
    ("Can I repeat a weak pass to improve?", ["upgrading|weak pass"]),
    ("What is the maximum grade after repeating?", ["C"]),
    ("What happens if I fail an NGPA course?", ["repeated"]),
    ("Who decides the grading of MGPA courses?", ["senate"]),
    ("When will supplementary examinations be held?", ["7th and 8th"]),
    ("Are supplementary exams held after semester 2?", ["not be held"]),
    ("What are the requirements to complete a semester?", ["C"]),
    ("How many D+ or C- grades are allowed in a semester?", ["one|1"]),
    ("When does a student get a warning?", ["2.0"]),
    ("What happens after a warning if SGPA stays low?", ["discontinu|warning"]),
    ("When is a student discontinued?", ["sixteen semesters"]),
    ("Can a student be discontinued for being absent?", ["two continuous semesters|absent"]),
    ("Can a medical board discontinue a student?", ["medical board"]),
    ("Can the maximum duration be extended?", ["medical"]),
    ("What is the maximum number of years to complete a degree?", ["8"]),
    ("What FGPA is needed for a first class?", ["3.70"]),
    ("FGPA for second upper", ["3.30"]),
    ("FGPA range for second lower", ["3.00"]),
    ("Minimum FGPA to pass the degree", ["2.00"]),
    ("Can I get first class with a failure grade?", ["failure"]),
    ("How many failure grades allowed for second class upper?", ["one|1"]),
    ("Within how many years must the requirements be completed for honours?", ["four"]),
    ("What are the criteria for awarding degrees?", ["GPA"]),
    ("Which trophies are awarded to graduates?", ["Best Graduant"]),
    ("Is there a trophy for software engineering?", ["Software Engineering"]),
    ("Who approves merit awards?", ["Board of Management"]),
    ("What is the minimum GPA for the whole degree?", ["2.00"]),
    ("When was KDU established?", ["1981"]),
    ("When was KDU elevated to university status?", ["1988"]),
    ("When was the Faculty of Computing established?", ["2015"]),
    ("Where is the Faculty of Computing?", ["Sooriyawewa"]),
    ("Is the faculty the first computing faculty in Sri Lanka?", ["first"]),
    ("How many students study in the faculty?", ["800"]),
    ("Is KDU a member of the Association of Commonwealth Universities?", ["Commonwealth"]),
    ("Can civil students study at KDU?", ["civil students"]),
    ("What is the mission of KDU?", ["learner-centered"]),
    ("What is the vision of the university?", ["tri-services"]),
    ("What is the faculty email?", ["hoditm@kdu.ac.lk"]),
    ("faculty phone number", ["2622995"]),
    ("university telephone", ["635268"]),
    ("Southern campus email", ["Southerncampus@kdu.ac.lk"]),
    ("southern campus telephone", ["2030800|3620812"]),
    ("What is the postal address?", ["Kandawala Road"]),
    ("faculty website", ["Faculty-of-Computing"]),
    ("university website", ["kdu.ac.lk"]),
    ("Which international guidelines are the degrees designed by?", ["ACM/IEEE"]),
    ("When do IT students choose between IT and IS?", ["third year"]),
    ("What is the age limit for military applicants?", ["22"]),
    ("Must military applicants be unmarried?", ["unmarried"]),
    ("What is the minimum weight for military students?", ["50 kg"]),
    ("Chest measurement requirement", ["81.25"]),
    ("height for female air force", ["162.5"]),
    ("height for male air force", ["167.6"]),
    ("height for female navy", ["160.0"]),
    ("height for male army", ["165.1"]),
    ("How many S passes at A/L?", ["three"]),
    ("What is the pass mark of the common general test?", ["30"]),
    ("English requirement for admission", ["Credit"]),
    ("Can technology stream students apply?", ["Technology"]),
    ("What does computer engineering degree involve?", ["hardware"]),
    ("What does the information systems degree cover?", ["Project Management|Enterprise Architecture"]),
    ("career opportunities for computer science graduates", ["software|research"]),
    ("career opportunities for software engineering graduates", ["software"]),
    ("What is the industrial training for SE?", ["SE4996|Industrial Training"]),
    ("Which language is taught in Fundamentals of Programming?", ["C++"]),
    ("Which language is used in Object Oriented Programming I?", ["Java"]),
    ("What is Logic Programming about?", ["Prolog"]),
    ("Computer Networks II curriculum", ["CCNA"]),
    ("What platform is used in Group Project in Hardware?", ["Arduino"]),
    ("What topics are in Information Security?", ["security|cryptograph"]),
    ("total GPA credits for computer science degree", ["128"]),
    ("total GPA credits for software engineering degree", ["127"]),
    ("total NGPA credits for the CS degree", ["15"]),
    ("total MGPA credits for SE", ["18"]),
]
for q, e in MANUAL_RULES:
    add("rules_general", q, e, "manual")
    if rng.random() < 0.25:
        add("rules_general", typo(q), e, "manual-typo")

# ---------------------------------------------------------------- out of scope (must say not found)
OOS = [
    "What is the hostel fee?", "How much is the tuition fee?", "semester fee amount", "Is there a gym?",
    "Is there a swimming pool on campus?", "What time does the library close?", "library opening hours",
    "Who is the president of Sri Lanka?", "What is the wifi password?", "When is the next intake?",
    "Are scholarships available?", "What is the bus route to campus?", "What is the canteen menu?",
    "Is there a parking fee?", "What is the dress code?", "What is the ragging policy?",
    "When is the convocation?", "Who won the cricket match?", "What is the weather today?",
    "How much does accommodation cost?", "Do you offer online degrees?", "What is the exam timetable?",
    "When does semester 1 start?", "How do I reset my LMS password?", "What is the price of a lunch?",
    "Can I bring my pet?", "Is there a medical centre for students?", "What sports clubs are there?",
    "How do I apply for a student loan?", "What is the student ID card fee?", "Is there a hostel for girls?",
    "What is the refund policy?", "How many buses go to Sooriyawewa?", "Who is the prime minister?",
    "What is the capital of France?", "What is a good laptop for programming?", "How do I learn Python?",
    "Is there a swimming team?", "What is the cafeteria phone number?", "When is the sports meet?",
]
for q in OOS:
    add("out_of_scope", q, ["couldn't find|could not find|not in the"], "manual")

json.dump(bench, open(OUT, "w"), indent=1, ensure_ascii=False)
from collections import Counter
print(len(bench), "questions")
print(Counter(b["cat"] for b in bench))
