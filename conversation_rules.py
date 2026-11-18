"""
conversation_rules.py - handbook-grounded answers for questions that keyword retrieval handles badly (v10).

Retrieval finds *a* sentence; these questions need the right rule stated plainly, so they are answered from
the structured knowledge base (grade table, class table, contacts) and the exact rule text of Sections 4-6:

  passing grade / pass mark ............ C (45%) overall and at least 35% in every component (4.1)
  one grade letter ("what if I get D+") what the grade means, GPV, pass / weak pass / fail, what to do next
  a grade that does not exist ("D") .... says so, and explains the grades around it
  weak passes .......................... D+ / C-, one per semester, repeat up to C, effect on classes
  repeat limits ("how many times") ..... no fixed number; the 16-semester / 8-year limit applies
  Pass-class requirements .............. FGPA band + degree-award criteria (6.1.1)
  prerequisites / coordinators ......... not listed in the handbook (graceful "not covered" answers)
  difficulty ("which is the hardest") .. not rated; credits are shown instead
  head of SE / IS ...................... no such department: names the department that offers the degree

NOT_COVERED lists topics the handbook never mentions (hostel, library, scholarships, timetables ...). They get
an honest "the handbook doesn't cover this" with the faculty contacts printed in the handbook, instead of an
unrelated retrieved sentence.

Everything here is rule-based (regular expressions + the knowledge base). No LLM.
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------------------------- grade words
# Grade letters that appear in Table 12, plus letters students commonly assume exist (D, E, F).
GRADE_TOKENS = ["a+", "a-", "a", "b+", "b-", "b", "c+", "c-", "c", "d+", "d", "e", "f", "ie", "ia", "ib", "ne",
                "ab", "ex"]
_GRADE_ALT = r"(?:a\+|a-|b\+|b-|c\+|c-|d\+|ie|ia|ib|ne|ab|ex|a|b|c|d|e|f)"
# "get a D+", "got an A-", "grade C-", "with a D", "an Ie", "'B+'"
GRADE_CONTEXT = re.compile(
    r"\b(?:get(?:ting)?|got|gets|receive[ds]?|scored?|obtain(?:ed)?|have|has|had|with|is|for|of|"
    r"grade|an?)\s+(?:an?\s+)?(?:grade\s+)?[\"'‘“]?(" + _GRADE_ALT + r")[\"'’”]?(?=[\s?.!,]|$)", re.I)
PASSING = {"a+", "a", "a-", "b+", "b", "b-", "c+", "c"}
WEAK = {"c-", "d+"}
FAIL_LETTERS = {"ie", "ia", "ib", "ne", "ab"}
NOT_IN_TABLE = {"d", "e", "f"}

CLASS_NAMES = {
    "first": "First Class",
    "upper": "Second Class (Upper Division)",
    "lower": "Second Class (Lower Division)",
    "pass": "Pass",
}

# ---------------------------------------------------------------------------------------------- not covered
# (topic label, pattern, options that ARE in the handbook). Every word here was checked: it does not
# appear in the Student Handbook 2022 (or only in an unrelated sense, e.g. "sports" as a career area).
NOT_COVERED: list[tuple[str, str, list[str]]] = [
    ("hostels or accommodation", r"\b(hostels?|accommodation|boarding|dorm(itor(y|ies))?|lodging|annex)\b",
     ["What are the entry requirements?", "How long is the degree?"]),
    ("the library", r"\blibrar(y|ies)\b|\bborrow(ing)? books?\b", ["What is the attendance requirement?"]),
    ("the canteen or food", r"\b(canteen|cafeteria|food|lunch|meals?)\b", []),
    ("scholarships or bursaries", r"\b(scholarships?|bursar(y|ies)|mahapola|financial aid|student loans?)\b",
     ["What awards and trophies are there?"]),
    ("a dress code", r"\b(dress code|what (to|should i) wear|uniform)\b", []),
    ("dates and timetables (semester start, exam dates, results release)",
     r"\b(timetable|time table|academic calendar|calendar)\b|\bwhen (does|do|will|is|are)\b.*\b(semester|"
     r"term|classes|lectures|exams?|results?|registration|orientation)\b.*\b(start|begin|open|end|finish|"
     r"released?|out|published|held|due|over)\b|\b(exam|semester|result) (dates?|schedule)\b",
     ["When are supplementary exams held?", "How many semesters are there?"]),
    ("parking or transport", r"\b(parking|park my|bus(es)?|shuttle|transport(ation)?)\b", []),
    ("Wi-Fi, LMS logins or passwords", r"\b(wi-?fi|internet access|lms (login|password|account)|password|"
                                       r"log ?in to)\b", []),
    ("clubs, societies or sports", r"\b(clubs?|societ(y|ies)|sports?|gym|swimming|cricket team)\b", []),
    ("counselling services", r"\b(counsel+(ing|or)|mental health|psychologist|therapist)\b",
     ["What happens if I miss an exam due to illness?"]),
    ("transfers between universities", r"\btransfer\w* (to|from) (another|a different|other) (university|"
                                       r"campus|faculty)\b", []),
    ("plagiarism rules", r"\bplagiari\w+\b|\bturnitin\b", []),
    ("internship placements", r"\binternships?\b|\bplacements?\b", ["Tell me about industrial training"]),
]
_NOT_COVERED = [(label, re.compile(p, re.I), opts) for label, p, opts in NOT_COVERED]
# Words that show the question is really about something the handbook covers ("exam timetable for
# supplementary exams" is still answered by the supplementary-exam rule).
COVERED_OVERRIDE = re.compile(r"\b(supplementary|re-?sit|repeat|industrial training|attendance|medical)\b", re.I)


class ConversationRules:
    """Rule answers built from the knowledge base. All methods return None when the rule does not apply."""

    def __init__(self, qa):
        self.qa = qa
        kb = qa.kb
        self.grade_rows: dict[str, list[dict[str, Any]]] = {}
        for r in kb.get("grades", {}).get("rows", []):
            self.grade_rows.setdefault(r["grade"].lower(), []).append(r)
        self.grade_page = kb.get("grades", {}).get("page")
        self.class_rows = {r["result"]: r for r in kb.get("classes", {}).get("rows", [])}
        self.class_page = kb.get("classes", {}).get("page")
        c = kb.get("contacts", {})
        self.contact = (f"the Faculty of Computing ({c.get('faculty_email', 'hoditm@kdu.ac.lk')}"
                        + (f", Dean's office {c['dean_phone']}" if c.get("dean_phone") else "")
                        + (f", website {c['faculty_web']}" if c.get("faculty_web") else "") + ")")
        self.max_years = next(iter(kb.get("max_duration", {}).values()), "8")

    # ------------------------------------------------------------------ helpers
    def page_of(self, needle: str) -> int | None:
        """Handbook page of the passage that contains `needle` (case-insensitive)."""
        n = needle.lower()
        for p in self.qa.kb.get("passages", []):
            if n in p["text"].lower():
                return p.get("page")
        return None

    def _cite(self, *needles: str) -> str:
        pages = sorted({p for p in (self.page_of(n) for n in needles) if p})
        return f" (Student Handbook, p. {', '.join(map(str, pages))})" if pages else ""

    @staticmethod
    def find_grade(q: str) -> str | None:
        """The grade letter a question is about, or None. "A" alone needs 'an A' / 'grade A' / upper case."""
        for m in GRADE_CONTEXT.finditer(q):
            g = m.group(1)
            lead = q[max(0, m.start(1) - 9):m.start(1)].lower()
            if g.lower() == "a" and not (g == "A" or re.search(r"(an|grade)\s+[\"'‘“]?$", lead)):
                continue
            if g.lower() in ("e", "f", "b", "c", "d") and g.islower() and not re.search(
                    r"(get|got|grade|an?|with|receive\w*|scored?|obtain\w*)\s+[\"'‘“]?$", lead):
                continue
            return g.lower()
        m = re.search(r"(?:^|\bis\s+|\bwhat\s+(?:is|does)\s+(?:grade\s+)?)[\"'‘“]?(a\+|a-|b\+|b-|c\+|c-|d\+|ie|ia|ib|"
                      r"ne|ab|ex)[\"'’”]?(?=[\s?.!,]|$)", q, re.I)
        return m.group(1).lower() if m else None

    @staticmethod
    def label(g: str) -> str:
        return {"ie": "Ie", "ia": "Ia", "ib": "Ib", "ne": "Ne", "ab": "Ab", "ex": "Ex"}.get(g, g.upper())

    # ------------------------------------------------------------------ grade answers
    def grade_reply(self, g: str, situation: str = "") -> str:
        G = self.label(g)
        rows = self.grade_rows.get(g, [])
        if g in NOT_IN_TABLE:
            return (f"There is no grade \"{G}\" in the KDU grading scheme. The lowest grades that still count as a "
                    "(weak) pass are D+ (35-39 marks, GPV 1.30) and C- (40-44, GPV 1.70). Below that, a course is "
                    "a failure graded by which component was under 35%: Ie (end-semester exam), Ia (continuous "
                    "assessment) or Ib (both), all with GPV 0.00. A failed GPA course unit must be repeated to "
                    "obtain a pass grade." + self._cite("Details of Grades", "shall be repeated to obtain a pass"))
        if not rows:
            return ""
        r = rows[0]
        marks = " / ".join(x["marks"] for x in rows)
        band = f"{marks} marks" if re.match(r"\d", marks) else marks
        gpv = f", GPV {r['gpv']}" if r.get("gpv") else ""
        if g in PASSING:
            extra = " That is the minimum pass grade." if g == "c" else ""
            return (f"{G} is a pass ({band}{gpv}).{extra} A course is passed with C (45%) or better, provided "
                    "every component (CA and end-semester exam) is at least 35%." + self._cite("total pass mark"))
        if g in WEAK:
            return (f"{G} ({band}{gpv}) is a \"weak pass\". The rules: you may have not more than one D+ or C- per "
                    "semester to complete that semester; you may repeat the course to upgrade it, but only up to a "
                    "maximum grade of C; and a weak pass carried over at the end rules out First Class and Second "
                    "Class (Upper Division)."
                    + self._cite("not more than one", "up to a maximum of grade", "not have carried over any weak"))
        if g == "ie":
            return (f"Ie (ES <35) means the end-semester exam mark was below 35% (GPV 0.00) - a failure. You complete the "
                    "course by re-sitting the end-semester (ES) exam; the CA mark you already earned is carried "
                    "over." + self._cite("grade “Ie”", "carried over when re-sitting"))
        if g == "ia":
            return ("Ia (CA < 35 / PBCA <35%) means the continuous assessment (CA or PBCA) mark was below 35% "
                    "(GPV 0.00) - a failure. "
                    "You complete the course by re-sitting the CA/PBCA component." + self._cite("grade “Ia”"))
        if g == "ib":
            return ("Ib (Both ES & CA < 35) means both the end-semester exam (ES) and the CA were below 35% "
                    "(GPV 0.00) - a failure. You "
                    "complete the course by re-sitting both the ES and the CA/PBCA components." + self._cite("grade “Ib”"))
        if g == "ab":
            return ("Ab means absent for the course unit (GPV 0.00). If the absence had a valid reason (e.g. "
                    "medical) and the faculty grants an Excuse, the grade becomes Ex instead, and the re-sit counts "
                    "as a normal first attempt." + self._cite("Details of Grades", "same basis as a normal first"))
        if g == "ne":
            return ("Ne means not eligible to sit the exam (GPV 0.00). The eligibility rule for a first-time "
                    "candidate is an attendance record of at least 80% (70% may be considered on valid medical or "
                    "other grounds by the Faculty Board)." + self._cite("attendance record of not less than 80"))
        if g == "ex":
            return ("Ex means excused on a valid reason. Re-sitting a course unit for which an Excuse has been "
                    "granted is on the same basis as a normal first-attempt candidate. Note that an Ex in a "
                    "semester means that semester is not yet complete." + self._cite("same basis as a normal first"))
        return f"Grade {G}: {band}{gpv}."

    def repeat_for_grade(self, g: str) -> str:
        G = self.label(g)
        if g in PASSING:
            return (f"{G} is already a pass. For GPA course units, the handbook's repeat rules cover only grades "
                    "below C (must be repeated) and weak passes (may be repeated, up to C); NGPA units with C or "
                    "higher may be repeated to upgrade." + self._cite("may be repeated if"))
        if g in WEAK:
            return (f"Yes. {G} is a permitted weak pass, so you may repeat the course to upgrade it, but the best "
                    "grade you can get on the repeat is C." + self._cite("up to a maximum of grade"))
        if g in ("ie", "ia", "ib"):
            return self.grade_reply(g)
        if g in NOT_IN_TABLE or g in ("ab", "ne"):
            return ("Yes - in fact you must. A GPA course unit with a grade below C that is not a permitted weak "
                    "pass has to be repeated to obtain a pass grade (re-sitting the failed component: Ie -> exam, "
                    "Ia -> CA, Ib -> both). A charge is levied by the registry for each re-sit."
                    + self._cite("shall be repeated to obtain a pass", "A charge shall be levied"))
        if g == "ex":
            return self.grade_reply(g)
        return ""

    # ------------------------------------------------------------------ main entry
    def answer(self, q: str, module: str | None = None, last_modules: list[str] | None = None) -> dict | None:
        """q: the resolved question. module: module code from context (for "pass it"). Returns a reply dict."""
        ql = q.lower().strip()
        code = self._module_in(q) or None

        # prerequisites (never listed in the handbook)
        if re.search(r"\bpre-?\s?requisites?\b|\b(need|have) to (do|take|complete|pass) before\b|\brequired before "
                     r"(taking|doing)\b", ql):
            mod = code or module
            what = f" for {mod} - {self.qa.courses[mod]['title']}" if mod in self.qa.courses else " for modules"
            opts = [f"Which semester is {mod} in?", f"Tell me about {mod}"] if mod in self.qa.courses else \
                ["What modules are in semester 1?"]
            return self._r(f"The Student Handbook does not list prerequisites{what}. It only gives each module's "
                           "semester, credits and description; modules are taken in the order of the semester "
                           f"tables. For prerequisites, ask your department or {self.contact}.", "not_covered_prereq",
                           opts, module=mod)

        # difficulty / workload ranking
        if re.search(r"\b(hardest|toughest|easiest|most difficult|least difficult|hard|difficult|easy|tough)\b", ql) \
                and re.search(r"\b(which|what|is|are|module|modules|subject|course|them|it|one)\b", ql) \
                and not re.search(r"\b(get|got) an? (hard|easy)\b", ql):
            mods = [m for m in (last_modules or []) if m in self.qa.courses]
            if code:
                mods = [code]
            msg = ("The Student Handbook does not rate how hard modules are, so I can't say which is the "
                   "hardest or easiest.")
            if mods:
                top = sorted(mods, key=lambda m: (not self._credits(m).endswith('GPA credits') or 'NGPA' in self._credits(m) or 'MGPA' in self._credits(m), -self._credit_value(m)))
                creds = "; ".join(f"{m} {self.qa.courses[m]['title']} ({self._credits(m)})" for m in top[:4])
                msg += (f" What it does give is the credit value, which reflects the workload: {creds}.")
            return self._r(msg, "not_covered_difficulty", ["What is the pass mark?"])

        # head of a degree that has no department of its own
        m = re.search(r"\b(head(s|ed)?|in charge of|leads?)\b.*\b(software engineering|information systems|se|is)\b"
                      r"(?! (a|an|the)\b)", ql)
        if m and not re.search(r"\bdepartment of (computer science|information technology|computer engineering|"
                               r"computational)\b", ql):
            prog = "SE" if m.group(3) in ("software engineering", "se") else "IS"
            if prog == "IS" and m.group(3) == "is" and not re.search(r"\b(IS)\b", q):
                prog = None
            if prog:
                dept = self._dept_of(prog)
                head = self.qa.answer(f"who is the head of the {dept.lower()}?") if dept else None
                if head and head.get("kind") == "person_role":
                    name = re.sub(r"^The Head of the .*? is ", "", head["answer"]).rstrip(".")
                    pname = "Software Engineering" if prog == "SE" else "Information Systems"
                    return self._r(f"There is no separate {pname} department: BSc (Hons) in {pname} is offered by "
                                   f"the {dept}, so its head is {name}.", "person_role", person=name,
                                   dept=dept)

        # repeating to improve a weak pass
        if re.search(r"\bweak pass(es)?\b", ql) and re.search(r"\b(repeat\w*|re-?sit\w*|improve|upgrad\w*|redo|retake)\b", ql):
            return self._r("Yes. A GPA course unit with a permitted weak pass (D+ or C-) may be repeated if you want "
                           "to upgrade it, but only up to a maximum of grade C." + self._cite("up to a maximum of grade"),
                           "rule_grade_repeat", grade="d+")

        # passing a module / passing grade / pass mark
        if re.search(r"\b(pass(ing)? (grade|mark|marks|score|percentage)|minimum (grade|marks?|score) (to|for|needed|"
                     r"required)|(grade|marks?|score|percentage) (do i need|needed|required|to get|should i get) "
                     r"(to|for) pass|how much (marks? )?(do i need|should i get|to get) to pass|what (grade|marks?) "
                     r"(is|counts as) a pass|need to pass|to pass (it|this|that|the (module|course|subject|exam))|"
                     r"how (do|can) i pass)\b", ql) \
                and not re.search(r"\b(semester|degree|class|graduat\w*|programme|program|weak|repeat\w*|re-?sit\w*)\b", ql):
            mod = code or (module if re.search(r"\b(it|this|that|the (module|course|subject))\b", ql) else None)
            head = ""
            if mod and mod in self.qa.courses:
                head = (f"For {mod} - {self.qa.courses[mod]['title']}, the general rule applies (the handbook "
                        "gives no different pass mark for individual modules): ")
            body = ("you need grade C or better, i.e. a total of at least 45%, and at least 35% in each component "
                    "(continuous assessment and the end-semester exam; usually weighted 30% / 70%). D+ (35-39) and "
                    "C- (40-44) are only \"weak passes\" - at most one is allowed per semester, and they can be "
                    "upgraded by repeating only up to C.")
            ans = (head + body) if head else ("To pass a course " + body)
            return self._r(ans[0].upper() + ans[1:] + self._cite("total pass mark", "not more than one"),
                           "rule_pass_mark", module=mod)

        # weak pass
        if re.search(r"\bweak pass(es)?\b", ql) and re.search(r"\b(what|meaning|mean|explain|define|which|how many|"
                                                             r"allowed|tell)\b", ql):
            return self._r("A weak pass is a D+ (35-39 marks, GPV 1.30) or C- (40-44 marks, GPV 1.70). To complete a "
                           "semester you may have not more than one D+ or C- in GPA course units. A weak pass may be "
                           "repeated to upgrade it, up to a maximum of C. For graduation you may not have more than "
                           "one D+/C- per semester, and carrying a weak pass over rules out First Class and Second "
                           "Class (Upper Division)." + self._cite("not more than one", "up to a maximum of grade",
                                                                  "not have carried over any weak"),
                           "rule_weak_pass", grade="d+")

        # how many times can I repeat / re-sit
        if re.search(r"\bhow many (times|attempts|chances)\b|\b(limit|maximum|max)\b.*\b(repeat\w*|re-?sit\w*|attempts?)"
                     r"\b|\b(repeat\w*|re-?sit\w*)\b.*\b(limit|how many times|unlimited|again and again)\b", ql) and \
                not re.search(r"\b(weak pass(es)?|d\+|c-) (are|is) allowed\b", ql):
            return self._r("The handbook does not set a fixed number of repeat attempts. The limits are: the degree "
                           f"must be completed within the maximum period ({self.max_years} years, i.e. sixteen "
                           "semesters) or you are deemed to have discontinued; each re-sit is charged by the "
                           "registry; supplementary exams are held only after Semesters 7 and 8; and a weak pass "
                           "can be upgraded only up to C."
                           + self._cite("maximum period of sixteen", "A charge shall be levied",
                                        "Supplementary Examinations will not"), "rule_repeat_limit")

        # repeat with a known grade ("can I repeat a module with a D?") - the Prolog layer handles D+/C-/C+
        g = self.find_grade(q)
        if g and re.search(r"\b(repeat|re-?sit|redo|retake|upgrade|improve)\b", ql):
            text = self.repeat_for_grade(g)
            if text:
                return self._r(text, "rule_grade_repeat", grade=g)

        # one grade letter: "what if I get a D?", "I got C-", "what does Ab mean?", "is D+ a pass?"
        if g and (re.search(r"\b(what (if|happens|does|is|about)|i (got|get|received|have)|is .* (a )?(pass|fail)|"
                             r"mean(ing|s)?|did i pass|do i pass|am i|pass or fail|got an?|get an?)\b", ql)
                  or re.fullmatch(r"(grade )?[\"'‘“]?" + _GRADE_ALT + r"[\"'’”]?\s*\??", ql)):
            if not re.search(r"\b(gpv|grade point|how many marks|marks for|marks is|range)\b", ql) or g in NOT_IN_TABLE:
                text = self.grade_reply(g)
                if text:
                    return self._r(text, "rule_grade", grade=g)

        # requirements for a Pass (the class) / a pass degree
        if re.search(r"\b(pass degree|general degree|ordinary pass|pass class|just (a )?pass|a pass)\b", ql) and \
                re.search(r"\b(requirements?|criteria|need(ed)?|get|for|conditions?|what)\b", ql) and \
                not re.search(r"\b(module|course|subject|exam|component|mark)\b", ql):
            return self._r(self.pass_class_text(), "class_criteria", topic_class="pass")
        return None

    def pass_class_text(self) -> str:
        row = self.class_rows.get("Pass")
        band = row["condition"] if row else "2.00 <= FGPA and FGPA < 3.00"
        return (f"Pass: {band}. A Pass has no extra conditions beyond graduating, which requires: completing the "
                "academic requirements of all semesters; the minimum GPA and NGPA credits of the curriculum; "
                "finishing within the maximum period; an FGPA of not less than 2.00; and not more than one D+ or "
                "C- per semester." + self._cite("Earning a GPA of not less than 2.00"))

    def section_fallback(self, q: str) -> dict | None:
        """Graceful degradation for a "not found": if every content word of a section title is in the question
        ("attendance rule" -> 4.3 Attendance), answer with the opening of that section, clearly labelled."""
        from handbook_qa import stems
        qs = set(stems(q.lower()))
        best = None
        for sec in self.qa.kb.get("sections", []):
            if sec["number"] in ("0",) or sec["number"].startswith(("7", "3.3")) or not sec.get("text"):
                continue
            ts = {t for t in stems(sec["title"].lower()) if len(t) > 2}
            if not ts or not ts <= qs:
                continue
            key = (len(ts), sec.get("level", 0))
            if best is None or key > best[0]:
                best = (key, sec)
        if not best:
            return None
        sec = best[1]
        text = re.sub(r"\s+", " ", sec["text"]).strip()
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        body = " ".join(sents[:2])[:450].rstrip()
        if len(body) < 40:
            return None
        return self._r(f"I didn't find an exact match, but section {sec['number']} ({sec['title']}) of the handbook "
                       f"says: {body}" + (f" (Student Handbook, p. {sec['page']})" if sec.get("page") else ""),
                       "section_fallback", [f"Tell me about {sec['title'].lower()}"])

    def not_covered(self, q: str) -> dict | None:
        ql = q.lower()
        if COVERED_OVERRIDE.search(ql):
            return None
        for label, rx, opts in _NOT_COVERED:
            if rx.search(ql):
                return self._r(f"I couldn't find anything about {label} in the Student Handbook - it doesn't cover "
                               f"that topic. For that, please contact {self.contact}.", "not_covered",
                               opts or ["What is the pass mark?", "How long is the degree?"])
        return None

    # ------------------------------------------------------------------ small helpers
    def _module_in(self, q: str) -> str | None:
        m = re.search(r"\b([A-Za-z]{2}\s?\d{4})\b", q)
        if m:
            c = m.group(1).replace(" ", "").upper()
            if c in self.qa.courses:
                return c
        t = self.qa.match_course_title(q.lower())
        return t[0] if t else None

    def _credit_value(self, code: str) -> float:
        o = (self.qa.courses[code].get("offerings") or [{}])[0]
        return float(o.get("gpa") or o.get("ngpa") or o.get("mgpa") or 0)

    def _credits(self, code: str) -> str:
        o = (self.qa.courses[code].get("offerings") or [{}])[0]
        for key in ("gpa", "ngpa", "mgpa"):
            if o.get(key):
                return f"{o[key]:g} {key.upper()} credits"
        return "credits not listed"

    def _dept_of(self, prog: str) -> str | None:
        from handbook_qa import DEPARTMENTS
        for code, d in self.qa.kb.get("departments", {}).items():
            if isinstance(d, dict) and prog in (d.get("degrees") or []):
                return DEPARTMENTS.get(code)
        return None

    @staticmethod
    def _r(text: str, kind: str, options: list[str] | None = None, **state) -> dict:
        out = {"answer": text, "kind": kind}
        if options:
            out["options"] = options
        out["state"] = {k: v for k, v in state.items() if v}
        return out
