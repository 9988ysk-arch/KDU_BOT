"""
logic_layer.py - Prolog reasoning layer for the Student Handbook assistant.

1. Generates handbook_facts.pl from the knowledge base (modules, grades, classes, programmes, staff,
   admission rules, credit tables). Every fact group is taken from the PDF, with its page in a comment.
2. Loads handbook_facts.pl + handbook_rules.pl into the bundled pure-Python Prolog interpreter
   (prolog_engine.py). No GPU, no LLM, nothing to install. The same two files also load in SWI-Prolog
   (swipl data/handbook_facts.pl handbook_rules.pl) for checking or teaching the rules.
3. Answers questions that need *reasoning* over several rules, which keyword retrieval cannot do:
   GPA calculators, "did I complete the semester?", "which class can I get with ...?", graduation checks,
   military / A-level stream eligibility, component grades (CA/ES), repeat rules, elective credits.
4. Verifies the engine's grade and class answers against the rules (a second, independent path).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from prolog_engine import Prolog, PrologError

HERE = Path(__file__).resolve().parent
RULES = HERE / "handbook_rules.pl"
GRADE = r"(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|Ie|Ia|Ib|Ne|Ab|Ex)"
GRADE_RX = re.compile(r"(?<![\w+\-])" + GRADE + r"(?![\w+\-])")
GRADE_LOWER_RX = re.compile(r"(?<![\w+\-])([abcd][+\-])(?![\w+\-])")
NUM_WORDS = {"no": 0, "zero": 0, "none": 0, "one": 1, "a": 1, "an": 1, "single": 1, "two": 2, "three": 3,
             "four": 4, "five": 5}
PROG_WORDS = {"CS": r"computer science|\bcs\b", "SE": r"software engineering|\bse\b",
              "CE": r"computer engineering|\bce\b", "IT": r"information technology|\bIT\b",
              "IS": r"information systems|\bIS\b"}


def q(atom: Any) -> str:
    """Python value -> Prolog atom/number literal."""
    if isinstance(atom, (int, float)):
        return repr(float(atom)) if isinstance(atom, float) else str(atom)
    s = str(atom).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"


def fmt(x: float) -> str:
    return f"{x:.2f}"


# ============================================================================ fact generation
def build_facts(qa) -> str:
    kb = qa.kb
    out = ["% handbook_facts.pl - generated from the Student Handbook 2022 by logic_layer.py. Do not edit.",
           ":- discontiguous unit/9.", ""]
    out.append("% unit(Code, Title, Programme, Level, Semester, Category, GPA, NGPA, MGPA)  (module tables, pp.17-52)")
    for code, c in sorted(qa.courses.items()):
        for o in c["offerings"]:
            cat = (o.get("category") or "unknown").lower()
            out.append(f"unit({q(code)}, {q(c['title'])}, {q(o['program'])}, {o['level']}, {o['semester']}, "
                       f"{cat}, {o['gpa'] or 0}, {o['ngpa'] or 0}, {o['mgpa'] or 0}).")
    out += ["", "% grade_band(Grade, LowMarks, HighMarks, GPV) and failure grades (Table 12, p.54)"]
    order = []
    for r in kb["grades"]["rows"]:
        order.append(r["grade"])
        if r["low"] is not None:
            out.append(f"grade_band({q(r['grade'])}, {r['low']}, {r['high']}, {float(r['gpv'])}).")
        elif r["grade"] != "Ex":
            out.append(f"failure_grade({q(r['grade'])}).")
    order = list(dict.fromkeys(order))
    out.append(f"grade_order([{', '.join(q(g) for g in order)}]).")
    out += ["", "% class bands (Table 13, p.58)"]
    out.append("class_order([" + ", ".join(q(r["result"]) for r in kb["classes"]["rows"]) + "]).")
    for r in kb["classes"]["rows"]:
        out.append(f"class_band({q(r['result'])}, {r['low']}, {r['high'] if r['high'] else 'none'}).")
    out += ["", "% programme(Code, Name, CivilYears, MilitaryYears, Stream)  (Table 1, p.10); max years (Table 11)"]
    for code, p in kb["programmes"].items():
        mil = 4.5 if "½" in str(p.get("military_years", "")) else float(p.get("military_years") or 0)
        out.append(f"programme({q(code)}, {q(p['name'])}, {float(p.get('civil_years') or 0)}, {mil}, "
                   f"{q(p.get('stream', ''))}).")
    for code, y in kb["max_duration"].items():
        out.append(f"max_years({q(code)}, {y}).")
    out += ["", "% departments and the degrees they offer (pp.9-10)"]
    for code, d in kb["departments"].items():
        out.append(f"department({q(code)}, {q(d['name'])}).")
        for deg in d.get("degrees", []):
            out.append(f"offers({q(code)}, {q(deg)}).")
    out += ["", "% staff(Name, Department, Designation, Email) and roles (pp.60-86)"]
    for st in kb["staff"]:
        out.append(f"staff({q(st['name'])}, {q(st['department'])}, {q(st.get('designation', ''))}, "
                   f"{q(st.get('email', ''))}).")
        role = st.get("role") or ""
        if role.startswith("Dean"):
            out.append(f"dean({q(st['name'])}).")
        m = re.search(r"Head\s*[–-]?\s*(?:the )?(Department of [A-Za-z ]+)", role)
        if m:
            dept = next((c for c, d in kb["departments"].items() if d["name"].lower() == m.group(1).strip().lower()),
                        None)
            if dept:
                out.append(f"head({q(dept)}, {q(st['name'])}).")
    out += ["", "% semester_total(Programme, Semester, GPA, NGPA, MGPA)  (credit summary tables, p.40 and p.52)"]
    for g in kb["gpa_summary"]:
        prog = "CS" if g["rows"]["Total"]["gpa"] == 128 else "SE"
        for name, r in g["rows"].items():
            m = re.match(r"Semester (\d)", name)
            if m:
                out.append(f"semester_total({q(prog)}, {m.group(1)}, {r['gpa']}, {r['ngpa']}, {r['mgpa']}).")
    out += ["", "% minimum height for military students (p.10)"]
    for key, val in kb["heights"].items():
        gender, service = key.split("|")
        cm = float(re.match(r"[\d.]+", val).group(0))
        out.append(f"height_min({q(gender)}, {q(service)}, {cm}).")
    # military requirements, read from the bullet list on p.10
    texts = " ".join(p["text"] for p in qa.passages if p["page"] == 10)
    age = re.search(r"not less than (\d+) years and not more than (\d+) years", texts)
    weight = re.search(r"body weight not less than (\d+) kg", texts)
    chest = re.search(r"chest not less than ([\d.]+) cm", texts)
    out += ["", "% military applicants (p.10)"]
    if age:
        out += [f"military_min_age({age.group(1)}).", f"military_max_age({age.group(2)})."]
    if weight:
        out.append(f"military_min_weight({weight.group(1)}).")
    if chest:
        out.append(f"military_min_chest({chest.group(1)}).")
    for req in ("age", "married", "weight", "chest", "citizen", "height"):
        out.append(f"military_requirement({req}).")
    return "\n".join(out) + "\n"


# ============================================================================ engine wrapper
class LogicLayer:
    def __init__(self, qa, facts_path: str | Path | None = None):
        self.qa = qa
        self.facts_text = build_facts(qa)
        self.facts_path = Path(facts_path) if facts_path else HERE / "data" / "handbook_facts.pl"
        try:
            self.facts_path.parent.mkdir(parents=True, exist_ok=True)
            self.facts_path.write_text(self.facts_text, encoding="utf-8")
        except OSError:
            pass
        self.pl = Prolog()
        self.pl.consult_text(self.facts_text)
        self.pl.consult(str(RULES))
        self.backend = "bundled interpreter (prolog_engine.py)"

    # -- helpers
    def ask(self, goal: str, limit: int | None = None) -> list[dict[str, Any]]:
        return self.pl.query(goal, limit)

    def holds(self, goal: str) -> bool:
        return self.pl.holds(goal)

    def with_student(self, facts: dict[str, Any], goal: str) -> list[dict[str, Any]]:
        """Run a goal with temporary student/2 facts (what the user told us)."""
        key = ("student", 2)
        saved = self.pl.db.get(key, [])
        self.pl.db[key] = []
        self.pl.consult_text("\n".join(f"student({k}, {q(v)})." for k, v in facts.items()) or "")
        try:
            return self.pl.query(goal)
        finally:
            self.pl.db[key] = saved

    def same_modules(self, period: str) -> bool | None:
        """'semester 3' / 'year 2' -> True when CS and SE have exactly the same modules (no degree slot)."""
        m = re.match(r"(semester|year) (\d)", period)
        if not m:
            return None
        n = int(m.group(2))
        sems = [n] if m.group(1) == "semester" else [2 * n - 1, 2 * n]
        return all(self.holds(f"same_modules({s})") for s in sems)

    # -- question understanding
    @staticmethod
    def grades_in(text: str) -> list[str]:
        found = [(m.start(), m.group(1)) for m in GRADE_RX.finditer(text)]
        found += [(m.start(), m.group(1).upper()) for m in GRADE_LOWER_RX.finditer(text)]
        return [g for _, g in sorted(found)]

    @staticmethod
    def count_of(pattern: str, text: str) -> int | None:
        m = re.search(rf"\b(\d|no|zero|none|one|a|an|single|two|three|four|five)\s+(?:\w+\s+)?{pattern}", text)
        if m:
            v = m.group(1)
            return int(v) if v.isdigit() else NUM_WORDS[v]
        return None

    def answer(self, question: str) -> dict[str, Any] | None:
        text = question.strip()
        low = text.lower()
        for handler in (self.gpa_calculator, self.component_grade, self.semester_check, self.class_check,
                        self.graduation_check, self.military_check, self.stream_check, self.repeat_check,
                        self.credit_sum, self.electives_needed):
            try:
                res = handler(text, low)
            except PrologError:
                res = None
            if res:
                res.setdefault("confidence", 0.95)
                res["source"] = "prolog"
                return res
        return None

    # 1. GPA calculator ------------------------------------------------------------------------------
    def gpa_calculator(self, text: str, low: str) -> dict | None:
        if not re.search(r"\b(s?gpa|fgpa|ygpa|calculate|compute|work out|average)\b", low):
            return None
        pairs = []
        for m in re.finditer(r"([A-Za-z]{2,3})\s?(\d{4})\s*(?:[:=\-]|got|is|was|with|grade|->)?\s*" + GRADE, text):
            pairs.append((m.group(1).upper() + m.group(2), m.group(3)))
        for m in re.finditer(GRADE + r"\s*(?:in|for|on|at)\s*([A-Za-z]{2,3})\s?(\d{4})", text):
            pairs.append((m.group(2).upper() + m.group(3), m.group(1)))
        items = []
        for m in re.finditer(GRADE + r"[\s(]*(?:in|for|with|on|from)?\s*(?:a\s*)?(\d(?:\.\d)?)\s*[- ]?credits?", text):
            items.append((m.group(1), float(m.group(2))))
        for m in re.finditer(r"(\d(?:\.\d)?)\s*[- ]?credits?\s*(?:module|course|subject|unit)?\s*(?:with|got|=|:|->|"
                             r"grade|and got|i got)?\s*" + GRADE + r"(?![\w+\-])", text):
            items.append((m.group(2), float(m.group(1))))
        if pairs:
            plist = "[" + ", ".join(f"m({q(c)}, {q(g)})" for c, g in pairs) + "]"
            r = self.pl.once(f"gpa_of_modules({plist}, S, Used, Skipped)")
            if not r:
                return {"answer": "I couldn't calculate that: none of those module codes has GPA credits in the "
                                  "handbook tables.", "kind": "logic_gpa"}
            credits = {c: self.pl.once(f"unit_gpa_credits({q(c)}, X)")["X"] for c in r["Used"]}
            terms = " + ".join(f"{self.pl.once(f'gpv({q(g)}, P)')['P']:.2f}×{credits[c]:g}" for c, g in pairs
                               if c in credits)
            total = sum(credits[c] for c, _ in pairs if c in credits)
            ans = f"GPA = ({terms}) / {total:g} = {fmt(r['S'])}."
            if r["Skipped"]:
                ans += " Not counted (no GPA credits in the handbook tables): " + ", ".join(r["Skipped"]) + "."
            return {"answer": ans + " (Only GPA credits count, using the SGPA formula on p.54.)", "kind": "logic_gpa",
                    "page": 54}
        if items:
            ilist = "[" + ", ".join(f"g({q(g)}, {c:g})" for g, c in items) + "]"
            r = self.pl.once(f"gpa_items({ilist}, S)")
            if not r:
                return None
            terms = " + ".join(f"{self.pl.once(f'gpv({q(g)}, P)')['P']:.2f}×{c:g}" for g, c in items if g != "Ex")
            total = sum(c for g, c in items if g != "Ex")
            return {"answer": f"GPA = ({terms}) / {total:g} = {fmt(r['S'])}. (SGPA formula, p.54.)",
                    "kind": "logic_gpa", "page": 54}
        return None

    # 2. grade from CA and ES components ----------------------------------------------------------------
    def component_grade(self, text: str, low: str) -> dict | None:
        ca = re.search(r"\b(?:ca|continuous assessment|coursework|course work)\s*(?:mark|marks|score)?\s*(?:is|was|=|:|of)?\s*"
                       r"(\d{1,3})", low)
        es = re.search(r"\b(?:es|we|exam|final exam|written exam|end semester exam|paper)\s*(?:mark|marks|score)?\s*"
                       r"(?:is|was|=|:|of)?\s*(\d{1,3})", low)
        if not (ca and es) or not re.search(r"\b(grade|pass|fail|result|get)\b", low):
            return None
        cav, esv = int(ca.group(1)), int(es.group(1))
        if cav > 100 or esv > 100:
            return None
        tot = re.search(r"\b(?:total|final mark|overall)\s*(?:is|was|=|:|of)?\s*(\d{1,3})", low)
        total = int(tot.group(1)) if tot else round(0.3 * cav + 0.7 * esv)
        r = self.pl.once(f"grade_of_components({total}, {cav}, {esv}, G, P)")
        if not r:
            return None
        g = r["G"]
        if g in ("Ie", "Ia", "Ib"):
            comp = self.pl.once(f"resit_component({q(g)}, C)")["C"]
            ans = (f"Grade {g} (GPV 0.00): every component needs at least 35%, so this is a failure grade. "
                   f"You re-sit the {comp} component(s).")
        else:
            how = "" if tot else " (total worked out with the usual 30% CA / 70% exam weighting, p.53)"
            ans = f"Total {total}{how} = grade {g} (GPV {r['P']:.2f})." + (
                " That is a pass." if self.holds(f"pass_grade({q(g)})") else
                " That is a weak pass (C- or D+)." if self.holds(f"weak_pass({q(g)})") else "")
        return {"answer": ans, "kind": "logic_component_grade", "page": 54}

    # 3. completing a semester -----------------------------------------------------------------------------
    def semester_check(self, text: str, low: str) -> dict | None:
        if not re.search(r"\b(semester|sem)\b", low) or not re.search(
                r"\b(pass|passed|complete|completed|clear|cleared|ok|okay|fine|progress|successful\w*)\b", low):
            return None
        grades = self.grades_in(text)
        if len(grades) < 2 or re.search(r"\b[a-z]{2,3}\s?\d{4}\b", low) and re.search(r"\bgpa\b", low):
            return None
        glist = "[" + ", ".join(q(g) for g in grades) + "]"
        probs = [r["R"] for r in self.ask(f"semester_problem({glist}, R)")]
        if not probs:
            return {"answer": f"Yes. With {', '.join(grades)} you meet the semester requirements: no failure grades "
                              f"or Ex, and at most one D+ or C- (p.55, 4.5.1).", "kind": "logic_semester", "page": 55}
        why = []
        for p in dict.fromkeys(map(str, probs)):
            if p.startswith("failure("):
                g = p[8:-1]
                comp = self.pl.once(f"resit_component({q(g)}, C)")
                why.append(f"{g} is a failure grade" + (f" (re-sit the {comp['C']} component(s))" if comp else ""))
            elif p == "excused":
                why.append("Ex means the unit is not completed yet (re-sit as a first attempt)")
            elif p.startswith("too_many_weak"):
                why.append(f"you have {p.split('(')[1][:-1]} weak passes (D+/C-), but only one is allowed per semester")
        return {"answer": "No, the semester is not completed yet: " + "; ".join(why) + ". (Rules on p.55, 4.5.1.)",
                "kind": "logic_semester", "page": 55}

    # 4. which class -------------------------------------------------------------------------------------
    def class_check(self, text: str, low: str) -> dict | None:
        f = re.search(r"\b(?:f?gpa|fgpa of|gpa of)\s*(?:of|is|=|:)?\s*([0-4]\.\d{1,2})\b|\b([0-4]\.\d{1,2})\s*(?:f?gpa)\b",
                      low)
        if not f or not re.search(r"\b(fail\w*|weak|carried|carry|repeat\w*|re-?sit\w*|d\+|c-|incomplete)\b", low):
            return None
        fgpa = float(f.group(1) or f.group(2))
        fails = self.count_of(r"fail(?:ure|ed|s)?\w*", low)
        if fails is None:  # "I failed 2 modules", "failed one course"
            m = re.search(r"\bfail(?:ed)?\s+(\d|one|two|three|four|five|a|an)\s+(?:\w+\s+)?(?:modules?|courses?|"
                          r"subjects?|units?|exams?|papers?|course units?)\b", low)
            if m:
                fails = int(m.group(1)) if m.group(1).isdigit() else NUM_WORDS[m.group(1)]
        if fails is None:
            fails = 1 if re.search(r"\bi (failed|have failed|got a fail)\b|\bfailure grade\b|\ba fail\b", low) else 0
        fails78 = fails if re.search(r"\b(semester|sem) ?(7|8)\b|\bfinal year\b|\bfourth year\b|\blevel 4\b", low) else 0
        weak = 0
        if re.search(r"\b(carried|carry|still have|not repeated|outstanding)\b.*\b(weak|d\+|c-)|\b(weak|d\+|c-)\b.*\b"
                     r"(carried|outstanding|not repeated)\b", low):
            weak = self.count_of(r"(?:weak|d\+|c-)", low) or 1
        cfail = 1 if re.search(r"\b(carried|carry|still have|not (cleared|repeated|re-?sat)|outstanding|incomplete)\b.*"
                               r"\bfail", low) or re.search(r"\bfail\w*\b.*\b(not (cleared|repeated|re-?sat)|still|"
                                                             r"outstanding|carried)\b", low) else 0
        args = f"{fgpa}, {fails}, {fails78}, {weak}, {cfail}"
        best = self.pl.once(f"best_class({args}, C)")
        order = self.pl.once("class_order(O)")["O"]
        lines = []
        for cls in order:
            if best and cls == best["C"]:
                break
            reasons = [str(r["R"]) for r in self.ask(f"blocker({q(cls)}, {args}, R)")]
            if reasons:
                lines.append(f"not {cls} because " + ", ".join(self._reason(r) for r in dict.fromkeys(reasons)))
        facts = (f"FGPA {fgpa:.2f}, {fails} failure grade(s)" + (f" ({fails78} in semesters 7-8)" if fails78 else "")
                 + (f", {weak} weak pass(es) carried over" if weak else "") + (", a failure still carried over"
                                                                                if cfail else ""))
        if best:
            ans = f"With {facts}: {best['C']}."
        else:
            ans = f"With {facts}: no class can be awarded yet."
        if lines:
            ans += " (" + "; ".join(lines) + ".)"
        return {"answer": ans + " Rules: Table 13 and p.57-58.", "kind": "logic_class", "page": 57}

    @staticmethod
    def _reason(r: str) -> str:
        m = re.match(r"(\w+)\((.*)\)", r)
        name, arg = (m.group(1), m.group(2)) if m else (r, "")
        if name == "fgpa_below":
            return f"the FGPA is below {float(arg):.2f}"
        if name == "failures":
            n, mx = arg.split(",")
            return f"{n.strip()} failure grade(s) (at most {mx.strip()} allowed)"
        if name == "failure_in_sem_7_8":
            return "a failure grade in semester 7 or 8"
        if name == "carried_weak":
            return "weak passes are still carried over"
        if name == "carried_fail":
            return "a failure grade is still carried over"
        return r

    # 5. graduation ---------------------------------------------------------------------------------------
    def graduation_check(self, text: str, low: str) -> dict | None:
        if not re.search(r"\b(graduate|graduation|get (my|the) degree|complete (my|the) degree|degree awarded)\b", low):
            return None
        g = re.search(r"\b(?:f?gpa)\s*(?:of|is|=|:)?\s*([0-4]\.\d{1,2})\b|\b([0-4]\.\d{1,2})\s*(?:f?gpa)?\b", low)
        if not g:
            return None
        gpa = float(g.group(1) or g.group(2))
        weak = self.count_of(r"(?:weak|d\+|c-)\w*", low) or 0
        probs = [str(r["R"]) for r in self.ask(f"graduation_problem({gpa}, {weak}, R)")]
        if not probs:
            cls = self.pl.once(f"best_class({gpa}, 0, 0, 0, 0, C)")
            return {"answer": f"Yes, a GPA of {gpa:.2f} meets the graduation GPA rule (at least 2.00 for the whole "
                              f"degree)" + (f"; on the FGPA alone that is {cls['C']}" if cls else "") +
                              ". The other criteria still apply: all semesters completed within the maximum period, "
                              "the minimum GPA and NGPA credits, and not more than one D+ or C- per semester (p.57).",
                    "kind": "logic_graduation", "page": 57}
        why = []
        for p in probs:
            if p.startswith("gpa_below"):
                why.append(f"a GPA of {gpa:.2f} is below the required 2.00")
            elif p.startswith("too_many_weak"):
                why.append("more than one D+ or C- in a semester is not allowed")
        return {"answer": "No: " + "; ".join(why) + " (criteria e and f, p.57).", "kind": "logic_graduation",
                "page": 57}

    # 6. military applicants -------------------------------------------------------------------------------
    def military_check(self, text: str, low: str) -> dict | None:
        if not re.search(r"\b(military|army|navy|air ?force|cadet|officer cadet)\b", low) or not re.search(
                r"\b(can i|eligible|eligibility|qualify|apply|join|allowed|enough|ok)\b", low):
            return None
        facts: dict[str, Any] = {}
        m = re.search(r"\b(\d{2})\s*(?:years?|yrs?)(?: old)?\b|\bage\s*(?:is|of|:)?\s*(\d{2})\b|\bi'?m (\d{2})\b|\bi am (\d{2})\b",
                      low)
        if m:
            facts["age"] = int(next(x for x in m.groups() if x))
        m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*kg", low)
        if m:
            facts["weight"] = float(m.group(1))
        m = re.search(r"chest\s*(?:is|of|:)?\s*(\d{2,3}(?:\.\d+)?)", low)
        if m:
            facts["chest"] = float(m.group(1))
        m = re.search(r"(\d{3}(?:\.\d+)?)\s*cm", low)
        if m and "chest" not in low[max(0, m.start() - 12):m.start()]:
            facts["height"] = float(m.group(1))
        if re.search(r"\b(unmarried|single|not married)\b", low):
            facts["married"] = "no"
        elif re.search(r"\b(married|my (wife|husband))\b", low):
            facts["married"] = "yes"
        if re.search(r"\b(not (a )?(sri lankan|citizen)|foreign\w*)\b", low):
            facts["citizen"] = "no"
        elif re.search(r"\b(sri lankan|citizen)\b", low):
            facts["citizen"] = "yes"
        g = re.search(r"\b(male|female|man|woman|boy|girl)\b", low)
        if g:
            facts["gender"] = "female" if g.group(1) in ("female", "woman", "girl") else "male"
        s = re.search(r"\b(army|navy|air ?force)\b", low)
        if s:
            facts["service"] = "air force" if "air" in s.group(1) else s.group(1)
        if not (set(facts) & {"age", "weight", "chest", "height", "married", "citizen"}):
            return None
        probs = [str(r["P"]) for r in self.with_student(facts, "military_problem(P)")]
        unknown = [r["R"] for r in self.with_student(facts, "military_unknown(R)")]
        limits = self.pl.once("military_min_age(A1), military_max_age(A2), military_min_weight(W), "
                              "military_min_chest(C)")
        words = {"age": f"age {limits['A1']}-{limits['A2']} on the closing date", "married": "unmarried",
                 "weight": f"weight at least {limits['W']} kg", "chest": f"chest at least {limits['C']} cm",
                 "citizen": "Sri Lankan citizen", "height": "minimum height for your gender and service"}
        if "height" in facts and not ({"gender", "service"} <= set(facts)):
            unknown.append("height")
        if probs:
            why = []
            for p in dict.fromkeys(probs):
                name = p.split("(")[0]
                if name == "height":
                    why.append(f"height {facts['height']:g} cm is below the minimum of "
                               f"{float(p.split(',')[1].strip(' )')):g} cm for a {facts['gender']} "
                               f"{facts['service']} applicant")
                else:
                    why.append(f"{name} does not meet the rule ({words[name]})")
            ans = "No, as a military applicant: " + "; ".join(why) + "."
        else:
            ans = "Yes, what you told me meets the military-applicant rules (" + ", ".join(
                words[k] for k in ("age", "married", "weight", "chest", "citizen", "height") if k in facts) + ")."
        if unknown:
            ans += " Also required: " + "; ".join(words[u] for u in dict.fromkeys(unknown)) + "."
        return {"answer": ans + " (p.10)", "kind": "logic_military", "page": 10}

    # 7. A/L stream -> degree ------------------------------------------------------------------------------
    def stream_check(self, text: str, low: str) -> dict | None:
        m = re.search(r"\b(bio|biology|commerce|arts?|maths?|mathematics|physical science|technology|tech)\b\s*"
                      r"(?:stream|subjects?)?", low)
        if not m or not re.search(r"\b(stream|a/?l|advanced level)\b", low) and not re.search(
                r"\bdid (bio|biology|commerce|arts|maths|technology)\b", low):
            return None
        if not re.search(r"\b(can i|eligible|qualify|apply|allowed|which degree|what degree|enter|join|follow|do)\b", low):
            return None
        word = m.group(1)
        stream = {"bio": "biology", "biology": "biology", "commerce": "commerce", "art": "arts", "arts": "arts",
                  "math": "maths", "maths": "maths", "mathematics": "maths", "physical science": "maths",
                  "technology": "technology", "tech": "technology"}[word]
        progs = [p for p, rx in PROG_WORDS.items() if re.search(rx, text if p in ("IT", "IS") else low)]
        ok = [r["P"] for r in self.ask(f"stream_ok(P, {q(stream)})")]
        ok = list(dict.fromkeys(ok))
        names = {p: self.pl.once(f"programme({q(p)}, N, _, _, _)")["N"] for p in ("CS", "SE", "CE", "IT", "IS")}
        tail = (" You also need three S passes at A/L, a Credit (C) pass in O/L English and 30+ marks in the Common "
                "General Test (p.10).")
        if progs:
            res = []
            for p in progs:
                if p in ok:
                    res.append(f"Yes for {names[p]}")
                else:
                    res.append(f"No for {names[p]}" + (" (it needs the Maths stream, or Maths, Physics and one of "
                                                       "Chemistry/Higher Maths/ICT)" if p in ("CS", "SE", "CE")
                                                       else " (Technology stream is not accepted)"))
            return {"answer": f"With the {stream} stream: " + "; ".join(res) + "." + tail, "kind": "logic_stream",
                    "page": 10}
        if not ok:
            return {"answer": f"No. The {stream} stream is not accepted: CS, SE and CE need the Maths stream (or Maths, "
                              f"Physics and one of Chemistry/Higher Maths/ICT), and IT/IS accept Biology, Maths, "
                              f"Commerce or Arts, except Technology." + tail,
                    "kind": "logic_stream", "page": 10}
        return {"answer": f"With the {stream} stream you can apply for: " + ", ".join(names[p] for p in ok) + "." + tail,
                "kind": "logic_stream", "page": 10}

    # 8. repeating a unit -----------------------------------------------------------------------------------
    def repeat_check(self, text: str, low: str) -> dict | None:
        if not re.search(r"\b(repeat|re-?sit|redo|upgrade|improve)\b", low):
            return None
        grades = self.grades_in(text)
        if len(grades) != 1 or re.search(r"\b(fee|charge|cost)\b", low):
            return None
        g = grades[0]
        if re.search(r"\bngpa\b", low):
            r = self.pl.once(f"ngpa_repeat_rule({q(g)}, R)")
            if r and r["R"] == "may_repeat_to_upgrade":
                return {"answer": f"Yes. Any NGPA course unit with a grade of C or higher may be repeated for upgrading the "
                                  f"given grade (p.55, rule i).", "kind": "logic_repeat",
                        "page": 55}
        r = self.pl.once(f"repeat_rule({q(g)}, R)")
        if not r:
            return None
        rule = r["R"]
        if rule == "must_repeat":
            comp = self.pl.once(f"resit_component({q(g)}, C)")
            ans = f"{g} is a failure grade, so the unit must be repeated" + (
                f" by re-sitting the {comp['C']} component(s)" if comp else "") + " (p.55, 4.5.2)."
        elif rule == "may_repeat_up_to_c":
            ans = (f"{g} is a permitted weak pass. You may repeat the unit to upgrade it, but only up to a maximum "
                   f"grade of C (p.55, rule j).")
        else:
            ans = (f"{g} is a pass. The handbook's repeat rules cover only GPA units below C (they must be repeated) "
                   f"and permitted weak passes D+/C- (they may be repeated, up to grade C); it gives no rule for "
                   f"repeating a GPA unit with {g}. An NGPA unit with C or higher may be repeated to upgrade (p.55).")
        return {"answer": ans, "kind": "logic_repeat", "page": 55}

    # 9. credits of several modules ------------------------------------------------------------------------
    def credit_sum(self, text: str, low: str) -> dict | None:
        codes = [a.upper() + b for a, b in re.findall(r"\b([A-Za-z]{2,3})\s?(\d{4})\b", text)]
        if len(codes) < 2 or not re.search(r"\bcredits?\b", low) or not re.search(
                r"\b(total|sum|together|all|altogether|if i (take|do|choose|select)|how many)\b", low):
            return None
        parts, total = [], 0.0
        for c in dict.fromkeys(codes):
            r = self.pl.once(f"unit({q(c)}, T, _, _, _, _, G, N, M)")
            if not r:
                parts.append(f"{c}: not in the module tables")
                continue
            val = r["G"] or r["N"] or r["M"]
            kind = "GPA" if r["G"] else "NGPA" if r["N"] else "MGPA"
            total += val
            parts.append(f"{c} {r['T']} {val:g} {kind}")
        return {"answer": f"Total {total:g} credits: " + "; ".join(parts) + ".", "kind": "logic_credit_sum"}

    # 10. electives needed in a semester ------------------------------------------------------------------
    def electives_needed(self, text: str, low: str) -> dict | None:
        if not re.search(r"\bhow many\b.*\belectives?\b|\belective credits?\b", low) or \
                not re.search(r"\b(take|choose|select|need|must|have to|required|do)\b", low):
            return None
        sem = re.search(r"\bsem(?:ester)?\s*([5-7])\b", low)
        if not sem:
            return None
        s = int(sem.group(1))
        progs = [p for p in ("CS", "SE") if re.search(PROG_WORDS[p], low)] or ["CS", "SE"]
        parts = []
        for p in progs:
            r = self.pl.once(f"elective_credits_needed({q(p)}, {s}, N), compulsory_credits({q(p)}, {s}, C), "
                             f"semester_total({q(p)}, {s}, T, _, _)")
            if r:
                n = r["N"]
                parts.append(f"{p}: {n:g} elective GPA credits (the semester total is {r['T']:g}, compulsory "
                             f"modules give {r['C']:g}), i.e. about {n / 2:g} two-credit elective(s)")
        if not parts:
            return None
        return {"answer": f"Semester {s} - " + "; ".join(parts) + ". This is worked out from the credit tables "
                          f"(p.40 and p.52); please confirm your elective choice with the faculty.",
                "kind": "logic_electives_needed", "confidence": 0.8}

    # ------------------------------------------------------------------ verification of engine answers
    def verify(self, question: str, result: dict[str, Any]) -> dict[str, Any]:
        """Cross-check grade and class answers with the rules; on a mismatch the Prolog answer wins."""
        kind, ans = result.get("kind", ""), result.get("answer", "")
        try:
            if kind == "grade_lookup":
                m = re.search(r"(\d{1,3}) marks", ans)
                g = re.search(r"grade " + GRADE, ans)
                if m and g:
                    r = self.pl.once(f"grade_of({m.group(1)}, G, P)")
                    if r and r["G"] != g.group(1):
                        result = dict(result, answer=f"{m.group(1)} marks = grade {r['G']} (GPV {r['P']:.2f}).",
                                      verified="corrected_by_prolog")
                    else:
                        result = dict(result, verified="prolog_ok")
            elif kind == "class_lookup":
                f = re.search(r"([0-4]\.\d{1,2})", ans)
                if f:
                    r = self.pl.once(f"best_class({f.group(1)}, 0, 0, 0, 0, C)")
                    expected = r["C"] if r else None
                    if expected and expected not in ans:
                        result = dict(result, verified="mismatch", prolog_answer=expected)
                    else:
                        result = dict(result, verified="prolog_ok")
        except PrologError:
            pass
        return result

    def consistency_report(self) -> list[str]:
        """Handbook checks that the rules can do on their own (run once at start-up)."""
        out = []
        for r in self.ask("credit_mismatch(P, S, T, Sum)"):
            out.append(f"{r['P']} semester {r['S']}: module table sums to {r['Sum']:g} GPA credits, "
                       f"summary table says {r['T']:g}")
        return out
