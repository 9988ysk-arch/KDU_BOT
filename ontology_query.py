"""
ontology_query.py - answers relational / multi-hop questions by running Prolog over the generated ontology (v11).

  question --(entity recognition: module, programme, department, staff, topic, semester)-->
  Prolog goal over data/handbook_ontology.pl + ontology_rules.pl --> answer + reasoning chain + page

Examples
  which department offers machine learning?            department_of_module('CS4042', D)
  who heads the department that offers CS4042?         head_for_module('CS4042', H)
  which degrees belong to the department headed by X?  degrees_of_head(staff_12, P)
  is CS4042 in both CS and SE?                          in_both('CS4042', prog_cs, prog_se)
  which semester 7 modules are only in SE?              only_in(M, prog_se, prog_cs, 7)
  how many modules do CS and SE share in semester 5?    shared_in_semester(M, prog_cs, prog_se, 5)
  who researches machine learning in the department that offers SE?
                                                        researcher_for_programme('machine learning', prog_se, S)
  does the computational mathematics department offer a degree?   department_without_degree(dept_cm)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from handbook_qa import DEPARTMENTS, PROGRAMS, normalize, programs_in
from ontology import Ontology, dept_atom, prog_atom, q
from prolog_engine import Prolog

RULES = Path(__file__).resolve().parent / "ontology_rules.pl"
OFFER_VERB = r"(offers?|offered|teach(es)?|runs?|owns?|conducts?|is responsible for|provides?|handles?)"


class OntologyQuery:
    def __init__(self, qa, ontology: Ontology | None = None, save: bool = True):
        self.qa = qa
        self.onto = ontology or Ontology(qa.kb)
        if save:
            try:
                self.onto.save()
            except OSError:
                pass
        self.pl = Prolog()
        self.pl.consult_text(self.onto.to_prolog())
        self.pl.consult(str(RULES))
        self.names = {r["X"]: r["N"] for r in self.pl.query("title(X, N)")}
        self.staff_by_name = {n: a for a, n in self.names.items() if str(a).startswith("staff_")}
        self.page = {}
        for k, f in self.onto.facts.items():
            self.page[k] = f.get("page")
        self.issues = [r["P"] for r in self.pl.query("programme_without_department(P)")]

    # ------------------------------------------------------------------ helpers
    def name(self, atom: Any) -> str:
        return str(self.names.get(atom, atom))

    def pg(self, *key: Any) -> str:
        p = self.page.get(tuple(key))
        return f"p. {p}" if p else ""

    def module_of(self, q_: str, ql: str) -> str | None:
        m = re.search(r"\b([A-Za-z]{2,3})\s?(\d{4})\b", q_)
        if m:
            code = (m.group(1) + m.group(2)).upper()
            if code in self.qa.courses:
                return code
        t = self.qa.match_course_title(ql)
        if t:
            return t[0]
        return self.qa.match_course_acronym(q_)

    def progs_of(self, q_: str) -> list[str]:
        ps = programs_in(q_)
        for p in ("CS", "SE", "CE", "IT", "IS"):
            if re.search(rf"\b{p}\b", q_) and p not in ps:
                ps.append(p)
        return ps

    def staff_of(self, ql: str) -> str | None:
        st = self.qa.match_person(ql)
        return self.staff_by_name.get(st["name"]) if st else None

    @staticmethod
    def semester_of(ql: str) -> int | None:
        m = re.search(r"\bsem(?:ester)?\s*0?([1-8])\b", ql)
        return int(m.group(1)) if m else None

    def reply(self, text: str, chain: str, kind: str, page: str = "") -> dict[str, Any]:
        src = f" ({page})" if page else ""
        return {"answer": f"{text}{src}\nHow I worked it out: {chain}.", "kind": kind, "source": "ontology+prolog",
                "confidence": 0.95}

    # ------------------------------------------------------------------ main
    def answer(self, question: str) -> dict[str, Any] | None:
        q_ = question.strip()
        ql = normalize(q_).lower()
        for h in (self.research_in_department, self.head_of_offering_department, self.degrees_of_head,
                  self.department_degree_check, self.module_in_both, self.only_in, self.shared,
                  self.department_of_module, self.programmes_of_module):
            try:
                r = h(q_, ql)
            except Exception:
                r = None
            if r:
                return r
        return None

    # 1. who researches T in the department that offers P / in department D
    def research_in_department(self, q_: str, ql: str) -> dict | None:
        m = re.search(r"\b(?:who|which (?:staff|lecturers?|people))\b.*?\b(?:research(?:es|ing)?(?: on| in)?|works? on|"
                      r"speciali[sz]es in|experts? in|interested in)\s+(.+?)\s+in (?:the )?(department\b.*|dept\b.*|"
                      r"[a-z ]+ department)\??$", ql)
        if not m:
            return None
        topic, where = m.group(1).strip(), m.group(2)
        dept = None
        how = ""
        if re.search(rf"\b(that|which) {OFFER_VERB}\b", where):
            progs = self.progs_of(where)
            mod = None if progs else self.module_of(where, where)
            if progs:
                r = self.pl.once(f"offers(D, {prog_atom(progs[0])})")
                dept = r["D"] if r else None
                how = f"{PROGRAMS[progs[0]]} is offered by the {self.name(dept)}"
            elif mod:
                r = self.pl.once(f"department_of_module({q(mod)}, D)")
                dept = r["D"] if r else None
                how = f"{mod} belongs to a degree offered by the {self.name(dept)}"
        else:
            code = self.qa.match_department(where)
            dept = dept_atom(code) if code else None
            how = f"department = {self.name(dept)}"
        if not dept:
            return None
        found = sorted({r["S"] for r in self.pl.query(f"researcher_in_department({q(topic)}, {dept}, S)")})
        if not found:
            return {"answer": f"No staff member of the {self.name(dept)} lists \"{topic}\" as a research area.\n"
                              f"How I worked it out: {how}; then searched research_topic facts of its members.",
                    "kind": "onto_research", "source": "ontology+prolog"}
        names = [self.name(s) for s in found]
        return self.reply(f"In the {self.name(dept)}, {len(names)} staff member{'s' if len(names) > 1 else ''} list "
                          f"{topic}: " + "; ".join(names) + ".",
                          f"{how}; researcher_in_department('{topic}', {dept}, S) over the staff research areas",
                          "onto_research")

    # 2. who heads the department that offers X
    def head_of_offering_department(self, q_: str, ql: str) -> dict | None:
        if not re.search(rf"\b(head|heads|headed|leads?|in charge)\b.*\bdepartment (that|which) {OFFER_VERB}\b", ql):
            return None
        tail = re.split(rf"\bdepartment (?:that|which) {OFFER_VERB}\b", ql)[-1]
        progs = self.progs_of(tail) or self.progs_of(re.split(r"\bdepartment\b", q_)[-1])
        mod = None if progs else self.module_of(q_, ql)
        if progs:
            r = self.pl.once(f"head_of_programme({prog_atom(progs[0])}, H)")
            d = self.pl.once(f"offers(D, {prog_atom(progs[0])})")
            if r and d:
                return self.reply(f"{self.name(r['H'])} heads the {self.name(d['D'])}, which offers "
                                  f"{PROGRAMS[progs[0]]}.", f"{PROGRAMS[progs[0]]} -> offered by {self.name(d['D'])} "
                                  f"-> head_of = {self.name(r['H'])}", "person_role", self.pg("head_of", r["H"], d["D"]))
        if mod:
            r = self.pl.once(f"head_for_module({q(mod)}, H)")
            d = self.pl.once(f"department_of_module({q(mod)}, D)")
            if r and d:
                return self.reply(f"{self.name(r['H'])} heads the {self.name(d['D'])}, which offers {mod} - "
                                  f"{self.name(mod)}.", f"{mod} -> in_programme -> offered by {self.name(d['D'])} -> "
                                  f"head_of = {self.name(r['H'])}", "person_role", self.pg("head_of", r["H"], d["D"]))
        return None

    # 3. degrees of the department headed by <person>
    def degrees_of_head(self, q_: str, ql: str) -> dict | None:
        if not re.search(r"\b(degrees?|programm?es?|courses?)\b.*\bdepartment (headed|led|run) by\b", ql):
            return None
        s = self.staff_of(ql)
        if not s:
            return None
        rows = self.pl.query(f"degrees_of_head({s}, P)")
        d = self.pl.once(f"head_of({s}, D)")
        if not d:
            return {"answer": f"{self.name(s)} is not listed as the head of a department in the handbook.",
                    "kind": "onto_degrees", "source": "ontology+prolog"}
        progs = [self.name(r["P"]) for r in rows]
        if not progs:
            return self.reply(f"{self.name(s)} heads the {self.name(d['D'])}, which does not offer a degree of its own.",
                              f"head_of({self.name(s)}, D) -> no offers(D, P) fact", "onto_degrees")
        return self.reply(f"{self.name(s)} heads the {self.name(d['D'])}, which offers " + " and ".join(progs) + ".",
                          f"head_of -> {self.name(d['D'])} -> offers", "onto_degrees",
                          self.pg("head_of", s, d["D"]))

    # 4. does department D offer a degree?
    def department_degree_check(self, q_: str, ql: str) -> dict | None:
        if not re.search(r"\b(does|do|is there)\b.*\bdepartment\b.*\b(offer|have|run)\b.*\b(degree|programm?e)s?\b|"
                         r"\bdepartment\b.*\bwithout (a )?degree\b", ql):
            return None
        code = self.qa.match_department(ql)
        if not code:
            return None
        d = dept_atom(code)
        # a specific degree named: "does the IT department offer the computer engineering degree?"
        dept_words = DEPARTMENTS[code].lower().replace("department of ", "")
        rest = re.sub(r"\b" + re.escape(dept_words) + r"\b|\b" + code.lower() + r" department\b", " ", ql)
        named = [p for p in self.progs_of(rest) if not (p == code and re.search(rf"\b{code.lower()} department\b", ql))]
        if not named:
            named = [p for p in ("CS", "SE", "CE", "IT", "IS") if re.search(rf"\bthe {p} degree\b", q_)]
        if named:
            p = named[0]
            pa = prog_atom(p)
            if self.pl.holds(f"offers({d}, {pa})"):
                return self.reply(f"Yes. The {self.name(d)} offers {PROGRAMS[p]}.", f"offers({d}, {pa}) holds",
                                  "onto_degrees")
            real = self.pl.once(f"offers(D, {pa})")
            return self.reply(f"No. {PROGRAMS[p]} is offered by the {self.name(real['D']) if real else 'another department'}, "
                              f"not the {self.name(d)}.", f"offers({d}, {pa}) fails; offers(D, {pa}) gives "
                              f"{self.name(real['D']) if real else '-'}", "onto_degrees")
        progs = [self.name(r["P"]) for r in self.pl.query(f"programmes_of_department({d}, P)")]
        if progs:
            return self.reply(f"Yes. The {self.name(d)} offers " + " and ".join(progs) + ".",
                              f"programmes_of_department({d}, P)", "onto_degrees")
        tf = [self.name(r["P"]) for r in self.pl.query(f"teaches_for({d}, P)")]
        extra = f" It teaches courses for {', '.join(tf)}." if tf else \
            " Its courses provide the mathematics and statistics needed by the faculty's degrees."
        return self.reply(f"No. The {self.name(d)} does not offer a degree of its own.{extra}",
                          f"department_without_degree({d}) holds: there is no offers({d}, P) fact", "onto_degrees")

    # 5. is M in both CS and SE / is M offered in SE?
    def module_in_both(self, q_: str, ql: str) -> dict | None:
        if not re.search(r"^(is|are|does)\b", ql) or not re.search(r"\b(both|in|for|part of|offered)\b", ql):
            return None
        mod = self.module_of(q_, ql)
        progs = self.progs_of(re.sub(r"\b[A-Za-z]{2,3}\s?\d{4}\b", " ", q_))
        if not mod or not progs or re.search(r"\b(compulsory|elective|optional|credits?|semester \d)\b", ql):
            return None
        have = [r["P"] for r in self.pl.query(f"programmes_of_module({q(mod)}, P)")]
        have_codes = [str(p).replace("prog_", "").upper() for p in have]
        asked = progs if len(progs) > 1 or "both" in ql else progs[:1]
        inside = [p for p in asked if p in have_codes]
        title = f"{mod} - {self.name(mod)}"
        where = " and ".join(have_codes) if have_codes else "no degree table"
        if len(inside) == len(asked):
            ans = f"Yes. {title} is in {' and '.join(asked)}."
        elif inside:
            ans = f"No - {title} is only in {' and '.join(inside)}, not in {' and '.join(p for p in asked if p not in inside)}."
        else:
            ans = f"No - {title} is not in {' or '.join(asked)}; it is in {where}."
        return self.reply(ans, f"programmes_of_module({mod}, P) gives {where}", "onto_membership",
                          self.pg("in_programme", mod, have[0]) if have else "")

    # 6. which (semester N) modules are only in P
    def only_in(self, q_: str, ql: str) -> dict | None:
        if not re.search(r"\bonly (in|for|offered in)\b|\bexclusive to\b|\bnot in (cs|se)\b", ql) or \
                not re.search(r"\b(modules?|courses?|subjects?)\b", ql):
            return None
        progs = [p for p in self.progs_of(q_) if p in ("CS", "SE")]
        if not progs:
            return None
        p = progs[0]
        other = "SE" if p == "CS" else "CS"
        sem = self.semester_of(ql)
        goal = f"only_in(M, {prog_atom(p)}, {prog_atom(other)}, {sem if sem else 'S'})"
        rows = self.pl.query(goal)
        mods = sorted({r["M"] for r in rows})
        when = f"Semester {sem}" if sem else "any semester"
        if not mods:
            return self.reply(f"No module in {when} is only in {p}.", goal, "onto_only")
        items = "; ".join(f"{m} {self.name(m)}" for m in mods)
        return self.reply(f"{len(mods)} module{'s' if len(mods) > 1 else ''} in {when} {'are' if len(mods) > 1 else 'is'} "
                          f"only in {p} (not in {other}): {items}.", goal + " (offered in one table, absent from the "
                          "other for the same semester)", "onto_only")

    # 7. shared modules of CS and SE in semester N
    def shared(self, q_: str, ql: str) -> dict | None:
        if not re.search(r"\b(share|shared|common|in common|same)\b", ql) or not re.search(r"\b(modules?|courses?|"
                                                                                             r"subjects?)\b", ql):
            return None
        progs = [p for p in self.progs_of(q_) if p in ("CS", "SE")]
        sem = self.semester_of(ql)
        if len(set(progs)) < 2 and not re.search(r"\bcs and se\b|\bse and cs\b", ql):
            return None
        goal = f"shared_in_semester(M, prog_cs, prog_se, {sem if sem else 'S'})"
        mods = sorted({r["M"] for r in self.pl.query(goal)})
        when = f"in Semester {sem}" if sem else "across all semesters"
        if re.search(r"\bhow many\b", ql):
            return self.reply(f"CS and SE share {len(mods)} modules {when}: " + ", ".join(mods) + ".", goal, "onto_shared")
        return self.reply(f"Modules CS and SE share {when} ({len(mods)}): " + "; ".join(f"{m} {self.name(m)}" for m in mods)
                          + ".", goal, "onto_shared")

    # 8. which department offers module M
    def department_of_module(self, q_: str, ql: str) -> dict | None:
        if not re.search(rf"\b(which|what) department\b.*\b{OFFER_VERB}\b|\bdepartment (of|for|behind)\b|"
                         rf"\b{OFFER_VERB} by which department\b|\bwhich department (is|does)\b", ql):
            return None
        if self.progs_of(re.sub(r"\b[A-Za-z]{2,3}\s?\d{4}\b", " ", q_)) or self.qa.match_person(ql):
            return None  # programme / staff questions: the engine answers those
        mod = self.module_of(q_, ql)
        if not mod:
            return None
        rows = self.pl.query(f"department_of_module({q(mod)}, D)")
        depts = sorted({r["D"] for r in rows})
        progs = sorted({str(r["P"]).replace("prog_", "").upper() for r in self.pl.query(f"in_programme({q(mod)}, P)")})
        if not depts:
            return None
        dn = " and ".join(f"the {self.name(d)}" for d in depts)
        return self.reply(f"{mod} - {self.name(mod)} is offered by {dn} (it is part of {' and '.join(progs)}).",
                          f"{mod} -> in_programme {'/'.join(progs)} -> offers -> {'/'.join(self.name(d) for d in depts)}",
                          "onto_department", self.pg("in_programme", mod, prog_atom(progs[0])) if progs else "")

    # 9. which degrees include module M
    def programmes_of_module(self, q_: str, ql: str) -> dict | None:
        if not re.search(r"\b(which|what) (degrees?|programm?es?)\b.*\b(include|includes|have|has|contain|offer|teach)\b|"
                         r"\bin which (degrees?|programm?es?)\b", ql):
            return None
        mod = self.module_of(q_, ql)
        if not mod:
            return None
        rows = self.pl.query(f"offered({q(mod)}, P, S)")
        if not rows:
            return None
        parts = [f"{PROGRAMS[str(r['P']).replace('prog_', '').upper()]} (Semester {r['S']})" for r in rows]
        return self.reply(f"{mod} - {self.name(mod)} is in: " + "; ".join(parts) + ".",
                          f"offered({mod}, P, S)", "onto_programmes")
