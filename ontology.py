"""
ontology.py - automatic ontology generation from the Student Handbook PDF extraction (v11, no LLM).

Schema (designed by hand, populated automatically every time a handbook is indexed):

  Classes     Faculty, Department, Programme, Module, Staff, Grade, ClassAward, ResearchTopic
  Relations   offers(Department, Programme)             head_of(Staff, Department)
              dean_of(Staff, Faculty)                    member_of(Staff, Department)
              teaches_for(Department, Programme)         (service teaching: "courses for BSc X")
              in_programme(Module, Programme)            offered(Module, Programme, Semester)
              module_level(Module, Programme, Level)     category(Module, Programme, Compulsory|Elective)
              gpa_credits / ngpa_credits / mgpa_credits(Module, Programme, Credits)
              title(Entity, Text)                        email(Staff, Text)
              research_topic(Staff, Topic)               grade_status(Grade, pass|weak_pass|fail|excused)
              class_min(ClassAward, FGPA)

Sources, each with a confidence:
  table   1.00  module tables, staff list, grade table, class table (pdfplumber tables)
  role    0.95  staff role strings ("Head - Department of Information Technology")
  text    0.85  sentence patterns in the department sections ("This department offers ... BSc (Hons) in X")
  derived 0.90  research topics: 1-3 word phrases split from each staff member's research-area text

Every fact is also stored as prov(Fact, Page, Source, Confidence). When a table fact and a text-pattern fact
agree, the fact is marked confirmed. A validation pass checks the ontology for consistency (each programme
offered by exactly one department, one head per department, every module offering points to a known
programme, every member's department exists, text and table sources agree).

Outputs: data/handbook_ontology.pl (Prolog facts, also loadable in SWI-Prolog),
         data/handbook_ontology.ttl (RDF/Turtle, opens in Protege), data/ontology_report.json (validation).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from handbook_qa import DEPARTMENTS, PROGRAMS, STOP

DATA = Path(__file__).resolve().parent / "data"
CONF = {"table": 1.0, "role": 0.95, "text": 0.85, "derived": 0.9}
PROG_PATTERNS = {"CS": r"computer science", "SE": r"software engineering", "CE": r"computer engineering",
                 "IT": r"information technology", "IS": r"information systems"}
TOPIC_STOP = STOP | {"and", "of", "in", "for", "with", "the", "based", "using", "systems", "system", "related",
                     "areas", "area", "applications", "application", "etc", "methods", "tools", "process"}


def q(x: Any) -> str:
    """Prolog atom / number."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return f"{x:g}" if isinstance(x, float) else str(x)
    return "'" + str(x).replace("\\", "\\\\").replace("'", "\\'") + "'"


def dept_atom(code: str) -> str:
    return f"dept_{code.lower()}"


def prog_atom(code: str) -> str:
    return f"prog_{code.lower()}"


class Ontology:
    def __init__(self, kb: dict[str, Any]):
        self.kb = kb
        self.facts: dict[tuple, dict[str, Any]] = {}  # (pred, args...) -> {page, sources:set, conf}
        self.issues: list[str] = []
        self.build()
        self.validate()

    # ------------------------------------------------------------------ population
    def add(self, pred: str, *args: Any, page: int | None = None, source: str = "table") -> None:
        key = (pred, *args)
        f = self.facts.setdefault(key, {"page": page, "sources": set(), "conf": 0.0})
        f["sources"].add(source)
        f["conf"] = max(f["conf"], CONF[source])
        if f["page"] is None:
            f["page"] = page
        if len(f["sources"]) > 1:  # two independent sources agree
            f["conf"] = 1.0

    def build(self) -> None:
        kb = self.kb
        self.add("type", "faculty_of_computing", "Faculty", page=1)
        self.add("title", "faculty_of_computing", "Faculty of Computing", page=1)
        # departments (table + text patterns in each department section)
        for code, d in kb.get("departments", {}).items():
            da = dept_atom(code)
            self.add("type", da, "Department", page=d.get("page"))
            self.add("title", da, d.get("name") or DEPARTMENTS.get(code, code), page=d.get("page"))
            self.add("part_of", da, "faculty_of_computing", page=d.get("page"))
            for p in d.get("degrees") or []:
                self.add("offers", da, prog_atom(p), page=d.get("page"), source="table")
        self._offers_from_text()
        # programmes
        for code, name in PROGRAMS.items():
            pa = prog_atom(code)
            info = kb.get("programmes", {}).get(code, {}) or {}
            page = info.get("page") if isinstance(info, dict) else None
            self.add("type", pa, "Programme", page=page)
            self.add("title", pa, name, page=page)
            self.add("short_name", pa, code, page=page)
            dur = kb.get("max_duration", {}).get(code)
            if dur:
                self.add("max_years", pa, int(dur), page=None)
        # modules
        for code, c in kb.get("courses", {}).items():
            self.add("type", code, "Module")
            self.add("title", code, c["title"])
            for o in c.get("offerings", []):
                pa, page = prog_atom(o["program"]), o.get("page")
                self.add("in_programme", code, pa, page=page)
                if o.get("semester"):
                    self.add("offered", code, pa, int(o["semester"]), page=page)
                if o.get("level"):
                    self.add("module_level", code, pa, int(o["level"]), page=page)
                if o.get("category"):
                    self.add("category", code, pa, o["category"].lower(), page=page)
                for k in ("gpa", "ngpa", "mgpa"):
                    if o.get(k):
                        self.add(f"{k}_credits", code, pa, float(o[k]), page=page)
        # staff (+ roles as a second source for heads / dean)
        for i, st in enumerate(kb.get("staff", [])):
            sa = f"staff_{i}"
            page = st.get("page")
            self.add("type", sa, "Staff", page=page)
            self.add("title", sa, st["name"], page=page)
            if st.get("designation"):
                self.add("designation", sa, st["designation"], page=page)
            if st.get("department"):
                self.add("member_of", sa, dept_atom(st["department"]), page=page)
            if st.get("email"):
                self.add("email", sa, st["email"], page=page)
            role = (st.get("role") or "").lower()
            if "head" in role:
                d = next((c for c, n in DEPARTMENTS.items() if n.lower().replace("department of ", "") in role), None)
                if d:
                    self.add("head_of", sa, dept_atom(d), page=page, source="role")
            if "dean" in role:
                self.add("dean_of", sa, "faculty_of_computing", page=page, source="role")
            for t in self.topics(st.get("research_areas") or ""):
                self.add("research_topic", sa, t, page=page, source="derived")
        # grades and classes
        for r in kb.get("grades", {}).get("rows", []):
            g = r["grade"]
            status = ("excused" if g == "Ex" else "fail" if r.get("gpv") in ("0.00", "0") else
                      "weak_pass" if g in ("C-", "D+") else "pass")
            self.add("type", g, "Grade", page=kb["grades"].get("page"))
            self.add("grade_status", g, status, page=kb["grades"].get("page"))
        for r in kb.get("classes", {}).get("rows", []):
            self.add("type", r["result"], "ClassAward", page=kb["classes"].get("page"))
            self.add("class_min", r["result"], float(r["low"]), page=kb["classes"].get("page"))

    def _offers_from_text(self) -> None:
        """Pattern extraction: department sections say which degrees they offer."""
        for sec in self.kb.get("sections", []):
            title = sec.get("title", "").lower()
            dcode = next((c for c, n in DEPARTMENTS.items() if n.lower() == title), None)
            if not dcode or not str(sec.get("number", "")).startswith("1."):  # descriptions, not staff lists
                continue
            text = re.sub(r"\s+", " ", sec.get("text", "")).lower()
            for m in re.finditer(r"\b(?:offers?|offering)\b(?=(.{0,220}))", text):
                span = re.split(r"(?<=[a-z)])\.\s", m.group(1))[0]  # stay inside the sentence
                # "offers ... courses FOR BSc X" is service teaching, not offering the degree
                rel = "teaches_for" if re.search(r"courses? for bsc", span) else "offers"
                for p, pat in PROG_PATTERNS.items():
                    if re.search(r"bsc \(hons\) in " + pat, span):
                        self.add(rel, dept_atom(dcode), prog_atom(p), page=sec.get("page"), source="text")

    @staticmethod
    def topics(text: str) -> list[str]:
        """Split a research-area string into 1-3 word topic phrases ("Deep Learning Machine Learning" ->
        deep learning, machine learning, learning ...)."""
        words = [w for w in re.findall(r"[a-z][a-z\-]+", text.lower())]
        out: set[str] = set()
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                ph = words[i:i + n]
                if ph[0] in TOPIC_STOP or ph[-1] in TOPIC_STOP or (n == 1 and len(ph[0]) < 4):
                    continue
                out.add(" ".join(ph))
        return sorted(out)

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        offers = defaultdict(list)
        heads = defaultdict(list)
        for k, f in self.facts.items():
            if k[0] == "offers":
                offers[k[2]].append(k[1])
                if f["sources"] == {"text"}:
                    self.issues.append(f"offers({k[1]}, {k[2]}) found only in text, not in the department table")
                if f["sources"] == {"table"}:
                    self.issues.append(f"offers({k[1]}, {k[2]}) not confirmed by the department text")
            if k[0] == "head_of":
                heads[k[2]].append(k[1])
        for p in PROGRAMS:
            n = len(offers.get(prog_atom(p), []))
            if n != 1:
                self.issues.append(f"programme {p} is offered by {n} departments (expected 1)")
        for d, hs in heads.items():
            if len(hs) > 1:
                self.issues.append(f"{d} has {len(hs)} heads")
        progs = {prog_atom(p) for p in PROGRAMS}
        depts = {k[1] for k in self.facts if k[0] == "type" and k[2] == "Department"}
        for k in self.facts:
            if k[0] == "in_programme" and k[2] not in progs:
                self.issues.append(f"{k[1]} points to unknown programme {k[2]}")
            if k[0] == "member_of" and k[2] not in depts:
                self.issues.append(f"{k[1]} is a member of unknown department {k[2]}")
        for d in depts:
            if not heads.get(d):
                self.issues.append(f"{d} has no head")

    # ------------------------------------------------------------------ export
    def stats(self) -> dict[str, int]:
        return dict(Counter(k[0] for k in self.facts))

    def to_prolog(self) -> str:
        lines = ["% handbook_ontology.pl - generated automatically from the Student Handbook PDF (ontology.py).",
                 "% prov(Fact, Page, Source, Confidence) records where every fact came from.",
                 ":- discontiguous(prov/4)." if False else ""]
        by_pred: dict[str, list] = defaultdict(list)
        for k, f in self.facts.items():
            by_pred[k[0]].append((k, f))
        for pred in sorted(by_pred):
            for k, f in sorted(by_pred[pred], key=lambda x: str(x[0])):
                args = ", ".join(q(a) if not re.fullmatch(r"(dept|prog|staff)_\w+|faculty_of_computing", str(a))
                                 else str(a) for a in k[1:])
                lines.append(f"{pred}({args}).")
        for k, f in self.facts.items():
            args = ", ".join(q(a) if not re.fullmatch(r"(dept|prog|staff)_\w+|faculty_of_computing", str(a))
                             else str(a) for a in k[1:])
            src = "+".join(sorted(f["sources"]))
            lines.append(f"prov({k[0]}({args}), {q(f['page'] or 0)}, {q(src)}, {f['conf']:.2f}).")
        return "\n".join(x for x in lines if x is not None) + "\n"

    def to_turtle(self) -> str:
        def iri(x: Any) -> str:
            return "hb:" + re.sub(r"[^A-Za-z0-9_]", "_", str(x))

        def lit(x: Any) -> str:
            if isinstance(x, (int, float)):
                return f"\"{x}\"^^xsd:decimal"
            return '"' + str(x).replace("\\", "\\\\").replace('"', '\\"') + '"'
        out = ["@prefix hb: <http://kdu.ac.lk/foc/handbook#> .", "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
               "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .", "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
               "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .", ""]
        for c in ("Faculty", "Department", "Programme", "Module", "Staff", "Grade", "ClassAward"):
            out.append(f"hb:{c} a owl:Class .")
        for p in ("offers", "teaches_for", "head_of", "dean_of", "member_of", "in_programme", "part_of"):
            out.append(f"hb:{p} a owl:ObjectProperty .")
        out.append("")
        for k in self.facts:
            pred, args = k[0], k[1:]
            if pred == "type":
                out.append(f"{iri(args[0])} a hb:{args[1]} .")
            elif pred in ("title",):
                out.append(f"{iri(args[0])} rdfs:label {lit(args[1])} .")
            elif len(args) == 2 and pred in ("offers", "teaches_for", "head_of", "dean_of", "member_of", "in_programme", "part_of"):
                out.append(f"{iri(args[0])} hb:{pred} {iri(args[1])} .")
            elif len(args) == 2:
                out.append(f"{iri(args[0])} hb:{pred} {lit(args[1])} .")
            elif pred == "offered":
                out.append(f"{iri(args[0])} hb:offered_in_{args[1]} {lit(args[2])} .")
        return "\n".join(out) + "\n"

    def save(self, folder: Path = DATA) -> dict[str, Any]:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "handbook_ontology.pl").write_text(self.to_prolog(), encoding="utf-8")
        (folder / "handbook_ontology.ttl").write_text(self.to_turtle(), encoding="utf-8")
        confirmed = sum(1 for f in self.facts.values() if len(f["sources"]) > 1)
        report = {"facts": len(self.facts), "confirmed_by_two_sources": confirmed, "by_relation": self.stats(),
                  "issues": self.issues}
        (folder / "ontology_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        return report


if __name__ == "__main__":
    import sys
    from handbook_qa import load_kb
    kb = load_kb(sys.argv[1] if len(sys.argv) > 1 else DATA / "handbook_kb.json")
    rep = Ontology(kb).save()
    print(json.dumps(rep, indent=1))
