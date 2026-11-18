"""
handbook_qa.py  -  CPU-only, non-LLM question answering over a Student Handbook PDF.

Architecture (three layers, no generative model, no neural embeddings, no GPU):

  1. Document understanding (offline, runs once per upload)
       PDF -> pages -> section tree (from the Table of Contents)
           -> tables (pdfplumber) -> typed records (courses, grades, classes,
              programmes, staff, appointments, contacts, glossary)
           -> sentence/list-item passages for retrieval
       Everything is saved as one JSON knowledge base (KB).

  2. Semantic parsing over the KB (pattern-based NLIDB, cf. Affolter et al. 2019)
       question -> normalisation + spelling correction
                -> entity linking (course codes/titles, programmes, semesters,
                   grades, marks, FGPA values, staff names, roles, departments)
                -> intent handlers that query the structured records and
                   build short, exact answers (counts, lists, numbers).

  3. Retrieval fallback (BM25F, Robertson & Zaragoza 2009)
       sentence-level BM25 with a section-title field, domain query expansion,
       answer-type re-ranking, and extractive short answers.

Only the Python standard library + pdfplumber are required.
pytesseract is optional (used only for image-only pages).
"""

from __future__ import annotations

import difflib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

try:
    import pytesseract  # optional OCR for image-only pages
except Exception:  # pragma: no cover
    pytesseract = None


KB_VERSION = 3

# ======================================================================
# 1. TEXT UTILITIES
# ======================================================================

STOP = set("""
a an the is are was were be been being am to of in on at for from by with and or if then than that this these those
it its as do does did can could would should must may might will shall what which who whom whose when where why how
i we you they he she their our your his her me my us them about into within through per such any all each both
please tell give need want know there here so also just get got some much many tell show list name
after before during since until without under over between upon via against
gets getting gotten
""".split())

# Light suffix stemmer (Porter-like, enough for matching "credits"->"credit", "offered"->"offer").
_SUFFIXES = ("ational", "ization", "fulness", "ousness", "iveness", "ments", "ment", "ings", "ing",
             "edly", "ies", "ied", "ers", "er", "ed", "es", "s")


def stem(word: str) -> str:
    w = word.lower()
    if len(w) <= 3 or w.isdigit() or re.match(r"^[a-z]{2,3}\d{4}$", w):
        return w
    if len(w) > 7:
        w = w.replace("isation", "ization").replace("ising", "izing").replace("ised", "ized")
    for suf in ("ization", "izing", "izes", "ized", "ize"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)] + "iz"
    for suf in _SUFFIXES:
        if suf == "s" and (w.endswith("ss") or w.endswith("us") or w.endswith("is")):
            continue  # class, pass, campus, analysis keep their final s
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            if suf in ("ies", "ied"):
                base += "y"
            if suf in ("ing", "ed", "er", "ers", "ings") and len(base) > 4 and base[-1] == base[-2] \
                    and base[-1] not in "lsz":
                base = base[:-1]  # programming -> programm -> program
            w = base
            break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]  # include/includes -> includ, course/courses -> cours
    return w


ABBREVIATIONS = {r"\bmin\b": "minimum", r"\bmax\b": "maximum", r"\bdept\b": "department",
                 r"\bdepts\b": "departments", r"\breq\b": "requirement", r"\bqualifs?\b": "qualifications",
                 r"\bexams\b": "exam", r"\binfo\b": "information", r"\buni\b": "university",
                 r"\bdvc\b": "deputy vice chancellor", r"\bhons\b": "honours", r"\bfoc\b": "faculty of computing"}


# Singlish (Sinhala written in Latin letters) question words -> English equivalents
SINGLISH = {
    r"\bkeeyak\b|\bkochchara\b|\bkochchama\b": "how much",
    r"\bkeeyada\b|\bkiyada\b|\bkeeya\b|\bkiyak\b": "how many",
    r"\bliyanna\b|\bliyanne\b": "sit the exam",
    r"\bthiyenne\b|\btiyenne\b": "located",
    r"\bmokakda\b|\bmonawada\b|\bmokadda\b|\bmokak\b": "what",
    r"\bkauda\b|\bkawda\b|\bkawuda\b": "who",
    r"\bkoheda\b|\bkohenda\b": "where",
    r"\bkawadada\b|\bkeeyatada\b": "when",
    r"\bkohomada\b": "how",
    r"\bwenna one\b|\bonada\b|\bone\s+(?=\?|$)": "needed",
    r"(?<=\bhow much )(\w+ )?one\b": r"\1needed",
    r"\bthiyenawada\b|\btiyenawada\b": "is there",
    r"\bganna\b|\bgannakota\b": "get",
    r"\bkiyanne\b|\bkiyana\b": "means",
    r"\bdenna\b|\bpennanna\b": "show",
    r"\bwala\b|\bwalata\b": "in",
    r"\beke\b|\beka\b|\bekak\b|\bda\b|\bthiyena\b|\bthiyana\b|\bpuluwanda\b": "",
}


def expand_abbreviations(text: str) -> str:
    for pat, full in ABBREVIATIONS.items():
        text = re.sub(pat, full, text, flags=re.I)
    text = re.sub(r"\bre-?s[ai]t(?:ting|s)?\b|\bresit\w*", "re-sitting", text, flags=re.I)
    for pat, eng in SINGLISH.items():
        text = re.sub(pat, eng, text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = t.replace("–", "-").replace("—", "-").replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"').replace("&", " and ")
    t = re.sub(r"\bb\.?\s?sc\.?\b", "bsc", t, flags=re.I)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


TOKEN_RE = re.compile(r"[a-z]{2,4}\d{4}|\d+(?:\.\d+)?|[a-z]+[+-]?", re.I)


def tokens(text: str, drop_stop: bool = True) -> list[str]:
    out = []
    for tok in TOKEN_RE.findall(normalize(text).lower()):
        tok = tok.rstrip("+-") if not re.fullmatch(r"[a-d][+-]", tok) else tok
        if drop_stop and tok in STOP:
            continue
        out.append(tok)
    return out


_COMPARATIVES = [
    (re.compile(r"\bnot less than\b|\bat least\b|\bno less than\b|>=|\u2265", re.I), " minimum "),
    (re.compile(r"\bnot more than\b|\bat most\b|\bno more than\b|\bup to\b|<=|\u2264", re.I), " maximum "),
]


def stems(text: str, drop_stop: bool = True) -> list[str]:
    """Stemmed index terms. Comparison phrases are canonicalised so that
    'not less than 2.00' also matches a question about the 'minimum'."""
    for rx, canon in _COMPARATIVES:
        if rx.search(text or ""):
            text = f"{text} {canon}"
    ints = re.findall(r"\b(\d+)\.0+\b", text or "")
    if ints:
        text = f"{text} {' '.join(ints)}"  # "2.0" also matches a question that says "2"
    return [stem(t) for t in tokens(text, drop_stop)]


# Domain thesaurus for query expansion (Azad & Deepak 2019: manual/ontology-based QE).
THESAURUS: dict[str, list[str]] = {
    "course": ["module", "subject", "unit"],
    "module": ["course", "subject", "unit"],
    "subject": ["course", "module"],
    "degree": ["programme", "program"],
    "program": ["programme", "degree"],
    "programme": ["program", "degree"],
    "exam": ["examination"],
    "examination": ["exam"],
    "attendance": ["attend"],
    "fail": ["failure"],
    "repeat": ["resit", "re-sit", "re-sitting"],
    "resit": ["re-sitting", "repeat"],
    "fee": ["charge", "levied"],
    "pay": ["charge", "levied"],
    "kick": ["discontinue", "discontinuation"],
    "dismiss": ["discontinue", "discontinuation"],
    "remove": ["discontinue", "discontinuation"],
    "warning": ["warn"],
    "medical": ["medical", "illness"],
    "sick": ["medical", "illness", "excused", "excuse"],
    "length": ["duration"],
    "long": ["duration", "years"],
    "eligible": ["eligibility", "requirement"],
    "requirement": ["eligibility", "minimum"],
    "entry": ["admission"],
    "enter": ["admission"],
    "join": ["admission"],
    "phone": ["tel", "telephone"],
    "telephone": ["tel"],
    "mail": ["email", "e-mail"],
    "email": ["e-mail"],
    "address": ["postal", "road"],
    "website": ["web", "page", "www"],
    "boss": ["head", "dean"],
    "hod": ["head"],
    "teacher": ["lecturer", "instructor"],
    "lecturer": ["lecturer", "instructor"],
    "job": ["career", "opportunity"],
    "career": ["opportunity", "job"],
    "award": ["awarding", "trophy"],
    "prize": ["trophy", "award"],
    "best": ["trophy", "highest"],
    "supplementary": ["supplementary"],
    "class": ["class", "fgpa"],
    "military": ["military", "cadet"],
    "mark": ["marks", "grade"],
    "established": ["establish", "founded"],
    "founded": ["established"],
    "start": ["established", "begin", "commence"],
    "limit": ["minimum", "maximum", "less", "more"],
    "minimum": ["least", "less"],
    "maximum": ["more"],
    "graduate": ["degree", "awarding", "graduation"],
    "graduation": ["degree", "awarding"],
    "calculate": ["formula", "calculated"],
    "happen": ["shall"],
    "kdu": ["university", "defence"],
    "activity": ["includes", "work"],
    "specializ": ["specialization"],
    "fac": ["faculty"],
    "message": ["dean"],
    "qualification": ["requirement", "pass", "qualified"],
    "approve": ["approval"],
    "approv": ["approval"],
    "locat": ["road", "address", "campus"],
    "located": ["road", "address", "campus"],
    "count": ["considered", "considering"],
    "counted": ["considered"],
    "overall": ["entire"],
    "extend": ["extend", "extension"],
    "trophy": ["trophy", "award"],
    "choose": ["specialization", "select"],
    "miss": ["absent", "absence", "excused"],
    "compulsory": ["requirement", "eligibility", "mandatory"],
    "below": ["less", "minimum"],
    "under": ["less", "minimum"],
    "boss": ["head", "dean", "chancellor"],
    "online": ["web", "website"],
    "cheat": ["offence", "offences", "punishment"],
    "cheating": ["offence", "offences", "punishment"],
    "copy": ["offence", "offences"],
    "absence": ["absent"],
    "clear": ["complete", "completed"],
}


# ======================================================================
# 2. DOCUMENT UNDERSTANDING  (PDF -> structured knowledge base)
# ======================================================================

PAGE_FOOTER_RE = re.compile(r"^\s*(Page\s*\|\s*[A-Z0-9]+|\d{1,3})\s*$")
TOC_LINE_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.+?)\s*\.{3,}\s*(\d+)\s*$")
CODE_RE = re.compile(r"\b([A-Z]{2,3}\d{4})\b")
SEM_ROW_RE = re.compile(r"semester\s*0?(\d)", re.I)

PROGRAMS = {
    "CS": "BSc (Hons) in Computer Science",
    "SE": "BSc (Hons) in Software Engineering",
    "CE": "BSc (Hons) in Computer Engineering",
    "IT": "BSc (Hons) in Information Technology",
    "IS": "BSc (Hons) in Information Systems",
}
PROGRAM_PATTERNS = {
    "CS": [r"computer science", r"\bcs\b"],
    "SE": [r"software engineering", r"\bse\b"],
    "CE": [r"computer engineering", r"\bce\b"],
    "IT": [r"information technology"],
    "IS": [r"information systems?"],
}
DEPARTMENTS = {
    "CS": "Department of Computer Science",
    "IT": "Department of Information Technology",
    "CE": "Department of Computer Engineering",
    "CM": "Department of Computational Mathematics",
}
DEPT_PATTERNS = {
    "CS": [r"computer science", r"\bcs\b"],
    "IT": [r"information technology", r"\bit\b"],
    "CE": [r"computer engineering", r"\bce\b"],
    "CM": [r"computational math", r"\bmath", r"\bcm\b", r"statistic"],
}


def _clean_cell(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v or "").replace("\n", " ")).strip()


def _num(v: str) -> float | None:
    v = (v or "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", v):
        return float(v)
    return None


def _first_int(v: str) -> int | None:
    m = re.match(r"\s*(\d+)", v or "")
    return int(m.group(1)) if m else None


def programs_in(text: str) -> list[str]:
    t = normalize(text).lower()
    found = []
    for code, pats in PROGRAM_PATTERNS.items():
        if any(re.search(p, t) for p in pats):
            found.append(code)
    return found


class HandbookExtractor:
    """Turns the PDF into a JSON-serialisable knowledge base."""

    def __init__(self, pdf_path: str | Path, enable_ocr: bool = True):
        self.pdf_path = Path(pdf_path)
        self.enable_ocr = enable_ocr and pytesseract is not None

    # ---------------- page text ----------------
    def run(self) -> dict[str, Any]:
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is required: pip install pdfplumber")
        pages: list[str] = []
        raw_tables: list[dict[str, Any]] = []
        word_pages: dict[int, list[dict[str, Any]]] = {}
        ocr_pages: dict[int, str] = {}
        with pdfplumber.open(str(self.pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                lines = [ln for ln in text.splitlines() if not PAGE_FOOTER_RE.match(ln)]
                if len(text.strip()) < 150 and page.images and self.enable_ocr:
                    try:
                        img = page.to_image(resolution=200).original
                        ocr_pages[i] = re.sub(r"\s+", " ", pytesseract.image_to_string(img))
                    except Exception:
                        pass
                pages.append("\n".join(lines))
                for t in page.extract_tables() or []:
                    rows = [[_clean_cell(c) for c in r] for r in t if r]
                    rows = [r for r in rows if any(r)]
                    if rows:
                        raw_tables.append({"page": i, "rows": rows})
                word_pages[i] = page.extract_words(keep_blank_chars=False)

        kb: dict[str, Any] = {"version": KB_VERSION, "document": self.pdf_path.name, "page_count": len(pages),
                              "ocr_pages": ocr_pages}  # noisy OCR kept for audit, not used for answers
        toc = self._parse_toc(pages)
        kb["toc"] = toc
        kb["sections"] = self._segment_sections(pages, toc)
        kb["tables"] = self._caption_tables(raw_tables, pages, kb["sections"])
        kb["courses"] = self._parse_courses(kb["tables"], kb["sections"], pages)
        kb["semester_totals"] = self._parse_semester_totals(kb["tables"])
        kb["gpa_summary"] = self._parse_gpa_summary(kb["tables"])
        kb["grades"] = self._parse_grades(kb["tables"])
        kb["classes"] = self._parse_classes(kb["tables"])
        kb["programmes"] = self._parse_programmes(kb["tables"], kb["sections"])
        kb["heights"] = self._parse_heights(kb["tables"])
        kb["max_duration"] = self._parse_max_duration(kb["tables"])
        kb["staff"] = self._parse_staff(kb["sections"])
        kb["departments"] = self._parse_departments(kb["sections"])
        kb["appointments"] = self._parse_appointments(word_pages, kb["sections"])
        kb["contacts"] = self._parse_contacts(pages)
        kb["glossary"] = self._parse_glossary(pages)
        kb["passages"] = self._build_passages(kb)
        return kb

    # ---------------- table of contents ----------------
    @staticmethod
    def _parse_toc(pages: list[str]) -> list[dict[str, Any]]:
        toc = []
        for text in pages[:12]:
            for ln in text.splitlines():
                m = TOC_LINE_RE.match(ln.strip())
                if m:
                    num, title, pg = m.groups()
                    toc.append({"number": num, "title": title.strip(" ."), "page": int(pg)})
        return toc

    # ---------------- section tree ----------------
    def _segment_sections(self, pages: list[str], toc: list[dict[str, Any]]) -> list[dict[str, Any]]:
        toc_by_num = {t["number"]: t for t in toc}
        toc_start_page = None
        for i, t in enumerate(pages):
            if "table of contents" in t.lower():
                toc_start_page = i + 1
                break
        # Body starts after the TOC pages (TOC lines contain dotted leaders).
        body_start = 1
        if toc_start_page:
            body_start = toc_start_page
            while body_start < len(pages) and re.search(r"\.{5,}", pages[body_start - 1]):
                body_start += 1

        sections: list[dict[str, Any]] = [{
            "number": "0", "title": "Front matter", "level": 0, "parent": None,
            "page": 1, "lines": [],
        }]
        for p in range(1, body_start):
            sections[0]["lines"] += [(p, ln) for ln in pages[p - 1].splitlines() if not re.search(r"\.{5,}", ln)]

        heading_re = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.{2,90})$")
        current = None
        for p in range(body_start, len(pages) + 1):
            for ln in pages[p - 1].splitlines():
                s = ln.strip()
                m = heading_re.match(s)
                is_heading = False
                if m:
                    num, title = m.groups()
                    t = toc_by_num.get(num)
                    if t and difflib.SequenceMatcher(None, normalize(t["title"]).lower(),
                                                     normalize(title).lower()).ratio() > 0.6:
                        is_heading = True
                        title = t["title"]
                    elif not toc and len(num.split(".")) <= 3 and title[:1].isupper():
                        is_heading = True
                if is_heading:
                    level = len(num.split("."))
                    parent = ".".join(num.split(".")[:-1]) or None
                    current = {"number": num, "title": title.strip(), "level": level,
                               "parent": parent, "page": p, "lines": []}
                    sections.append(current)
                    continue
                if current is None:
                    current = {"number": "0.1", "title": "Introduction", "level": 1,
                               "parent": None, "page": p, "lines": []}
                    sections.append(current)
                current["lines"].append((p, s))
        for s in sections:
            s["text"] = "\n".join(l for _, l in s["lines"])
            s["pages"] = sorted({p for p, _ in s["lines"]} or {s["page"]})
            s["children"] = [c["number"] for c in sections if c.get("parent") == s["number"]]
            del s["lines"]
        return sections

    @staticmethod
    def _section_for_page(sections: list[dict[str, Any]], page: int, prefer_deepest: bool = True) -> dict[str, Any]:
        cand = [s for s in sections if page in s.get("pages", [])]
        if not cand:
            cand = [s for s in sections if s["page"] <= page] or sections
            return cand[-1]
        return cand[0] if not prefer_deepest else sorted(cand, key=lambda s: (s["page"]))[0]

    # ---------------- tables ----------------
    def _caption_tables(self, raw_tables, pages, sections):
        out = []
        for idx, t in enumerate(raw_tables):
            text = pages[t["page"] - 1]
            caps = re.findall(r"(Table\s*\d+\s*:?[^\n]*)", text)
            caption = caps[0].strip() if len(caps) == 1 else ""
            if len(caps) > 1:
                # Pick the caption whose words overlap most with the header row.
                head = " ".join(t["rows"][0]).lower()
                caption = max(caps, key=lambda c: len(set(stems(c)) & set(stems(head))))
            sec = self._section_for_page(sections, t["page"])
            # Find which numbered section owns the caption (the last heading before the table on that page)
            owners = [s for s in sections if t["page"] in s.get("pages", []) and s["number"] != "0"]
            owner = owners[-1] if owners else sec
            for s in owners:
                if caption and caption.split()[0:2] and caption[:12] in s.get("text", ""):
                    owner = s
            out.append({"id": idx, "page": t["page"], "caption": caption, "rows": t["rows"],
                        "section": owner["number"], "section_title": owner["title"]})
        return out

    @staticmethod
    def _is_course_table(t) -> bool:
        head = " ".join(" ".join(r) for r in t["rows"][:2]).lower()
        return "module" in head and "code" in head

    def _parse_courses(self, tables, sections, pages):
        sec_by_num = {s["number"]: s for s in sections}
        courses: dict[str, dict[str, Any]] = {}

        def owning_level_section(num: str) -> dict[str, Any]:
            # Walk up to the "Level" section (e.g. 3.3.1 BSc (Hons) in CS and BSc (Hons) in SE (Level 1)).
            s = sec_by_num.get(num)
            while s and "level" not in s["title"].lower() and s.get("parent"):
                s = sec_by_num.get(s["parent"])
            return s or sec_by_num.get(num, {"title": ""})

        for t in tables:
            if not self._is_course_table(t):
                continue
            lvl_sec = owning_level_section(t["section"])
            progs = [p for p in programs_in(lvl_sec["title"]) if p in ("CS", "SE", "CE", "IT", "IS")] or ["CS"]
            lv = re.search(r"level\s*(\d)", lvl_sec["title"], re.I)
            level = int(lv.group(1)) if lv else None
            semester = None
            for r in t["rows"]:
                joined = " ".join(r)
                sm = SEM_ROW_RE.search(joined)
                if sm and not CODE_RE.search(joined) and "total" not in joined.lower():
                    semester = int(sm.group(1))
                    continue
                code_m = CODE_RE.search(r[0]) if r else None
                if not code_m:
                    continue
                code = code_m.group(1)
                cells = [c for c in r]
                # Column layout: code, (blank), name, category, GPA, NGPA, MGPA, norms...
                name = cells[2] if len(cells) > 2 else ""
                cat = cells[3] if len(cells) > 3 else ""
                gpa, ngpa, mgpa = (cells[4:7] + ["", "", ""])[:3]
                cat_name = {"C": "Compulsory", "E": "Elective", "M": "Military"}.get(cat.strip().upper(), "")
                note = ""
                nm = re.search(r"\(([^)]*(?:semester|credits?)[^)]*)\)", name, re.I)
                if nm:
                    note = nm.group(1)
                    name = name.replace(nm.group(0), "").strip()
                rec = courses.setdefault(code, {"code": code, "title": name, "offerings": [], "descriptions": []})
                if len(name) > len(rec["title"]) and not note:
                    rec["title"] = name
                for prog in progs:
                    rec["offerings"].append({
                        "program": prog, "level": level, "semester": semester,
                        "category": cat_name, "gpa": _num(gpa), "ngpa": _num(ngpa), "mgpa": _num(mgpa),
                        "note": note, "page": t["page"], "table": t["caption"],
                    })

        # Course descriptions: "Title CODE" line ... "Credits NN TYPE-Category"
        for s in sections:
            lines = s["text"].splitlines()
            for i, ln in enumerate(lines):
                m = re.match(r"^(.*?)\s*\b([A-Z]{2,3}\d{4})\s*$", ln.strip())
                if not m:
                    continue
                title, code = m.group(1).strip(), m.group(2)
                if not title and i > 0:
                    title = lines[i - 1].strip()
                desc, credit_line = [], ""
                for nxt in lines[i + 1:i + 14]:
                    if re.match(r"^Credits\s+\d", nxt.strip()):
                        credit_line = nxt.strip()
                        break
                    if re.match(r"^(.*?)\s*\b[A-Z]{2,3}\d{4}\s*$", nxt.strip()):
                        break
                    desc.append(nxt.strip())
                if not credit_line:
                    continue
                cm = re.match(r"Credits\s+(\d+)\s+([A-Z]+)\s*[-–]?\s*(\w+)", credit_line)
                rec = courses.setdefault(code, {"code": code, "title": title, "offerings": [], "descriptions": []})
                if not rec["title"]:
                    rec["title"] = title
                rec["descriptions"].append({
                    "title": title, "text": " ".join(desc).strip(),
                    "credits": int(cm.group(1)) if cm else None,
                    "credit_type": cm.group(2) if cm else "",
                    "category": cm.group(3).capitalize() if cm else "",
                    "section": s["number"], "section_title": s["title"],
                    "programs": programs_in(sec_by_num.get(".".join(s["number"].split(".")[:3]), s)["title"]),
                })
        return courses

    @staticmethod
    def _parse_semester_totals(tables):
        totals = []
        for t in tables:
            if not HandbookExtractor._is_course_table(t):
                continue
            for r in t["rows"]:
                j = " ".join(r)
                m = re.search(r"Total for Semester\s*(\d)", j, re.I)
                if m:
                    nums = [c for c in r if _num(c) is not None]
                    if len(nums) >= 3:
                        totals.append({"semester": int(m.group(1)), "gpa": float(nums[-3]),
                                       "ngpa": float(nums[-2]), "mgpa": float(nums[-1]),
                                       "page": t["page"], "section": t["section"]})
        return totals

    def _parse_gpa_summary(self, tables):
        out = []
        for t in tables:
            head = " ".join(t["rows"][0]).lower() + " " + t["caption"].lower()
            if "gpa summary" not in head and not ("semester" in head and "ngpa" in head and "mgpa" in head):
                continue
            if self._is_course_table(t):
                continue
            rows = {}
            for r in t["rows"]:
                lab = r[0]
                nums = [c for c in r[1:] if _num(c) is not None]
                if len(nums) >= 3 and (lab.lower().startswith("semester") or lab.lower().startswith("total")):
                    rows[lab] = {"gpa": float(nums[0]), "ngpa": float(nums[1]), "mgpa": float(nums[2])}
            if rows:
                out.append({"page": t["page"], "section": t["section"], "rows": rows})
        return out

    @staticmethod
    def _parse_grades(tables):
        for t in tables:
            head = " ".join(t["rows"][0]).lower()
            if "grade" in head and "gpv" in head:
                grades = []
                for r in t["rows"][1:]:
                    vals = [c for c in r if c]
                    if len(vals) >= 2:
                        mk, gr = vals[0], vals[1]
                        gpv = vals[2] if len(vals) > 2 else ""
                        rng = re.match(r"(\d+)\s*-\s*(\d+)", normalize(mk))
                        grades.append({"marks": mk, "grade": gr, "gpv": gpv,
                                       "low": int(rng.group(1)) if rng else None,
                                       "high": int(rng.group(2)) if rng else None})
                return {"page": t["page"], "caption": t["caption"], "rows": grades}
        return {"rows": []}

    @staticmethod
    def _parse_classes(tables):
        for t in tables:
            head = " ".join(t["rows"][0]).lower()
            if "fgpa" in head and "result" in head:
                rows = []
                for r in t["rows"][1:]:
                    vals = [c for c in r if c]
                    if len(vals) >= 2:
                        cond, res = vals[0], vals[1]
                        nums = [float(x) for x in re.findall(r"\d+\.\d+", cond)]
                        low = nums[0] if nums else None
                        high = nums[1] if len(nums) > 1 else None
                        rows.append({"condition": cond, "result": res, "low": low, "high": high})
                return {"page": t["page"], "rows": rows}
        return {"rows": []}

    @staticmethod
    def _parse_programmes(tables, sections):
        progs = {}
        for t in tables:
            head = " ".join(t["rows"][0]).lower()
            if "degree program" in head and "duration" in head:
                for r in t["rows"][1:]:
                    name = r[0]
                    code = (programs_in(name) or [None])[0]
                    if not code:
                        continue
                    dur = r[1] if len(r) > 1 else ""
                    mil = re.search(r"Military:\s*([\d\s/]+)", dur)
                    civ = re.search(r"Civil:\s*([\d\s/]+)", dur)
                    progs[code] = {"code": code, "name": PROGRAMS.get(code, name),
                                   "military_years": (mil.group(1).strip().replace("04 1/2", "4½").replace("04", "4") if mil else ""),
                                   "civil_years": (civ.group(1).strip().replace("04", "4") if civ else ""),
                                   "stream": r[2] if len(r) > 2 else "", "page": t["page"]}
        # Programme descriptions (section titled "Courses Offered ...")
        for s in sections:
            if "courses offered" in s["title"].lower():
                for c in s["children"]:
                    child = next(x for x in sections if x["number"] == c)
                    code = (programs_in(child["title"]) or [None])[0]
                    if code:
                        progs.setdefault(code, {"code": code, "name": child["title"]})
                        progs[code]["description"] = child["text"]
                        progs[code]["section"] = child["number"]
        return progs

    @staticmethod
    def _parse_heights(tables):
        for t in tables:
            head = " ".join(t["rows"][0]).lower()
            if "army" in head and "navy" in head:
                cols = t["rows"][0]
                out = {}
                for r in t["rows"][1:]:
                    for ci, c in enumerate(r[1:], start=1):
                        if ci < len(cols) and cols[ci]:
                            out[f"{r[0].lower()}|{cols[ci].lower()}"] = c
                return out
        return {}

    @staticmethod
    def _parse_max_duration(tables):
        for t in tables:
            head = " ".join(t["rows"][0]).lower()
            if "max" in head and "year" in head:
                out = {}
                for r in t["rows"][1:]:
                    code = (programs_in(r[0]) or [None])[0]
                    if code and len(r) > 1:
                        out[code] = r[1]
                return out
        return {}

    # ---------------- departments ----------------
    @staticmethod
    def _parse_departments(sections):
        """Department -> degree programmes it offers (from 'offer ... BSc ...' sentences)."""
        out = {}
        for s in sections:
            code = next((c for c, n in DEPARTMENTS.items() if n.lower() == s["title"].lower()), None)
            if not code or not s["number"].startswith("1."):
                continue
            text = re.sub(r"\s+", " ", s["text"])
            sents = re.split(r"(?<=\.)\s+(?=[A-Z])", text)
            degrees, subjects = [], ""
            for sent in sents:
                if re.search(r"\boffer", sent, re.I) and "BSc" in sent and not re.search(r"percentage|as well", sent):
                    for p in programs_in(sent):
                        if p not in degrees:
                            degrees.append(p)
                m = re.search(r"subject areas,?\s*namely,?\s*(.+?)\.", sent)
                if m:
                    subjects = m.group(1)
            out[code] = {"code": code, "name": DEPARTMENTS[code], "section": s["number"], "page": s["page"],
                         "degrees": degrees, "subject_areas": subjects}
        return out

    # ---------------- staff ----------------
    @staticmethod
    def _parse_staff(sections):
        honor = r"(?:Dr\.|Mr\.|Mrs\.|Ms\.|Miss|Maj\.|Eng\.|Prof\.|Capt\.|Lt\.)"
        desig_re = re.compile(r"(lecturer|instructor|professor)", re.I)
        staff, seen = [], set()
        for s in sections:
            if not s["number"].startswith("7"):
                continue
            dept_code = None
            for code, name in DEPARTMENTS.items():
                if name.lower() in s["title"].lower():
                    dept_code = code
            lines = s["text"].splitlines()
            i = 0
            while i < len(lines):
                ln = lines[i].strip()
                nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if re.match(honor, ln) and desig_re.search(nxt) and not desig_re.search(ln.split("(")[0]):
                    name_part = re.sub(r"\(.*?(Dean|Head).*$", "", ln).strip()
                    roles = " ".join(re.findall(r"\((?:[^()]*?)(?:Dean|Head)[^)]*\)?", ln + " " + nxt))
                    if "Head" in nxt and "Head" not in roles:
                        roles += " " + nxt[nxt.find("Head"):]
                    block = []
                    j = i + 2
                    while j < len(lines):
                        l2 = lines[j].strip()
                        n2 = lines[j + 1].strip() if j + 1 < len(lines) else ""
                        if re.match(honor, l2) and desig_re.search(n2):
                            break
                        block.append(l2)
                        j += 1
                    btxt = "\n".join(block)
                    email = re.search(r"[\w.\-]+@[\w.\-]+", btxt)
                    ra = re.search(r"Research Areas?\s*(.*?)(?:\ne-mail|\Z)", btxt, re.S | re.I)
                    quals = btxt.split("Research Area")[0].strip()
                    designation = re.sub(r"\(\s*Head.*$|–\s*\(Head.*$|Head\s+Department.*$", "", nxt).strip(" –-")
                    key = re.sub(r"\W", "", name_part.lower())
                    if key not in seen:
                        seen.add(key)
                        staff.append({
                            "name": re.sub(r"\s+", " ", name_part.replace("Mrs.M", "Mrs. M")),
                            "designation": designation,
                            "role": re.sub(r"[()]", "", roles).replace("-", "– ").strip(),
                            "department": dept_code or ("CS" if s["number"] == "7" else None),
                            "qualifications": re.sub(r"\s*\n\s*", "; ", quals),
                            "research_areas": re.sub(r"\s*,\s*", ", ", re.sub(r"\s*\n\s*", " ", ra.group(1).strip())).strip(" ,") if ra else "",
                            "email": email.group(0) if email else "",
                            "section": s["number"], "page": s["page"],
                        })
                    i = j
                    continue
                i += 1
        return staff

    # ---------------- key appointments (two-column layout) ----------------
    @staticmethod
    def _parse_appointments(word_pages, sections):
        appts = []
        target_pages = sorted({p for s in sections if s["number"].startswith("8.") or s["number"] == "8"
                               for p in s["pages"]})
        for p in target_pages:
            words = word_pages.get(p, [])
            lines: dict[int, dict[str, list[str]]] = defaultdict(lambda: {"L": [], "R": []})
            for w in words:
                y = int(round(w["top"]))
                key = next((k for k in lines if abs(k - y) <= 2), y)
                lines[key]["L" if w["x0"] < 200 else "R"].append(w["text"])
            cur = None
            for y in sorted(lines):
                left = " ".join(lines[y]["L"]).strip()
                right = " ".join(lines[y]["R"]).strip()
                if re.match(r"^\d+(\.\d+)?\s", left) or left.startswith("Page"):
                    continue
                if cur and right and not left and re.match(r"^[A-Z][a-z]+,", right) and len(cur["name"].split()) <= 5:
                    cur["name"] += " " + right.split(",")[0]
                    continue
                if cur and right and re.match(r"^(Tel|Fax|Mobile)", left) and re.match(r"^[A-Z][a-z]+,", right) \
                        and len(cur["name"].split()) <= 5 and "(C)" in cur["name"]:
                    cur["name"] += " " + right.split(",")[0]
                if left and not re.match(r"^(Tel|Fax|Mobile)", left):
                    if cur and not cur["tel"] and y - cur["y"] < 16:
                        cur["role"] += " " + left
                        cur["y"] = y
                        if right and not cur["name"]:
                            cur["name"] = right
                        continue
                    cur = {"role": left, "name": right, "tel": "", "fax": "", "mobile": "", "page": p, "y": y}
                    appts.append(cur)
                elif cur and left:
                    k = left.split(":")[0].lower()
                    v = left.split(":", 1)[1].strip() if ":" in left else ""
                    if k in ("tel", "fax", "mobile"):
                        cur[k] = v
        for a in appts:
            a["role"] = re.sub(r"\s+", " ", a["role"]).strip()
            # Keep only the personal name part (drop post-nominals after a stop list)
            a["name"] = re.split(r"\s(?:RWP|RSP|USP|VSV|psc|ndc|\(Retired\))", a["name"])[0].strip(" ,")
            a.pop("y", None)
        return [a for a in appts if a["name"] and re.search(r"[a-z]", a["role"])]

    # ---------------- contacts ----------------
    @staticmethod
    def _parse_contacts(pages):
        alltext = "\n".join(pages)
        return {
            "emails": sorted(set(re.findall(r"[\w.\-]+@kdu\.ac\.lk", alltext, re.I))),
            "faculty_email": (re.search(r"E-mail:\s*([\w.\-]+@[\w.\-]+)", alltext) or [None, ""])[1],
            "faculty_web": (re.search(r"Faculty Web Page:\s*(\S+)", alltext) or [None, ""])[1],
            "university_web": (re.search(r"University Web Page:\s*(\S+)", alltext) or [None, ""])[1],
            "university_tel": (re.search(r"Tel:\s*([+\d][\d\s]+\d)", alltext[alltext.rfind("University Contact Details"):])
                               or [None, ""])[1],
            "southern_tel": " / ".join((re.search(r"Southern Campus,.*?Tel:\s*((?:0\d{2}\s?\d{7}\s*)+)", alltext, re.S)
                                        or [None, ""])[1].split("\n")).strip(" /"),
            "university_address": re.sub(r"\s+", " ", (re.search(r"University Contact Details\s*\n(.*?)\nTel", alltext[alltext.rfind("University Contact Details") - 30:], re.S) or [None, ""])[1]).strip(),
            "dean_message_author": (re.search(r"Message of the Dean.*?\n(Dr\.[^\n]+)", alltext, re.S) or [None, ""])[1].strip(),
            "dean_phone": (re.search(r"DEAN-?\s*FACULTY OF COMPUTING\s*([+\d\-]+)", alltext) or [None, ""])[1],
            "dept_phone": (re.search(r"DEPARTMENT OF COMPUTER SCIENCE\s*([+\d\-]+)", alltext) or [None, ""])[1],
            "postal": " ".join(l.strip() for l in (re.search(r"Postal Address:\s*\n(.*?)Rathmalana\.", alltext, re.S)
                                                    or [None, ""])[1].splitlines()) + " Rathmalana.",
        }

    # ---------------- glossary of acronyms ----------------
    @staticmethod
    def _parse_glossary(pages):
        alltext = normalize(" ".join(pages))
        glossary = {}
        for m in re.finditer(r"([A-Z][A-Za-z]+(?:[\s\-][A-Za-z]+){1,6})\s*\(([A-Z][A-Za-z/]{1,6})\)", alltext):
            full, ac = m.group(1).strip(), m.group(2)
            words = full.split()
            # keep the shortest suffix whose initials match the acronym
            initials = re.sub(r"[^A-Z]", "", ac.upper())
            for k in range(len(words)):
                cand = words[k:]
                if "".join(w[0].upper() for w in cand if w.lower() not in ("of", "and", "in")) == initials:
                    glossary.setdefault(ac.upper(), " ".join(cand))
                    break
        # Acronym first: "GPA (Grade Point Average)"
        for m in re.finditer(r"\b([A-Z]{2,6})\s*\(([A-Z][a-z]+(?:[\s-]+[A-Za-z]+){1,6})\)", alltext):
            ac, full = m.group(1), m.group(2)
            initials = "".join(w[0].upper() for w in re.split(r"[\s-]+", full) if w.lower() not in ("of", "and", "in"))
            if initials == ac:
                glossary[ac] = full
        glossary = {k: v[0].upper() + v[1:] for k, v in glossary.items()}
        return glossary

    # ---------------- retrieval passages ----------------
    @staticmethod
    def _build_passages(kb):
        """Sentence / list-item passages. Each list item keeps the sentence that introduced
        the list as hidden context, and items of one list share a group id."""
        passages = []
        sec_title = {s["number"]: s["title"] for s in kb["sections"]}
        item_re = re.compile(r"^(?:•|[a-z]\.|[a-z]\)|\d\.)\s")
        group_id = [0]

        def add(text, sec, page, kind="sentence", context="", group=None):
            text = re.sub(r"\s+", " ", text).strip()
            toks = tokens(text)
            if kind != "table_row" and len(toks) > 12 and \
                    sum(t.replace(".", "").isdigit() for t in toks) / len(toks) > 0.35:
                return  # raw table text left in the page text; the table itself is indexed row by row
            if len(text) >= 12 and not re.match(r"^Credits\b", text):
                passages.append({"id": len(passages), "text": text, "section": sec,
                                 "section_title": sec_title.get(sec, ""), "page": page, "kind": kind,
                                 "context": context, "group": group})

        for s in kb["sections"]:
            if s["number"].startswith("7"):
                continue  # staff pages are answered from structured records
            text = s["text"]
            if s["number"] == "0":
                text = "\n".join(l for l in text.splitlines() if not re.search(r"\.{4,}|^Table of Contents", l))
            # paragraph breaks around course headings and credit lines, and before list items
            text = re.sub(r"\n(?=[^\n]*\b[A-Z]{2,3}\d{4}\s*\n)", "\n\n", text)
            text = re.sub(r"\n(Credits\s+\d[^\n]*)\n?", r"\n\n\1\n\n", text)
            text = re.sub(r"\n(?=(?:•|[a-z]\.|[a-z]\)|\d\.)\s)", "\n\n", text)
            text = re.sub(r"\n(?=Table\s*\d+)", "\n\n", text)
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
            intro, in_list, gid = "", False, None
            for para in re.split(r"\n{2,}", text):
                para = para.strip()
                if not para or re.match(r"^Credits\b", para):
                    continue
                is_item = bool(item_re.match(para))
                if is_item:
                    if not in_list:
                        group_id[0] += 1
                        gid = group_id[0]
                        in_list = True
                    # an item may be glued to the next normal sentence ("... Test. The Following ...")
                    parts = re.split(r"(?<=\.)\s+(?=The\s|A\s|An\s|This\s|These\s|Following\s)", para, maxsplit=1)
                    add(parts[0], s["number"], s["page"], "list_item", intro, gid)
                    if len(parts) == 1:
                        continue
                    para = parts[1]
                in_list = False
                sents = re.split(r"(?<=[.;:])\s+(?=[A-Z(\u2211])", para)
                for sent in sents:
                    add(sent, s["number"], s["page"])
                last = sents[-1].strip() if sents else ""
                if last.endswith(":") or re.search(r"\bfollowing\b", last, re.I):
                    intro = last
                elif not is_item:
                    intro = intro if len(para) < 40 else ""
        for t in kb["tables"]:
            if HandbookExtractor._is_course_table(t):
                continue
            header = t["rows"][0]
            for r in t["rows"][1:]:
                cells = [f"{h}: {c}" if h else c for h, c in zip(header, r) if c]
                add(f"{t['caption']} - " + "; ".join(cells), t["section"], t["page"], kind="table_row")
        return passages


# ======================================================================
# 3. RETRIEVAL  (BM25F over sentences with a section-title field)
# ======================================================================

def program_tokens(text: str) -> list[str]:
    """Index tokens for degree programmes mentioned in a text (case-sensitive for IT/IS/CS/SE/CE)."""
    out = []
    for code in PROGRAMS:
        if re.search(rf"\b{code}\b", text) or any(re.search(p, text.lower()) for p in PROGRAM_PATTERNS[code][:1]):
            out.append("prog" + code.lower())
    return out


class BM25F:
    def __init__(self, docs: list[dict[str, Any]], k1: float = 1.2, b: float = 0.6, title_weight: float = 1.5,
                 glossary: dict[str, str] | None = None):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        for d in docs:
            tf = Counter(stems(d["text"]))
            for tkn in stems(d.get("section_title", "")):
                tf[tkn] += title_weight
            for tkn in stems(d.get("context", "")):
                tf[tkn] += 0.6
            # programme tokens: "IT"/"IS" in capitals are programme names, not English stop words
            for tok in program_tokens(d["text"]):
                tf[tok] += 1.0
            # acronym linking: a passage that says "CA" also matches "continuous assessment"
            for ac, full in (glossary or {}).items():
                if re.search(rf"\b{re.escape(ac)}\b", d["text"]):
                    for tkn in stems(full):
                        tf[tkn] += 0.5
            self.tf.append(tf)
            self.df.update(set(tf))
        self.N = len(docs)
        self.avgdl = sum(sum(t.values()) for t in self.tf) / max(self.N, 1)

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, q_weights: dict[str, float], i: int) -> float:
        tf = self.tf[i]
        dl = sum(tf.values())
        s = 0.0
        for term, w in q_weights.items():
            f = tf.get(term, 0)
            if f:
                s += w * self.idf(term) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, q_weights: dict[str, float], k: int = 10) -> list[tuple[float, int]]:
        scored = [(self.score(q_weights, i), i) for i in range(self.N)]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(reverse=True)
        return scored[:k]


# ======================================================================
# 4. QUESTION ANSWERING ENGINE
# ======================================================================

@dataclass
class Answer:
    text: str
    score: float
    kind: str
    page: int | None = None
    evidence: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
            "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8, "8th": 8,
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def fmt_num(x: float | None) -> str:
    if x is None:
        return "-"
    return str(int(x)) if float(x).is_integer() else str(x)


def join_list(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class HandbookQA:
    def __init__(self, kb: dict[str, Any], debug: bool = False):
        self.kb = kb
        self.debug = debug
        self.sections = {s["number"]: s for s in kb["sections"]}
        self.courses: dict[str, dict[str, Any]] = kb["courses"]
        self.passages = kb["passages"]
        self.bm25 = BM25F(self.passages, glossary=kb.get("glossary", {}))
        # vocabulary for spelling correction
        vocab = Counter()
        for p in self.passages:
            vocab.update(tokens(p["text"], drop_stop=False))
        for s in kb["sections"]:
            vocab.update(tokens(s["title"] + " " + s["text"], drop_stop=False))
        for c in self.courses.values():
            vocab.update(tokens(c["title"], drop_stop=False))
        for st in kb["staff"]:
            vocab.update(tokens(st["name"] + " " + st["research_areas"], drop_stop=False))
        extra = "how many what who where when which list show tell semester credit credits gpa ngpa mgpa module modules " \
                "course courses subject subjects lecturer lecturers staff email phone dean head department elective " \
                "compulsory optional degree degrees programme programmes program programs mark marks grade class " \
                "duration year years fgpa sgpa gpv pass fail offered research grading system attendance requirement " \
                "examination supplementary discontinue military requirements eligibility taught offered value module " \
                "number phone telephone website email address contact compare comparison difference differences " \
                "versus between better enough which whether should could would possible allowed"
        vocab.update(extra.split())
        self.vocab = {w for w, c in vocab.items() if len(w) > 2}
        # typo targets: frequent domain terms only (a rare word like "tagging" must not absorb "ragging")
        self.correction_targets = [w for w, c in vocab.items() if c >= 6 and len(w) > 3] + extra.split()
        # module acronyms from titles: "Human Computer Interaction" -> HCI (unique, 2+ letters)
        reserved = {"CS", "SE", "CE", "IT", "IS", "CA", "ES", "WE", "GPA", "NGPA", "MGPA", "FGPA", "SGPA", "IP"}
        acr: dict[str, list[str]] = defaultdict(list)
        for code, c in self.courses.items():
            words = [w for w in re.findall(r"[A-Za-z]+", c["title"]) if w.lower() not in ("and", "of", "in", "for", "the", "to")]
            roman = [w for w in words if w in ("I", "II", "III")]
            core = [w for w in words if w not in ("I", "II", "III")]
            if len(core) >= 2:
                a = "".join(w[0].upper() for w in core) + ("".join(" " + r for r in roman))
                acr[a].append(code)
        self.course_acronyms = {a: v[0] for a, v in acr.items() if len(v) == 1 and a.split()[0] not in reserved}
        # course title index (normalized title -> code)
        self.title_index: list[tuple[list[str], str]] = []
        for code, c in self.courses.items():
            names = {c["title"]} | {d["title"] for d in c["descriptions"] if d["title"]}
            for n in names:
                toks = [t for t in tokens(n) if t not in ("i", "ii")] + [t for t in tokens(n, False) if t in ("i", "ii")]
                if toks:
                    self.title_index.append((tokens(n, drop_stop=False), code))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    FOLLOWUP_RE = re.compile(r"^(what about|how about|and|what of|same for|for)\b|\b(it|its|it's|this module|that module|"
                             r"this course|that course|he|she|his|her|him|them|they|their)\b", re.I)

    def resolve_followup(self, question: str, previous: str | None) -> str:
        """Rewrite a short follow-up using the previous question.
        'what about semester 5?' after 'how many modules in semester 4?' -> 'how many modules in semester 5?'
        'how many credits is it?' after 'What is CS2082?'              -> 'how many credits is CS2082?'"""
        if not previous or not self.FOLLOWUP_RE.search(question) or len(question.split()) > 9:
            return question
        q, prev = question.strip(), previous.strip()
        slots = [
            (r"\bsem(?:ester)?\s*0?[1-8]\b", "semester"),
            (r"\b[A-Za-z]{2,3}\s?\d{4}\b", "code"),
            (r"\b\d\.\d{1,2}\b", "fgpa"),
            (r"\b(?:(?i:computer science|software engineering|computer engineering|information technology|"
             r"information systems)|CS|SE|CE|IT|IS|cs|se|ce)\b", "programme"),
            (r"(?<![\d.])\b\d{2,3}\b(?![.\d])", "marks"),
        ]
        # 1) slot substitution: the follow-up names a new value of a slot the previous question had.
        #    Only for real follow-up openers ("what about semester 5?", "and CS1073?") or very short questions.
        is_short_followup = re.match(r"^(what about|how about|and|what of|same for|for)\b", q, re.I) or \
            len(q.split()) <= 3
        for pat, name in (slots if is_short_followup else []):
            flags = 0 if name == "programme" else re.I
            new = re.search(pat, q, flags)
            old = re.search(pat, prev, flags)
            if new and old:
                return prev[:old.start()] + new.group(0) + prev[old.end():]
        # 2) pronoun substitution: "it/its/this module" -> the module/person of the previous question.
        #    Skipped when the question already names its own subject ("it department" = IT department).
        cur_l = normalize(q).lower()
        cur_ctx = self.parse(normalize(q), q)
        if cur_ctx["codes"] or self.match_course_title(cur_l) or self.match_person(cur_l) or \
                re.search(r"\bit\s+(department|dept|degree|programme|program|stream|students?|graduates?)\b", cur_l) or \
                re.search(r"\b(department|faculty|university|kdu)\b", cur_l) and not re.search(r"\b(its|it's)\b", cur_l):
            return q
        prev_ctx = self.parse(normalize(prev), prev)
        subject = None
        if prev_ctx["codes"]:
            subject = prev_ctx["codes"][0]
        else:
            m = self.match_course_title(normalize(prev).lower())
            if m:
                subject = self.courses[m[0]]["code"]
        person = self.match_person(normalize(prev).lower())
        if person is None and re.search(r"\b(he|she|his|her|him)\b", q, re.I):
            # "Who is the dean?" -> the person named in the previous answer
            prev_ans = self._answer(prev)["answer"]
            person = self.match_person(normalize(prev_ans).lower())
        if person and re.search(r"\b(he|she|his|her|him)\b", q, re.I):
            return re.sub(r"\b(he|she|his|her|him)\b", person["name"], q, count=1, flags=re.I)
        if subject and re.search(r"\b(it|its|it's|this module|that module|this course|that course)\b", q, re.I):
            return re.sub(r"\b(it's|its|it|this module|that module|this course|that course)\b", subject, q,
                          count=1, flags=re.I)
        # 3) "what about X" with a new topic word: keep the question frame of the previous question
        m = re.match(r"^(?:what about|how about|and|what of|same for)\s+(.+?)\??$", q, re.I)
        if m and person is None and subject is None:
            return q
        return q

    def answer(self, question: str, previous: str | None = None) -> dict[str, Any]:
        original = (question or "").strip()
        question = self.resolve_followup(original, previous)
        res = self._answer(question)
        if question != original:
            res["interpreted_question"] = question  # shown in the UI for follow-up questions
        return res

    # ------------------------------------------------------------------
    # v4: yes/no verdicts computed from the structured records
    # ------------------------------------------------------------------
    CAT_WORDS = {"compulsory": "Compulsory", "mandatory": "Compulsory", "core": "Compulsory",
                 "elective": "Elective", "optional": "Elective", "military": "Military"}

    def yes_no_layer(self, ql: str, ctx: dict[str, Any], ans: "Answer") -> "Answer":
        if ctx["qtype"] != "yesno" or ans.kind.startswith("not_found"):
            return ans
        verdict = None
        # (a) is <module> compulsory / elective / military ?
        asked_cat = next((v for k, v in self.CAT_WORDS.items() if re.search(rf"\b{k}\b", ql)), None)
        if asked_cat and ans.kind.startswith("course"):
            code = next((c for c in ctx["codes"] if c in self.courses), None)
            if not code:
                m = self.match_course_title(ql)
                code = m[0] if m else self.match_course_acronym(ctx["raw"])
            if code:
                offs = self.course_offering(code, [p for p in ctx["programs"] if p in ("CS", "SE")])
                cats = {o["category"] for o in offs if o["category"]}
                if cats == {asked_cat}:
                    verdict = "Yes."
                elif asked_cat not in cats:
                    verdict = "No."
                else:
                    verdict = "It depends on the degree:"
        # (b) does department X offer degree Y ?
        elif ans.kind == "department_degrees":
            dept = self.match_department(ql)
            d = self.kb.get("departments", {}).get(dept, {})
            dept_words = DEPARTMENTS.get(dept, "").lower().replace("department of ", "")
            rest = ql.replace(dept_words, " ") if dept_words else ql
            asked = [p for p in programs_in(rest)] + [p for p in ctx["programs"] if p not in programs_in(ql)]
            asked = [p for p in dict.fromkeys(asked) if p != dept or dept_words not in ql]
            if asked and d:
                if all(p in d["degrees"] for p in asked):
                    verdict = "Yes."
                else:
                    owner = next((x for x in self.kb["departments"].values() if asked[0] in x["degrees"]), None)
                    extra = f" {PROGRAMS[asked[0]]} is offered by the {owner['name']}." if owner else ""
                    ans = Answer(ans.text + extra, ans.score, ans.kind, ans.page)
                    verdict = "No."
        # (c) is the degree N years?
        elif ans.kind == "programme_duration" and ctx["ints"]:
            n = ctx["ints"][0]
            p = self.kb["programmes"].get((ctx["programs"] or ["CS"])[0], {})
            who = "military" if re.search(r"military", ql) else "civil"
            val = p.get("military_years" if who == "military" else "civil_years", "")
            num = re.search(r"\d+", val.replace("½", ""))
            if num:
                verdict = "Yes." if int(num.group(0)) == n and ("½" not in val) else "No."
        # (d) do I pass with N marks?
        elif ans.kind == "grade_lookup" and re.search(r"\bpass\b", ql) and ctx["ints"]:
            m = ctx["ints"][0]
            verdict = "Yes." if m >= 45 else "No."
            ans = Answer(ans.text + " The pass mark is 45% (grade C).", ans.score, ans.kind, ans.page)
        # (e) is FGPA x enough for <class>?
        elif ans.kind == "class_lookup" and ctx["decimals"]:
            target = "First Class" if re.search(r"first", ql) else "Upper" if re.search(r"upper", ql) else \
                "Lower" if re.search(r"lower", ql) else "Pass" if re.search(r"\bpass\b", ql) else None
            if target:
                order = ["Pass", "Lower", "Upper", "First Class"]
                got = next((t for t in order if t in ans.text), None)
                if got:
                    ok = order.index(got) >= order.index(target)
                    verdict = "Yes." if ok else "No."
                    if ok and target != "Pass":
                        ans = Answer(ans.text + " It meets the FGPA requirement; the other class conditions "
                                     "(e.g. no failure grades) must also be met.", ans.score, ans.kind, ans.page)
        if verdict:
            return Answer(f"{verdict} {ans.text}", ans.score, ans.kind + "_yesno", ans.page, ans.evidence)
        return ans

    def suggest_topics(self, ql: str) -> str:
        """For 'not found' answers: handbook section titles that share a topic word with the question."""
        qs = {t for t in stems(ql) if t not in self.VERBS and len(t) > 3}
        for t in tokens(ql):  # include thesaurus forms: exam -> examination, cheating -> offence
            for syn in THESAURUS.get(t, []) + THESAURUS.get(stem(t), []):
                qs |= set(stems(syn))
        hits = []
        for sec in self.kb["sections"]:
            if sec["number"] in ("0",) or sec["number"].startswith("7"):
                continue
            ov = qs & set(stems(sec["title"]))
            if ov:
                hits.append((len(ov), -len(sec["number"]), sec))
        hits.sort(key=lambda x: (-x[0], x[1]))
        seen, names = set(), []
        for _, _, sec in hits:
            if sec["title"] not in seen:
                seen.add(sec["title"])
                names.append(f"{sec['title']} ({sec['number']})")
        return (" Closest handbook topics: " + "; ".join(names[:3]) + ".") if names else ""

    # ------------------------------------------------------------------
    # v4: comparisons ("compare X and Y", "difference between", "vs")
    # ------------------------------------------------------------------
    def h_degree_structure(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        """'How many semesters / levels are in the degree?' (course tables list Semester 1-8, Levels 1-4)."""
        if not re.search(r"\bhow many (semesters?|levels?)\b|\bnumber of (semesters?|levels?)\b", ql):
            return None
        if re.search(r"\b(max|maximum|within|allowed|extend\w*|discontinu\w*|absent)\b", ql) or ctx["codes"]:
            return None
        sems = sorted({o["semester"] for c in self.courses.values() for o in c["offerings"] if o["semester"]})
        levels = sorted({o["level"] for c in self.courses.values() for o in c["offerings"] if o["level"]})
        if not sems:
            return None
        progs = [p for p in ctx["programs"] if p in self.kb["programmes"]]
        who = join_list([PROGRAMS[p] for p in progs]) if progs else "Each degree programme"
        p = self.kb["programmes"].get(progs[0] if progs else "CS", {})
        mil = f" Military students follow the programme over {p['military_years']} years." if p.get("military_years") else ""
        text = (f"{who} has {len(sems)} semesters: {len(levels)} academic levels (years) with two semesters each "
                f"(Semester 1 to {max(sems)}).{mil} The maximum allowed period is 16 semesters (8 years).")
        return Answer(text, 0.95, "degree_structure")

    def alternatives(self, question: str, exclude: str = "", k: int = 2) -> list[str]:
        """Next-best evidence sentences for a question (used when the user says an answer was not right)."""
        ql = self.correct_spelling(expand_abbreviations(normalize(question))).lower()
        qw = self.expand_query(ql)
        q_terms = {t for t, w in qw.items() if w >= 1.0}
        out = []
        for s, i in self.bm25.search(qw, k=30):
            p = self.passages[i]
            if p["text"][:60] in exclude or any(p["text"][:60] in o for o in out):
                continue
            if self.coverage(q_terms, self.bm25.tf[i]) >= 0.34:
                out.append(self.shorten(p["text"], 300))
            if len(out) >= k:
                break
        return out

    def h_compare(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        if not re.search(r"\b(compare|comparison|difference|differences|differ|vs\.?|versus|which has more|"
                         r"which is more|which has less|which is longer)\b", ql):
            return None
        # two or more modules
        codes = [c for c in ctx["codes"] if c in self.courses]
        if len(codes) < 2:
            qt = " " + " ".join(tokens(ql, drop_stop=False)) + " "
            found = []
            for toks, code in sorted(self.title_index, key=lambda x: -len(x[0])):
                phrase = " " + " ".join(toks) + " "
                if len(toks) >= 2 and phrase in qt and code not in found:
                    found.append(code)
                    qt = qt.replace(phrase, " # ", 1)
            codes = codes + [c for c in found if c not in codes]
        if len(codes) >= 2:
            parts, credit_of = [], {}
            for code in codes[:3]:
                offs = self.course_offering(code, ctx["programs"])
                credit_of[code] = max((o["gpa"] or o["ngpa"] or o["mgpa"] or 0) for o in offs) if offs else 0
                parts.append(f"{code} {self.courses[code]['title']}: {self.credit_text(code, offs)}; "
                             f"{self.describe_offerings(offs)}")
            text = " | ".join(parts) + "."
            if re.search(r"\bmore credits\b|\bless credits\b|\bfewer credits\b", ql):
                a, b = codes[0], codes[1]
                if credit_of[a] == credit_of[b]:
                    text = f"Both carry {fmt_num(credit_of[a])} credits. " + text
                else:
                    big = a if credit_of[a] > credit_of[b] else b
                    text = f"{big} has more credits ({fmt_num(credit_of[big])}). " + text
            return Answer(text, 0.98, "compare_modules")
        # two or more programmes
        progs = list(dict.fromkeys(ctx["programs"] + programs_in(ql)))
        if len(progs) >= 2 and re.search(r"\b(degrees?|programmes?|programs?|bsc)\b", ql) or \
                (len(progs) >= 2 and not re.search(r"\b(module|course|semester)\b", ql)):
            totals = {}
            for g in self.kb["gpa_summary"]:
                tprogs = programs_in(self._level_section(g["section"], want="degree")) or ["CS"]
                tot = next((v for k, v in g["rows"].items() if k.lower().startswith("total")), None)
                for tp in tprogs:
                    if tot:
                        totals[tp] = tot
            parts = []
            for p in progs[:3]:
                info = self.kb["programmes"].get(p, {})
                dept = next((d["name"] for d in self.kb.get("departments", {}).values() if p in d["degrees"]), "")
                bits = [PROGRAMS[p]]
                if dept:
                    bits.append(dept)
                if info.get("civil_years"):
                    bits.append(f"{info['civil_years'].replace(' Years', '')} yrs civil / {info['military_years']} yrs military")
                if info.get("stream"):
                    bits.append(f"A/L {info['stream']} stream")
                if p in totals:
                    bits.append(f"{fmt_num(totals[p]['gpa'])} GPA credits")
                parts.append(", ".join(bits))
            note = ""
            if set(progs[:2]) == {"IT", "IS"}:
                note = (" IT and IS share a common curriculum for the first two years; IT is more technical, "
                        "IS more management/business oriented.")
            text = " | ".join(parts) + "." + note
            ps = [p for p in progs[:3] if p in totals]
            if len(ps) >= 2 and re.search(r"\b(more|most|less|fewer|least) credits\b", ql):
                vals = {p: totals[p]["gpa"] for p in ps}
                fewer = re.search(r"\b(less|fewer|least)\b", ql)
                pick = (min if fewer else max)(ps, key=lambda p: vals[p])
                if len(set(vals.values())) == 1:
                    text = f"They have the same number of GPA credits ({fmt_num(vals[pick])}). " + text
                else:
                    word = "fewer" if fewer else "more"
                    text = (f"{PROGRAMS[pick]} has {word} GPA credits ({fmt_num(vals[pick])} vs "
                            + ", ".join(fmt_num(vals[p]) for p in ps if p != pick) + "). " + text)
            return Answer(text, 0.98, "compare_programmes")
        # two semesters
        sems = [int(x) for x in re.findall(r"\bsem(?:ester)?\s*0?([1-8])\b", ql)]
        if len(sems) >= 2:
            parts = []
            for sem in sems[:3]:
                tots = [t for t in self.kb["semester_totals"] if t["semester"] == sem]
                if tots:
                    t = tots[0]
                    parts.append(f"Semester {sem}: {fmt_num(t['gpa'])} GPA, {fmt_num(t['ngpa'])} NGPA, "
                                 f"{fmt_num(t['mgpa'])} MGPA credits")
            if parts:
                return Answer(" | ".join(parts) + " (CS; see the semester tables for SE differences).", 0.97,
                              "compare_semesters")
        # two grades
        if len(ctx["grades"]) >= 2:
            rows = {r["grade"].lower(): r for r in self.kb["grades"]["rows"]}
            parts = [f"{g}: {rows[g.lower()]['marks']} marks, GPV {rows[g.lower()]['gpv']}" for g in ctx["grades"][:3]
                     if g.lower() in rows]
            if len(parts) >= 2:
                return Answer(" | ".join(parts) + ".", 0.97, "compare_grades")
        return None

    def _passage(self, needle: str) -> dict[str, Any] | None:
        return next((p for p in self.kb["passages"] if needle.lower() in p["text"].lower()), None)

    def h_situations(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        """Common student situations phrased in everyday words, mapped to the handbook rule that covers them
        (or to an explicit "the handbook does not cover this" when it has no rule)."""
        if re.search(r"\b(miss(ed|ing)?|skip(ped)?|absent|could ?n.?t (sit|attend|face)|can.?t (sit|attend|face)|"
                     r"unable to (sit|attend|face))\b", ql) and re.search(r"\b(exams?|examinations?|papers?|tests?)\b", ql) \
                and re.search(r"\b(sick|ill|illness|medical|hospital\w*|fever|accident|unwell)\b", ql):
            p1 = self._passage("for which an Excuse has been granted")
            p2 = self._passage("Final Marks: Excused")
            if p1:
                text = ("If you miss an exam for a valid medical reason and the faculty grants an Excuse, the course "
                        "gets the grade Ex (Excused), and " + p1["text"].split("a. ", 1)[-1][0].lower()
                        + p1["text"].split("a. ", 1)[-1][1:])
                if p2:
                    text += " (An unexcused absence is graded Ab, GPV 0.00.)"
                return Answer(text, 0.9, "situation_medical", page=p1["page"], evidence=p1["text"])
        def group_text(g: int) -> str:
            items = [p for p in self.passages if p.get("group") == g]
            return (items[0].get("context", "") + " " + " ".join(p["text"] for p in items)).strip() if items else ""

        def by_group_of(needle: str) -> tuple[str, int | None]:
            p = self._passage(needle)
            return (group_text(p["group"]) if p and p.get("group") is not None else (p["text"] if p else "")), \
                (p["page"] if p else None)

        # vision / mission statements (KDU's, on page 1), not "computer vision"
        m = re.search(r"\b(vision|mission)\b", ql)
        if m and not re.search(r"computer vision|image|graphics|module|course", ql):
            p = next((x for x in self.passages if x["text"].startswith(m.group(1).title() + " ")), None)
            if p:
                return Answer(f"KDU's {m.group(1)}: " + p["text"].split(" ", 1)[1], 0.95, "vision_mission",
                              page=p["page"], evidence=p["text"])
        # changing / switching degree programme
        if re.search(r"\b(change|switch|transfer|move)\b.*\b(degree|programme|program|stream|course|field)\b", ql) \
                and not re.search(r"\bcredit|weight|component\b", ql):
            p = self._passage("specialization in either IT or in IS begins from the third year")
            return Answer("The Student Handbook does not describe how to change to another degree programme; please "
                          "ask the Faculty office. The only related rule: " + (p["text"] if p else ""), 0.85,
                          "not_found", page=p["page"] if p else None)
        # awards / prizes
        if re.search(r"\b(awards?|prizes?|trophies|trophy|medals?|best graduant|merit)\b", ql) and \
                not re.search(r"\b(class|degree|first|second|fgpa)\b", ql):
            text, page = by_group_of("Trophy for the Best Graduant in Computer Science")
            p0 = self._passage("Students obtaining the highest GPA in Academic Studies")
            if text:
                return Answer(((p0["text"] + " ") if p0 else "") + text, 0.9, "awards", page=page)
        # how many modules may I fail / D+ and C- limits
        if re.search(r"\bhow many\b.*\b(fail|weak pass|d\+|c-)|\b(fail|weak pass)\w*\b.*\bhow many\b|"
                     r"\b(minimum )?requirements? to (complete|pass) a semester\b|\bhow (do|can) i pass a semester\b", ql):
            text, page = by_group_of("obtain not more than one")
            if text:
                return Answer(text, 0.9, "semester_requirements", page=page)
        # passing GPA / GPA needed to graduate
        if re.search(r"\b(passing|pass) (f?gpa)\b|\b(minimum|lowest|required) f?gpa (to|for|needed to) "
                     r"(pass|graduate|get (a|the) degree)\b|\bf?gpa (needed|required) to (pass|graduate)\b", ql):
            p = self._passage("Earning a GPA of not less than 2.00")
            if p:
                return Answer("You need a GPA of at least 2.00 for the entire degree programme. This is also the "
                              "lowest FGPA for a Pass (2.00 <= FGPA < 3.00). Handbook rule: " + p["text"], 0.92,
                              "passing_gpa", page=p["page"], evidence=p["text"])
        # how re-sitting works (section 4.5.2)
        if re.search(r"\bhow (do|can|to|does|should)\b.*\bre-?sit\w*|\bre-?sit\w* (rules?|process|procedure|"
                     r"system|works?)\b|\bwhat (are|is) the re-?sit\w*|\bhow (do|can) i (repeat|redo) (a |the )?"
                     r"(module|course|subject|exam)\b", ql) and not re.search(r"\b(fee|charge|cost|pay|price)\b", ql):
            parts = [self._passage(n) for n in ("shall be completed by re-sitting the ES component",
                                                "shall be completed by re-sitting the CA/PBCA component",
                                                "shall be completed by re-sitting the both ES",
                                                "would be carried over when re-sitting",
                                                "for re-sitting a course unit",
                                                "for which an Excuse has been granted would be")]
            parts = [p for p in parts if p]
            if parts:
                text = "Re-sitting depends on your grade: " + " ".join(
                    re.sub(r"^[a-e]\.\s*", "", p["text"]) for p in parts)
                return Answer(text, 0.9, "resit_rules", page=parts[0]["page"])
        # how the GPA is calculated -> the SGPA formula sentence
        if re.search(r"\bhow\b.*\b[sfy]?gpa\b.*\bcalculat\w*|\b[sfy]?gpa\b.*\b(formula|calculation)\b|\bcalculat\w* "
                     r"(the |my )?[sfy]?gpa\b", ql) and not re.search(r"\bmilitary\b", ql):
            p = self._passage("is calculated from GPV earned for individual courses")
            p2 = self._passage("Only the GPA credits are considered when calculating SGPA")
            if p:
                text = ("SGPA = sum of (grade point x credit value) over the GPA course units of the semester, "
                        "divided by the total GPA credits of the semester. "
                        + (p2["text"] + " " if p2 else "") + "(For military students, MGPA credits are counted too.)")
                return Answer(text, 0.92, "gpa_formula", page=p["page"], evidence=p["text"])
        # how students are evaluated
        if re.search(r"\bhow (are|will|is|do)\b.*\b(evaluat\w*|assess\w*|graded|marked|examined)\b|"
                     r"\b(assessment|evaluation) (method|methods|criteria|structure|scheme)\b", ql) and \
                not re.search(r"\b(gpa|class|grade point)\b", ql):
            p = self._passage("The CA component is generally 30%")
            if p:
                return Answer("Each course is assessed by Continuous Assessment (CA) and an End Semester Written "
                              "Examination (WE). " + p["text"], 0.9, "assessment", page=p["page"], evidence=p["text"])
        # medium / language of instruction
        if re.search(r"\b(medium|language) of (instruction|teaching|study|studies)\b|\b(taught|conducted|lectures?) "
                     r"in (which|what) language\b|\bwhich language\b.*\b(taught|lectures?|study)\b", ql):
            p = self._passage("in the medium of En")
            if p:
                return Answer("English. The handbook says the Writing and Speaking Skills module (DL3152) prepares "
                              "students for academic studies in the medium of English, and a Credit (C) pass in "
                              "O/L English is an entry requirement.", 0.85, "medium", page=p["page"],
                              evidence=p["text"])
        # what a credit is
        if re.search(r"^(what is|whats|what are) (a |the )?credits?\??$|\bcredit hours?\b|\bhours? (per|for|in) "
                     r"(a|one) credit\b|\bwhat does (a )?credit mean\b", ql):
            return Answer("The Student Handbook does not say how many hours one credit is. It uses three kinds of "
                          "credits: GPA credits (count towards your GPA), NGPA credits (non-GPA, not counted in the "
                          "GPA) and MGPA credits (military courses, for military "
                          "students).", 0.8, "credit_types")
        m = re.search(r"\b(online|distance|part[- ]?time|evening|weekend|external)\b.*\b(degrees?|programmes?|"
                      r"programs?|courses?|study|studies|classes)\b|\bpart[- ]?time (job|work|jobs)\b|"
                      r"\b(work|job)\b.*\bwhile (studying|i study)\b", ql)
        if m:
            what = re.sub(r"part[- ]?time", "part-time", m.group(0))
            return Answer(f"I couldn't find anything about {what} in the Student Handbook. It describes the five "
                          "BSc (Hons) degree programmes (4 years for civil and 4½ years for military students) "
                          "and does not cover this. Please check with the Faculty office.", 0.9, "not_found")
        return None

    def _answer(self, question: str) -> dict[str, Any]:
        q_raw = (question or "").strip()
        if not q_raw:
            return {"answer": "Please type a question.", "kind": "empty", "confidence": 0.0}
        q = self.correct_spelling(expand_abbreviations(normalize(q_raw)))
        ql = q.lower()
        ctx = self.parse(q, q_raw)
        handlers = [
            self.h_degree_structure, self.h_compare, self.h_course, self.h_semester_courses, self.h_credits, self.h_grades, self.h_classes,
            self.h_programmes, self.h_departments, self.h_people, self.h_contacts, self.h_height,
            self.h_max_duration, self.h_glossary, self.h_section_count, self.h_situations,
        ]
        candidates: list[Answer] = []
        for h in handlers:
            try:
                a = h(ql, ctx)
            except Exception as exc:  # a broken handler must never break the chat
                a = None
                print(f"[handler {h.__name__}] {exc}")
            if a:
                candidates.append(a)
        best = max(candidates, key=lambda a: a.score) if candidates else None
        if not best or best.score < 0.6:
            r = self.retrieve_answer(ql, ctx)
            if r and (not best or r.score >= best.score):
                best = r
        if best:
            best = self.yes_no_layer(ql, ctx, best)
            if best.kind.startswith("not_found") and best.kind != "not_found_amount":
                best = Answer(best.text + self.suggest_topics(ql), best.score, best.kind)
        if not best:
            return {"answer": "I couldn't find a clear answer to that in the Student Handbook. "
                              "Try rephrasing with words used in the handbook (e.g. module code, semester, grade).",
                    "kind": "not_found", "confidence": 0.0, "corrected_question": q if q != normalize(q_raw) else None}
        return {"answer": best.text, "kind": best.kind, "confidence": round(min(best.score, 1.0), 3),
                "page": best.page, "evidence": best.evidence[:400],
                "corrected_question": q if q.lower() != normalize(q_raw).lower() else None,
                "llm_used": False}

    # ------------------------------------------------------------------
    # language processing
    # ------------------------------------------------------------------
    def correct_spelling(self, q: str) -> str:
        def fix(m: re.Match) -> str:
            w = m.group(0)
            lw = w.lower()
            if len(lw) < 5 or lw in self.vocab or lw in STOP or not lw.isalpha():
                return w
            # adjacent-letter swap first ("cerdit" -> "credit", "tauhgt" -> "taught")
            targets = self._target_set if hasattr(self, "_target_set") else set(self.correction_targets)
            self._target_set = targets
            for k in range(len(lw) - 1):
                sw = lw[:k] + lw[k + 1] + lw[k] + lw[k + 2:]
                if sw in targets or sw in ("taught", "credit", "credits", "choose") or sw in self.vocab:
                    return sw
            cand = difflib.get_close_matches(lw, self.correction_targets, n=3, cutoff=0.84)
            cand = [c for c in cand if abs(len(c) - len(lw)) <= 1 and c[0] == lw[0]]  # typos keep length and first letter
            return cand[0] if cand else w
        return re.sub(r"[A-Za-z]+", fix, q)

    def parse(self, q: str, q_raw: str) -> dict[str, Any]:
        ql = q.lower()
        ctx: dict[str, Any] = {"raw": q_raw, "q": q}
        ctx["codes"] = [c.upper() for c in re.findall(r"\b([a-z]{2,3}\s?\d{4})\b", ql)]
        ctx["codes"] = [c.replace(" ", "") for c in ctx["codes"]]
        # programmes (upper-case IT / IS only when typed in capitals)
        progs = programs_in(q)
        if re.search(r"\bIT\b", q_raw) and "IT" not in progs:
            progs.append("IT")
        if re.search(r"\bIS\b", q_raw) and "IS" not in progs:
            progs.append("IS")
        ctx["programs"] = progs
        # semester
        sem = re.search(r"\bsem(?:ester)?\s*0?([1-8])\b", ql) or re.search(r"\bs([1-8])\b", ql)
        if not sem:
            m = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|[1-8](?:st|nd|rd|th))\s+sem", ql)
            ctx["semester"] = ORDINALS.get(m.group(1)) if m else None
        else:
            ctx["semester"] = int(sem.group(1))
        lv = re.search(r"\blevel\s*([1-4])\b", ql) or re.search(r"\byear\s*([1-4])\b", ql)
        m = re.search(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+year\b", ql)
        ctx["level"] = int(lv.group(1)) if lv else (ORDINALS.get(m.group(1)) if m else None)
        # numbers
        ctx["decimals"] = [float(x) for x in re.findall(r"\b\d\.\d{1,2}\b", ql)]
        ctx["ints"] = [int(x) for x in re.findall(r"(?<![\d.])\b(\d{1,3})\b(?![.\d])", ql)]
        # grades e.g. A+, B-, C, Ib
        g = re.findall(r"(?:grade\s+|\b)([abcd][+-]|\b(?:ie|ia|ib|ne|ab|ex)\b)", ql)
        g2 = re.findall(r"grade\s+([abcd])\b|\b([abcd])\s+grade|"
                        r"\b(?:gpv|grade point value|grade point)\s+(?:of|for)\s+(?:an?\s+)?([abcd])\b(?![+-])", ql)
        g2 = [(a or b or c, "") for a, b, c in g2]
        ctx["grades"] = [x.upper() if len(x) == 1 or x[1] in "+-" else x.capitalize() for x in g] + \
                        [(a or b).upper() for a, b in g2]
        ctx["qtype"] = self.qtype(ql)
        return ctx

    @staticmethod
    def qtype(ql: str) -> str:
        if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal number\b", ql):
            return "count"
        if re.search(r"\bhow much\b", ql):
            return "amount"
        if re.search(r"^(list|name|give|show|mention|tell me)\b|\b(names of|name of the (degrees?|modules?|courses?|departments?|programmes?))\b", ql):
            return "list"
        if re.search(r"^who\b|\bwho is\b|\bwho are\b|\bwhose\b", ql):
            return "who"
        if re.search(r"\b(e-?mail|mail address)\b", ql):
            return "email"
        if re.search(r"\b(phone|telephone|tel|contact number|call)\b", ql):
            return "phone"
        if re.search(r"^when\b|\bwhat year\b|\bwhich year\b", ql):
            return "when"
        if re.search(r"^where\b", ql):
            return "where"
        if re.search(r"\b(list|what are|which are|name the|show|mention|what modules|what courses|which modules|which courses)\b", ql):
            return "list"
        if re.search(r"^(is|are|can|does|do|will|shall|should)\b", ql):
            return "yesno"
        if re.search(r"^how\b", ql):
            return "how"
        return "what"

    def expand_query(self, ql: str) -> dict[str, float]:
        weights: dict[str, float] = defaultdict(float)
        for ac, full in self.kb.get("glossary", {}).items():
            if full.lower() in ql:
                weights[stem(ac.lower())] = max(weights[stem(ac.lower())], 0.6)
        self._syn_of: dict[str, set[str]] = defaultdict(set)
        for t in tokens(ql):
            weights[stem(t)] = max(weights[stem(t)], 1.0)
            for syn in THESAURUS.get(t, []) + THESAURUS.get(stem(t), []):
                for s in stems(syn):
                    weights[s] = max(weights[s], 0.4)
                    self._syn_of[stem(t)].add(s)
        return dict(weights)

    # ------------------------------------------------------------------
    # course helpers
    # ------------------------------------------------------------------
    def match_course_acronym(self, q_raw: str) -> str | None:
        """'AI module', 'is HCI compulsory' -> module code (acronym typed in capitals, or followed by module/course)."""
        for a, code in self.course_acronyms.items():
            first = a.split()[0]
            if re.search(rf"\b{re.escape(a)}\b", q_raw) or \
                    re.search(rf"\b{re.escape(first.lower())}\s+(module|course|subject)\b", q_raw.lower()):
                return code
        return None

    def match_course_title(self, ql: str) -> tuple[str, float] | None:
        qt = tokens(ql, drop_stop=False)
        qs = " " + " ".join(qt) + " "
        best = None
        for toks, code in self.title_index:
            phrase = " " + " ".join(toks) + " "
            if phrase in qs:
                score = 1.0 + 0.01 * len(toks)
            else:
                content = [t for t in toks if t not in STOP]
                if len(content) < 2:
                    continue
                hit = sum(1 for t in content if t in qt)
                score = hit / len(content)
                if score < 0.85:
                    continue
                score *= 0.9
            if not best or score > best[1]:
                best = (code, score)
        if best:
            # Avoid treating degree names ("software engineering degree") as a course title.
            title = self.courses[best[0]]["title"].lower()
            if title in ("software engineering", "computer science") or len(title) < 4:
                return None
        return best

    def course_offering(self, code: str, programs: list[str]) -> list[dict[str, Any]]:
        offs = self.courses[code]["offerings"]
        if programs:
            sel = [o for o in offs if o["program"] in programs]
            if sel:
                return sel
        return offs

    def describe_offerings(self, offs: list[dict[str, Any]]) -> str:
        groups: dict[tuple, list[str]] = defaultdict(list)
        for o in offs:
            groups[(o["semester"], o["category"])].append(o["program"])
        parts = []
        for (sem, cat), progs in groups.items():
            progs = sorted(set(progs), key=list(PROGRAMS).index)
            parts.append(f"Semester {sem} ({'/'.join(progs)}, {cat.lower() or 'n/a'})")
        return "; ".join(parts)

    def credit_text(self, code: str, offs: list[dict[str, Any]]) -> str:
        c = self.courses[code]
        vals = []
        for o in offs:
            for typ in ("gpa", "ngpa", "mgpa"):
                if o[typ]:
                    vals.append(f"{fmt_num(o[typ])} {typ.upper()} credits")
        vals = list(dict.fromkeys(vals))
        if not vals:
            for d in c["descriptions"]:
                if d["credits"]:
                    vals.append(f"{d['credits']} {d['credit_type']} credits")
            vals = list(dict.fromkeys(vals))
        if any(o.get("note") for o in offs) and code.endswith("999"):
            notes = [f"Semester {o['semester']}: {fmt_num(o['gpa']) if o['gpa'] else '-'} ({o['note']})" for o in offs]
            return " / ".join(vals) + " - " + "; ".join(dict.fromkeys(notes))
        return " / ".join(vals) if vals else "not stated"

    # ------------------------------------------------------------------
    # intent handlers  (each returns Answer or None)
    # ------------------------------------------------------------------
    def h_course(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        code, conf = None, 0.0
        if ctx["qtype"] == "who":
            return None
        if ctx["codes"]:
            code = next((c for c in ctx["codes"] if c in self.courses), None)
            if not code:
                return Answer(f"I couldn't find module {ctx['codes'][0]} in the handbook.", 0.9, "course_unknown")
            conf = 1.0
        else:
            m = self.match_course_title(ql)
            if m:
                code, conf = m[0], min(m[1], 1.0) * 0.95
            else:
                code = self.match_course_acronym(ctx["raw"])
                conf = 0.93 if code else 0.0
        if not code:
            return None
        if ctx["semester"] and re.search(r"\b(modules?|courses?|subjects?)\b", ql) and not ctx["codes"] and conf < 1:
            return None  # this is a semester listing question
        c = self.courses[code]
        offs = self.course_offering(code, ctx["programs"])
        title = c["title"]
        head = f"{code} - {title}"
        desc = next((d for d in c["descriptions"] if not ctx["programs"] or set(ctx["programs"]) & set(d["programs"])),
                    c["descriptions"][0] if c["descriptions"] else None)
        page = offs[0]["page"] if offs else None
        if re.search(r"\bcredit|\bworth\b|\bweight", ql):
            return Answer(f"{head}: {self.credit_text(code, offs)}.", conf, "course_credits", page)
        if re.search(r"\b(elective|compulsory|optional|mandatory|category|core)\b|\ba military (module|course)\b", ql):
            cats = sorted({o["category"] for o in offs if o["category"]})
            per = self.describe_offerings(offs)
            return Answer(f"{head} is {join_list([c.lower() for c in cats]) or 'not categorised'} - {per}.",
                          conf, "course_category", page)
        if re.search(r"\b(which|what|when)\b.*\b(semester|year|level)\b|\bsemester\b", ql) or ctx["qtype"] == "when":
            return Answer(f"{head} is offered in {self.describe_offerings(offs)}.", conf, "course_semester", page)
        if re.search(r"\b(which|what)\b.*\b(degrees?|programmes?|programs?)\b", ql):
            progs = sorted({o["program"] for o in c["offerings"]}, key=list(PROGRAMS).index)
            return Answer(f"{head} is part of: {join_list([PROGRAMS[p] for p in progs])}.", conf, "course_programs", page)
        if re.search(r"\bcode\b", ql) and not ctx["codes"]:
            return Answer(f"The module code of {title} is {code}.", conf, "course_code", page)
        if re.search(r"\b(name|title|called)\b", ql) and ctx["codes"]:
            return Answer(f"{code} is {title}.", conf, "course_name", page)
        if desc and re.search(r"\b(about|describe|description|cover\w*|topics?|learn\w*|content|aims?|objectives?|"
                              r"teach\w*|taught|language|tools?|software|hardware|use[sd]?|aligned|curriculum|"
                              r"syllabus|include\w*|focus\w*|purpose|goal)\b", ql):
            return Answer(f"{head}: {self.course_aspect(desc['text'], title, ql)}", conf, "course_description", page)
        # default summary
        summary = f"{head}: {self.credit_text(code, offs)}; {self.describe_offerings(offs)}."
        if desc and ctx["qtype"] == "what" and not ctx["codes"]:
            summary += " " + self.shorten(desc["text"], 200)
        return Answer(summary, conf * 0.95, "course_summary", page)

    LANGUAGES = r"C\+\+|\bJava\b|\bPython\b|\bProlog\b|\bC#|\bSQL\b|\bHTML\b|JavaScript|\bPHP\b|\bR\b|MATLAB|Flash"

    def course_aspect(self, desc: str, title: str, ql: str) -> str:
        """Pick the description sentence(s) that answer the specific aspect asked about."""
        sents = [x.strip() for x in re.split(r"(?<=[.;])\s+(?=[A-Z(])", desc) if len(x.strip()) > 10]
        if len(sents) <= 1:
            return self.shorten(desc, 360)
        generic = {"what", "which", "about", "module", "course", "subject", "cover", "covered", "topic", "is", "in",
                   "the", "describe", "tell", "me", "does", "do"}
        residual = set(stems(ql)) - set(stems(title)) - {stem(g) for g in generic}
        wants_topics = re.search(r"\b(cover\w*|topics?|include\w*|content|learn\w*|taught|areas?)\b", ql)
        wants_lang = re.search(r"\b(language|programming language|tool|software)\b", ql)
        wants_names = re.search(r"\b(which|what)\b.*\b(systems?|tools?|languages?|software|platforms?|frameworks?|"
                                r"interpreters?|compilers?)\b", ql)
        best, best_s = None, 0.0
        for i, sent in enumerate(sents):
            sc = len(residual & set(stems(sent)))
            if wants_lang and re.search(self.LANGUAGES, sent):
                sc += 3
            if wants_names:
                qwords = set(re.findall(r"[A-Za-z_]+", ql.lower()))
                sc += sum(1 for w in re.findall(r"\b[A-Z][A-Za-z_]*[A-Z_][A-Za-z_]*\b|\b[A-Z]{2,}\b", sent)
                          if w.lower() not in qwords)
            if wants_topics and re.search(r"\b(topics?|covers?|covered|cover|include\w*|such as|namely)\b|"
                                          r"understanding of|knowledge of", sent, re.I):
                sc += 2
            elif wants_topics and re.search(r"\bintroduc\w*", sent, re.I):
                sc += 1
            if sc > best_s:
                best, best_s = i, sc
        if best is None:
            return self.shorten(desc, 360)
        out = sents[best]
        if len(out) < 90 and best + 1 < len(sents):
            out += " " + sents[best + 1]
        return self.shorten(out, 360)

    def _semester_rows(self, sem: int | None, level: int | None, programs: list[str]):
        rows = []
        for code, c in self.courses.items():
            for o in c["offerings"]:
                if sem and o["semester"] != sem:
                    continue
                if level and not sem and o["level"] != level:
                    continue
                if programs and o["program"] not in programs:
                    continue
                rows.append((code, c["title"], o))
        return rows

    def h_semester_courses(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        sem, level = ctx["semester"], ctx["level"]
        if not (sem or level):
            return None
        if not re.search(r"\b(modules?|courses?|subjects?|units?|electives?|offered|taught|study|learn)\b", ql):
            return None
        if re.search(r"\bcredits?\b", ql) and not re.search(r"\b(modules?|courses?|subjects?)\b.*\b(list|which|what)\b", ql) \
                and ctx["qtype"] != "list":
            return None  # credit totals are handled by h_credits
        progs = [p for p in ctx["programs"] if p in ("CS", "SE")]
        rows = self._semester_rows(sem, level, progs)
        if not rows:
            return Answer(f"The handbook lists no modules for that {'semester' if sem else 'level'}"
                          f"{' in ' + '/'.join(progs) if progs else ''}.", 0.7, "semester_courses")
        filt = ""
        if re.search(r"\belectives?\b|\boptional\b", ql):
            rows = [r for r in rows if r[2]["category"] == "Elective"]
            filt = "elective "
        elif re.search(r"\bcompulsory\b|\bcore\b|\bmandatory\b", ql):
            rows = [r for r in rows if r[2]["category"] == "Compulsory"]
            filt = "compulsory "
        elif re.search(r"\bmilitary\b", ql):
            rows = [r for r in rows if r[2]["category"] == "Military"]
            filt = "military "
        elif re.search(r"\bngpa\b", ql):
            rows = [r for r in rows if r[2]["ngpa"]]
            filt = "NGPA "
        if not rows:  # v11: e.g. electives in Semester 1 - there are none
            scope = f"Semester {sem}" if sem else f"Level {level}"
            return Answer(f"There are no {filt.strip() or 'matching'} modules in {scope}"
                          + (" - every module in that semester is compulsory." if filt.strip() == "elective" else "."),
                          0.9, "semester_courses", None)
        # Group by programme; if CS and SE lists are identical, merge them.
        by_prog: dict[str, list[tuple]] = defaultdict(list)
        for code, title, o in rows:
            if (code, title) not in [(x[0], x[1]) for x in by_prog[o["program"]]]:
                by_prog[o["program"]].append((code, title, o))
        lists = {p: tuple(sorted(x[0] for x in v)) for p, v in by_prog.items()}
        groups: dict[tuple, list[str]] = defaultdict(list)
        for p, key in lists.items():
            groups[key].append(p)
        scope = f"Semester {sem}" if sem else f"Level {level}"
        parts = []
        for key, plist in groups.items():
            items = by_prog[plist[0]]
            names = [f"{c} {t}" for c, t, _ in sorted(items)]
            label = "/".join(sorted(plist, key=list(PROGRAMS).index))
            if ctx["qtype"] == "count":
                parts.append(f"{label}: {len(items)} {filt}modules ({'; '.join(names)})")
            else:
                parts.append(f"{label} ({len(items)}): " + "; ".join(names))
        text = f"{scope} {filt}modules - " + " | ".join(parts) + "."
        return Answer(text, 0.95, "semester_courses", rows[0][2]["page"])

    def h_credits(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        if not re.search(r"\bcredits?\b", ql) or ctx["codes"]:
            return None
        if self.match_course_title(ql) and not re.search(r"\b(semester|degree|total)\b", ql):
            return None
        types = [t for t in ("ngpa", "mgpa", "gpa") if re.search(rf"\b{t}\b", ql)]
        sem = ctx["semester"]
        if sem:
            tots = self.kb["semester_totals"]
            progs = [p for p in ctx["programs"] if p in ("CS", "SE")]
            found = []
            for t in tots:
                sec = self.sections.get(t["section"], {})
                lvl = self._level_section(t["section"])
                tprogs = programs_in(lvl) or ["CS"]
                if t["semester"] == sem and (not progs or set(progs) & set(tprogs)):
                    found.append((tprogs, t))
            if not found:
                return None
            uniq = {}
            for tprogs, t in found:
                uniq.setdefault((t["gpa"], t["ngpa"], t["mgpa"]), []).extend(tprogs)
            parts = []
            for (g, n, m), plist in uniq.items():
                plist = sorted(set(plist), key=list(PROGRAMS).index)
                if types:
                    vals = {"gpa": g, "ngpa": n, "mgpa": m}
                    parts.append(f"{'/'.join(plist)}: " + ", ".join(f"{fmt_num(vals[x])} {x.upper()}" for x in types))
                else:
                    parts.append(f"{'/'.join(plist)}: {fmt_num(g)} GPA, {fmt_num(n)} NGPA, {fmt_num(m)} MGPA credits")
            return Answer(f"Semester {sem} credits - " + "; ".join(parts) + ".", 0.95, "semester_credits",
                          found[0][1]["page"])
        if re.search(r"\b(degree|programme|program|total|graduate|graduation|whole|entire|complete|overall|all)\b", ql) \
                or ctx["programs"]:
            sums = []
            for g in self.kb["gpa_summary"]:
                tprogs = programs_in(self._level_section(g["section"], want="degree")) or ["CS"]
                tot = next((v for k, v in g["rows"].items() if k.lower().startswith("total")), None)
                if tot:
                    sums.append((tprogs, tot, g["page"]))
            progs = ctx["programs"]
            sel = [s for s in sums if not progs or set(progs) & set(s[0])]
            if progs and not sel:
                names = join_list([PROGRAMS[p] for p in progs])
                return Answer(f"The handbook gives a credit breakdown only for Computer Science and Software Engineering, "
                              f"not for {names}.", 0.8, "degree_credits")
            parts = []
            for tprogs, tot, page in sel:
                lab = "/".join(tprogs)
                if types:
                    parts.append(f"{lab}: " + ", ".join(f"{fmt_num(tot[x])} {x.upper()}" for x in types))
                else:
                    parts.append(f"{lab}: {fmt_num(tot['gpa'])} GPA credits (core), {fmt_num(tot['ngpa'])} NGPA and "
                                 f"{fmt_num(tot['mgpa'])} MGPA credits")
            if parts:
                return Answer("Total degree credits - " + "; ".join(parts) + ".", 0.9, "degree_credits", sel[0][2])
        return None

    def _level_section(self, num: str, want: str = "level") -> str:
        s = self.sections.get(num)
        while s:
            if want in s["title"].lower() or programs_in(s["title"]):
                return s["title"]
            s = self.sections.get(s.get("parent")) if s.get("parent") else None
        return ""

    def h_grades(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        rows = self.kb["grades"]["rows"]
        if not rows:
            return None
        page = self.kb["grades"].get("page")
        gradey = re.search(r"\b(grade|grades|grading|gpv|grade point|mark|marks|score|get|got|scored)\b", ql)
        if not gradey:
            return None
        if ctx["qtype"] == "who" or re.search(r"\b(decide|decides|decided|senate|military|mgpa)\b", ql):
            return None
        if ctx["grades"] and re.search(r"\b(how|complete|clear|should i do|what to do)\b", ql) and \
                not re.search(r"\b(gpv|mean|means|marks?)\b", ql):
            g = ctx["grades"][0]  # "how do I clear grade Ia?" -> only sentences that mention that grade
            hit = self.retrieve_scoped(ql + " completed re-sitting", lambda p: re.search(
                rf"[\u2018\u2019\u201c\u201d'\"]\s*{re.escape(g)}\s*[\u2018\u2019\u201c\u201d'\"]", p["text"])
                and p["kind"] != "table_row")
            if hit:
                return Answer(hit[1], 0.9, "grade_procedure", page)
        if re.search(r"\b(how|complete|re-?sit\w*|repeat\w*|happen\w*|upgrad\w*|improv\w*|re-?take|weak pass)\b", ql) and \
                not re.search(r"\b(gpv|how many marks|how much)\b", ql):
            return None  # procedural questions go to retrieval
        # marks -> grade
        marks = [n for n in ctx["ints"] if 0 <= n <= 100]
        marks_ctx = re.search(r"\b(marks?|score|scored|get|got|%|percent)\b", ql) or \
            (re.search(r"\bgrade\b", ql) and not ctx["semester"] and not ctx["codes"] and not ctx["level"])
        if marks and marks_ctx and not re.search(r"\bfgpa|semester|level|year\b", ql):
            m = marks[0]
            for r in rows:
                if r["low"] is not None and r["low"] <= m <= r["high"]:
                    return Answer(f"{m} marks = grade {r['grade']} (GPV {r['gpv']}).", 0.97, "grade_lookup", page)
            if m < 35:
                return Answer(f"{m} marks is below 35, which is a failing component mark (grade Ie/Ia/Ib, GPV 0.00 "
                              f"depending on the failed component).", 0.9, "grade_lookup", page)
        # grade -> marks / GPV
        for w, gcode in {"absent": "Ab", "absence": "Ab", "excused": "Ex"}.items():
            if re.search(rf"\b{w}\b", ql) and not ctx["grades"] and re.search(r"\bgrade\b", ql):
                r = next((r for r in rows if r["grade"] == gcode), None)
                if r:
                    return Answer(f"Grade {gcode} ({r['marks']}), GPV {r['gpv'] or '0.00'}.", 0.93, "grade_lookup", page)
        if ctx["grades"]:
            g = ctx["grades"][0]
            r = next((r for r in rows if r["grade"].lower() == g.lower()), None)
            if r:
                if re.search(r"\bgpv\b|grade point", ql):
                    return Answer(f"Grade {r['grade']} has a GPV of {r['gpv'] or 'not assigned'} ({r['marks']} marks).",
                                  0.97, "grade_gpv", page)
                txt = f"Grade {r['grade']}: {r['marks']}" + (" marks" if r["low"] is not None else "") + \
                      (f", GPV {r['gpv']}." if r["gpv"] else ".")
                return Answer(txt, 0.95, "grade_lookup", page)
        if re.search(r"\b(grading|all grades|grade table|grading system|gpv table|list)\b", ql) or \
                (ctx["qtype"] == "list" and "grade" in ql):
            items = [f"{r['grade']} {r['marks']} ({r['gpv']})" for r in rows if r["low"] is not None]
            return Answer("Grades: " + "; ".join(items) + ". Below 35 or absent = 0.00.", 0.85, "grade_table", page)
        if re.search(r"\b(highest|maximum|top|best)\b.*\b(grade|gpv)\b", ql):
            r = rows[0]
            return Answer(f"The highest grade is {r['grade']} ({r['marks']} marks, GPV {r['gpv']}).", 0.9, "grade_lookup", page)
        return None

    def h_classes(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        rows = self.kb["classes"]["rows"]
        if not rows or not re.search(r"\b(class|fgpa|division|honours|honors|pass degree)\b", ql):
            return None
        page = self.kb["classes"].get("page")
        if re.search(r"\b(merit|trophy|award for best)\b", ql) or re.search(r"\bwithin\b|\bhow many years\b", ql):
            return None
        if ctx["decimals"]:
            v = ctx["decimals"][0]
            for r in rows:
                lo = r["low"] if r["low"] is not None else 0
                hi = r["high"] if r["high"] is not None else 99
                if lo <= v < hi or (r["high"] is None and v >= lo):
                    return Answer(f"An FGPA of {v:.2f} gives: {r['result']} ({r['condition']}).", 0.97, "class_lookup", page)
            return Answer(f"An FGPA of {v:.2f} is below 2.00, the minimum needed to pass the degree.", 0.9, "class_lookup", page)
        target = None
        if re.search(r"first class|1st class", ql):
            target = "First Class"
        elif re.search(r"upper", ql):
            target = "Second Class (Upper Division)"
        elif re.search(r"lower", ql):
            target = "Second Class (Lower Division)"
        elif re.search(r"second class", ql):
            target = "Second Class"
        elif re.search(r"\bpass\b", ql) and "fgpa" in ql:
            target = "Pass"
        if target:
            sel = [r for r in rows if r["result"].startswith(target)]
            if not sel:
                return None
            txt = "; ".join(f"{r['result']}: {r['condition']}" for r in sel)
            # attach the extra conditions listed under the class heading
            cond = self.class_conditions(sel[0]["result"]) if len(sel) == 1 else ""
            return Answer(txt + (". " + cond if cond else "."), 0.95, "class_criteria", page)
        if re.search(r"\b(classes|class criteria|awarding classes|how .*class)\b", ql) or ctx["qtype"] in ("list", "count"):
            txt = "; ".join(f"{r['result']}: {r['condition']}" for r in rows)
            return Answer(f"Classes are awarded by FGPA - {txt}.", 0.85, "class_table", page)
        return None

    def class_conditions(self, result: str) -> str:
        for s in self.kb["sections"]:
            if s["number"] in ("6.1.2",):
                txt = s["text"]
                start = txt.rfind("For the award of a " + result)
                if start < 0:
                    continue
                chunk = txt[start:].split("\n")
                items = []
                for ln in chunk[1:]:
                    if re.match(r"^(Second Class|First Class|\.$|6\.)", ln.strip()) and items:
                        break
                    items.append(ln.strip())
                body = " ".join(items)
                body = re.sub(r"\s+and\s+(?=[a-d]\.)", " ", body)
                parts = [p.strip(" .") for p in re.split(r"\b[a-d]\.\s", body) if p.strip(" .")]
                parts = [re.sub(r"\s+and$", "", p) for p in parts]
                return "Conditions: " + "; ".join(parts[:4]) + "."
        return ""

    def h_programmes(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        progs = self.kb["programmes"]
        if not progs:
            return None
        about_prog = re.search(r"\b(degrees?|programmes?|programs?|courses?|bsc)\b", ql)
        # duration
        if re.search(r"\b(duration|how long|years?|length)\b", ql) and \
                not re.search(r"\bmax|maximum|extend|within\b|how long can|take to (finish|complete)", ql) \
                and (about_prog or ctx["programs"]) and not re.search(r"\b(established|founded)\b", ql):
            sel = [progs[p] for p in ctx["programs"] if p in progs] or list(progs.values())
            if all(p.get("military_years") for p in sel):
                same = len({(p["military_years"], p["civil_years"]) for p in sel}) == 1
                if same:
                    p = sel[0]
                    who = join_list([x["name"] for x in sel]) if len(sel) < len(progs) else "All five degree programmes"
                    return Answer(f"{who}: {p['civil_years'].replace(' Years', '')} years for civil students and {p['military_years']} years for "
                                  f"military students.", 0.93, "programme_duration", p.get("page"))
        # streams / eligibility
        if re.search(r"\b(stream|a/?l|advanced level|eligib|qualif|apply|entry|enter|admission|requirements?)\b", ql) \
                and (ctx["programs"] or about_prog) and not re.search(r"\b(military|height|weight|age|chest|o/?l|english|general test)\b", ql):
            sel = [progs[p] for p in ctx["programs"] if p in progs] or list(progs.values())
            by_stream: dict[str, list[str]] = defaultdict(list)
            for p in sel:
                by_stream[p.get("stream", "")].append(p["code"])
            parts = [f"{'/'.join(v)}: {k} stream" for k, v in by_stream.items()]
            extra = ""
            if any(p["code"] in ("CS", "SE", "CE") for p in sel):
                extra = (" CS/SE/CE need the Maths stream, or Mathematics, Physics and one of Chemistry/Higher "
                         "Mathematics/ICT.")
            if any(p["code"] in ("IT", "IS") for p in sel):
                extra += " IT/IS accept Biology, Maths, Commerce or Arts (not Technology)."
            return Answer("A/L stream - " + "; ".join(parts) + "." + extra +
                          " A minimum of three S passes is required.", 0.9, "programme_stream", sel[0].get("page"))
        # how many / list degree programmes
        if re.search(r"\b(degrees?|programmes?|programs?|courses)\b", ql) and \
                re.search(r"\b(offer|offered|offers|available|faculty|foc|provide|conduct)\b", ql) and \
                ctx["qtype"] != "yesno" and \
                not ctx["semester"] and not ctx["programs"] and not re.search(r"\bdepartment\b", ql):
            if re.search(r"\b(modules?|subjects?|units?)\b", ql) or any(self.is_oov(stem(t)) for t in tokens(ql)):
                return None
            names = [p["name"] for p in progs.values()]
            sec = next((s for s in self.kb["sections"] if "courses offered" in s["title"].lower()), None)
            lead = f"The Faculty of Computing offers {len(names)} degree programmes"
            return Answer(f"{lead}: {join_list(names)}.", 0.96 if ctx["qtype"] in ("count", "list", "what") else 0.85,
                          "programme_list", sec["page"] if sec else None)
        # describe one programme
        if ctx["programs"] and len(ctx["programs"]) == 1 and \
                re.search(r"\b(about|describe|what is|overview|focus|cover\w*|include\w*|areas?|learn|study|teach\w*)\b", ql) \
                and about_prog and not re.search(r"\b(career|job)\b", ql):
            p = progs.get(ctx["programs"][0])
            if p and p.get("description"):
                desc = re.sub(r"\s+", " ", p["description"])
                first = re.split(r"(?<=\.)\s+(?=[A-Z])", desc)[0]
                aspect = self.course_aspect(desc, p["name"], ql)
                text = aspect if aspect.startswith(first[:40]) else self.shorten(first, 220) + " " + aspect
                return Answer(f"{p['name']}: {self.shorten(text, 460)}", 0.8, "programme_about")
        return None

    def h_departments(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        if not re.search(r"\b(departments?|dept)\b", ql):
            return None
        if re.search(r"\b(head|hod|lecturers?|staff|who|instructors?|members?)\b", ql) and \
                not re.search(r"\bwhich department offers\b", ql):
            return None  # people handler
        depts = self.kb.get("departments", {})
        asks_degrees = re.search(r"\b(degrees?|programmes?|programs?|bsc|offer\w*)\b", ql) and \
            not re.search(r"\b(modules?|subjects? areas?)\b|\b(type|types|kind|kinds|nature) of\b", ql)
        # inverse: which department offers <degree>?
        if re.search(r"\b(which|what)\s+department\b", ql) and ctx["programs"]:
            p = ctx["programs"][0]
            d = next((d for d in depts.values() if p in d["degrees"]), None)
            if d:
                return Answer(f"The {PROGRAMS[p]} degree is offered by the {d['name']}.", 0.96, "department_of_degree",
                              d["page"])
        dept = self.match_department(ql)
        whole_faculty = re.search(r"\bdepartments\b", ql) or not dept
        if whole_faculty and (ctx["qtype"] in ("count", "list") or re.search(r"\bdepartments\b", ql)) and \
                not (dept and asks_degrees):
            sec = next((s for s in self.kb["sections"] if "academic departments" in s["title"].lower()), None)
            if sec:
                names = [self.sections[c]["title"] for c in sec["children"]]
                return Answer(f"The faculty has {len(names)} departments: {join_list(names)}.", 0.95,
                              "department_list", sec["page"])
        if not dept:
            return None
        d = depts.get(dept, {"name": DEPARTMENTS[dept], "degrees": [], "subject_areas": "", "page": None})
        sec = next((s for s in self.kb["sections"] if s["title"] == DEPARTMENTS[dept] and s["number"].startswith("1.")), None)
        if asks_degrees or re.search(r"\bsubject areas?\b", ql):
            if d["degrees"] and not re.search(r"\bsubject areas?\b", ql):
                names = [PROGRAMS[p] for p in d["degrees"]]
                n = len(names)
                lead = f"The {d['name']} offers {n} degree programme{'s' if n > 1 else ''}"
                return Answer(f"{lead}: {join_list(names)}.", 0.97, "department_degrees", d["page"])
            if d["subject_areas"]:
                no_deg = "" if d["degrees"] else " It does not offer its own degree programme;"
                return Answer(f"The {d['name']} teaches three subject areas: {d['subject_areas']}.{no_deg}"
                              f" its courses provide the Mathematics and Statistics needed for the faculty's degrees."
                              if not d["degrees"] else f"The {d['name']} teaches: {d['subject_areas']}.",
                              0.93, "department_subjects", d["page"])
        if sec and (ctx["qtype"] == "when" or re.search(r"\bestablish|found|start", ql)):
            m = re.search(r"established on ([^.]*?\d{4})", sec["text"]) or \
                re.search(r"(recently established|newly established|oldest)[^.]*", sec["text"])
            if m:
                return Answer(f"{DEPARTMENTS[dept]}: {m.group(0)}.", 0.88, "department_info", sec["page"])
        if sec:
            hit = self.retrieve_scoped(ql, lambda p: p["section"] == sec["number"])
            if hit:
                return Answer(hit[1], 0.82, "department_info", sec["page"])
            return Answer(self.shorten(sec["text"].replace("\n", " "), 300), 0.65, "department_info", sec["page"])
        return None

    def match_department(self, ql: str) -> str | None:
        for code, pats in DEPT_PATTERNS.items():
            if any(re.search(p, ql) for p in pats):
                return code
        return None

    def match_person(self, ql: str) -> dict[str, Any] | None:
        best, best_s = None, 0.0
        qt = set(tokens(ql, drop_stop=False))
        for st in self.kb["staff"]:
            parts = [p for p in re.sub(r"\(.*?\)", "", st["name"]).replace(".", " ").split()
                     if (len(p) > 2 or p.isupper()) and p.lower() not in ("mrs", "miss", "eng", "maj", "de", "silva",
                                                                          "dr", "mr", "ms")]
            if not parts:
                continue
            surname = parts[-1].lower()
            s = 0.0
            local = st["email"].split("@")[0].lower() if st["email"] else ""
            if local and (local in qt or st["email"].lower() in ql):
                s = 0.95
            elif surname in qt:
                s = 0.8 + 0.1 * sum(1 for p in parts[:-1] if p.lower() in qt)
                initials = [p.lower() for p in parts[:-1] if p.isupper() and len(p) <= 5]
                if initials and any(i in qt for i in initials):
                    s += 0.15  # "Eng. DB Wijesinghe" vs "Mrs. PRD Wijesinghe"
            else:
                close = difflib.get_close_matches(surname, list(qt), n=1, cutoff=0.85)
                if close:
                    s = 0.7
            if s > best_s:
                best, best_s = st, s
        return best if best_s >= 0.7 else None

    TOPIC_RE = (r"\b(?:research(?:es|ing|ers?)?(?:\s+(?:in|on))?|works?\s+(?:in|on)|working\s+(?:in|on)|"
                r"speciali[sz]\w*\s+in|experts?\s+in|expertise\s+in|interested\s+in)\s+(.+?)\??$")

    def h_people(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        staff = self.kb["staff"]
        appts = self.kb["appointments"]
        # 1) role questions: dean / head of department / vice chancellor ...
        if re.search(r"\b(head|hod)\b", ql) and not re.search(r"\bheight\b", ql):
            dept = self.match_department(ql)
            if dept:
                for st in staff:
                    if "head" in st["role"].lower() and DEPARTMENTS[dept].split(" of ")[1].lower() in st["role"].lower():
                        return self._person_answer(st, ql, f"Head of the {DEPARTMENTS[dept]}")
            if dept is None and re.search(r"\bdepartments?\b", ql):
                heads = [f"{DEPARTMENTS[c]}: {st['name']}" for c in DEPARTMENTS for st in staff
                         if "head" in st["role"].lower() and DEPARTMENTS[c].split(" of ")[1].lower() in st["role"].lower()]
                if heads:
                    return Answer("Heads of department - " + "; ".join(heads) + ".", 0.85, "department_heads")
        role_map = [
            (r"\bdeputy vice[- ]?chancellor\b.*\bacademic|\bdvc\b.*academic", "Deputy Vice Chancellor (Academic)"),
            (r"\bdeputy vice[- ]?chancellor\b.*\b(defen[cs]e|admin)", "Deputy Vice Chancellor (Defense"),
            (r"\bdeputy vice[- ]?chancellor\b", "Deputy Vice Chancellor"),
            (r"\bvice[- ]?chancellor\b|\bvc\b", "Vice Chancellor"),
            (r"\bchancellor\b", "Chancellor"),
            (r"\brector\b", "Rector"),
            (r"\bdeputy registrar\b", "Deputy Registrar"),
            (r"\bregistrar\b", "Registrar"),
            (r"\bforeign relations?\b", "Foreign Relations Officer"),
            (r"\bdean\b.*\bgraduate\b", "Dean Faculty of Graduate Studies"),
            (r"\bdean\b.*\b(defen[cs]e|strategic)\b", "Dean Faculty of Defense"),
        ]
        for pat, role in role_map:
            if re.search(pat, ql):
                cand = [a for a in appts if a["role"].startswith(role)]
                if role == "Vice Chancellor" or role == "Chancellor":
                    cand = [a for a in appts if a["role"] == role or a["role"].startswith(role + " ")
                            and "deputy" not in a["role"].lower()]
                    cand = [a for a in cand if not a["role"].lower().startswith("vice") or role == "Vice Chancellor"]
                if cand:
                    a = cand[0]
                    if ctx["qtype"] == "phone" or re.search(r"\b(tel|phone|contact|number|call)\b", ql):
                        nums = ", ".join(f"{k} {a[k]}" for k in ("tel", "mobile", "fax") if a.get(k))
                        return Answer(f"{a['role']} ({a['name']}): {nums or 'no number listed'}.", 0.95, "appointment",
                                      a["page"])
                    return Answer(f"The {a['role']} is {a['name']}.", 0.95, "appointment", a["page"])
        if re.search(r"\bmessage\b", ql) and re.search(r"\b(who|whose|wrote|written|author|by)\b", ql):
            author = self.kb["contacts"].get("dean_message_author", "")
            if author:
                return Answer(f"The message in the handbook is the Message of the Dean, written by {author.split(' PhD')[0]}.",
                              0.92, "message_author")
        if re.search(r"\bdean\b", ql):
            dean = next((st for st in staff if "dean" in st["role"].lower()), None)
            a = next((a for a in appts if a["role"].startswith("Dean Faculty of Computing")), None)
            if dean:
                ans = self._person_answer(dean, ql, "Dean of the Faculty of Computing")
                if a and ans.kind == "person_phone" and a.get("tel"):
                    ans.text = f"Dean of the Faculty of Computing ({dean['name']}): Tel {a['tel']}."
                return ans
        # 2) a named person
        person = self.match_person(ql)
        if person:
            return self._person_answer(person, ql, None)
        # 3) staff counting / listing per department
        if re.search(r"\b(boss|head|leader)\b.*\b(university|kdu)\b", ql) and not re.search(r"\bdepartment\b", ql):
            vc = next((a for a in appts if a["role"] == "Vice Chancellor"), None)
            if vc:
                return Answer(f"The head of the university is the Vice Chancellor, {vc['name']}.", 0.9, "appointment",
                              vc["page"])
        if re.search(r"\bwho\b.*\b(teach\w*|works? in|members?)\b", ql) and self.match_department(ql) and \
                re.search(r"\bdepartment\b", ql):
            ql = ql + " staff list"
            ctx = {**ctx, "qtype": "list"}
        if re.search(r"\b(lecturers?|staff|instructors?|academics?|teachers?)\b", ql):
            dept = self.match_department(ql)
            sel = [s for s in staff if not dept or s["department"] == dept]
            # research-topic search: "who teaches / researches machine learning"
            topic = re.search(self.TOPIC_RE, ql)
            if topic:
                return self._research_search(topic.group(1), sel)
            if ctx["qtype"] == "count":
                lect = [s for s in sel if re.search(r"lecturer", s["designation"], re.I)]
                instr = [s for s in sel if re.search(r"instructor", s["designation"], re.I)]
                where = DEPARTMENTS[dept] if dept else "the Faculty of Computing"
                return Answer(f"{where} lists {len(sel)} academic staff ({len(lect)} lecturers and {len(instr)} instructors).",
                              0.85, "staff_count")
            if ctx["qtype"] in ("list", "who") and dept:
                names = [f"{s['name']} ({s['designation']})" for s in sel]
                return Answer(f"{DEPARTMENTS[dept]} staff: " + "; ".join(names) + ".", 0.85, "staff_list")
        topic = re.search(r"\bwho\b.*?" + self.TOPIC_RE, ql)
        if topic:
            return self._research_search(topic.group(1), staff)
        return None

    def _research_search(self, topic: str, pool: list[dict[str, Any]]) -> Answer:
        topic = re.sub(r"\b(the|a|an|area|areas|field)\b", " ", topic.lower()).strip(" ?.")
        tq = [stem(t) for t in tokens(topic)]
        exact, partial = [], []
        for s in pool:
            ra = s["research_areas"].lower()
            if topic and topic in ra:
                exact.append(s)
                continue
            areas = [set(stems(a)) for a in re.split(r",|;", s["research_areas"])]
            if tq and any(set(tq) <= a for a in areas):
                exact.append(s)
            elif tq and any(len(set(tq) & a) / len(tq) >= 0.5 for a in areas):
                partial.append(s)
        hits = exact or partial
        if not hits:
            return Answer(f"No staff member lists '{topic}' as a research area.", 0.75, "research_search")
        names = [f"{s['name']} ({s['department']})" for s in hits[:10]]
        label = "Staff researching" if exact else "Staff with related research areas for"
        return Answer(f"{label} {topic}: " + "; ".join(names) + ".", 0.9, "research_search")

    def _person_answer(self, st: dict[str, Any], ql: str, role_label: str | None) -> Answer:
        name = st["name"]
        if re.search(r"\b(e-?mail|mail)\b", ql):
            return Answer(f"{name}'s email is {st['email'] or 'not listed'}.", 0.96, "person_email", st["page"])
        if re.search(r"\bresearch|interest|area", ql):
            return Answer(f"{name}'s research areas: {st['research_areas'] or 'not listed'}.", 0.95, "person_research",
                          st["page"])
        if re.search(r"\b(qualification|degree|phd|studied|education|masters?)\b", ql):
            return Answer(f"{name}: {self.shorten(st['qualifications'], 300)}", 0.93, "person_qualifications", st["page"])
        if re.search(r"\b(designation|position|post|title|rank|grade)\b", ql) and not role_label:
            return Answer(f"{name} is a {st['designation']}{' (' + st['role'] + ')' if st['role'] else ''}.", 0.93,
                          "person_designation", st["page"])
        if re.search(r"\b(phone|tel|telephone|contact number)\b", ql):
            return Answer(f"{name}: {st['email'] or ''} (no personal phone listed).", 0.8, "person_phone", st["page"])
        if role_label:
            return Answer(f"The {role_label} is {name}.", 0.96, "person_role", st["page"])
        dept = DEPARTMENTS.get(st["department"], "")
        return Answer(f"{name} - {st['designation']}{', ' + st['role'] if st['role'] else ''}, {dept}. "
                      f"Email: {st['email']}.", 0.9, "person_info", st["page"])

    def h_contacts(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        if any(self.is_oov(stem(t)) for t in tokens(ql)):
            return None  # asks about something the handbook never mentions (cafeteria, hostel, ...)
        c = self.kb["contacts"]
        if re.search(r"\b(dean|head|vice|registrar|rector|chancellor)\b", ql) or self.match_person(ql):
            return None
        if re.search(r"\b(e-?mail|mail)\b", ql):
            if re.search(r"\b(southern|sooriyawewa)\b", ql):
                e = next((x for x in c["emails"] if "southern" in x.lower()), "")
                return Answer(f"Southern Campus email: {e}.", 0.9, "contact")
            if re.search(r"\buniversity\b|\bkdu\b", ql) and not re.search(r"\bfaculty\b", ql):
                e = next((x for x in c["emails"] if "kdudefence" in x.lower()), "")
                return Answer(f"University email: {e}.", 0.9, "contact")
            return Answer(f"Faculty of Computing email: {c['faculty_email']}.", 0.9, "contact", 2)
        if re.search(r"\b(phone|telephone|tel|contact|call)\b|\b(phone|contact|telephone|mobile) number\b", ql):
            if re.search(r"\b(southern|sooriyawewa)\b", ql) and c.get("southern_tel"):
                return Answer(f"Southern Campus telephone: {c['southern_tel']}.", 0.92, "contact")
            if re.search(r"\buniversity\b|\bkdu\b", ql) and not re.search(r"\bfaculty|department\b", ql) \
                    and c.get("university_tel"):
                return Answer(f"University (KDU, Ratmalana) telephone: {c['university_tel']}; email "
                              f"{next((x for x in c['emails'] if 'kdudefence' in x), '')}.", 0.92, "contact")
            return Answer(f"Dean, Faculty of Computing: {c['dean_phone']}; Department of Computer Science: "
                          f"{c['dept_phone']}; email {c['faculty_email']}.", 0.9, "contact", 2)
        if re.search(r"\b(web ?site|web page|url)\b", ql) or \
                (re.search(r"\bonline\b", ql) and re.search(r"\b(where|find|visit|access|see)\b", ql)):
            return Answer(f"Faculty web page: {c['faculty_web']} (university: {c['university_web']}).", 0.9, "contact", 2)
        if re.search(r"\b(address|postal|post)\b", ql):
            return Answer(f"Postal address: {c['postal']}", 0.9, "contact", 2)
        if re.search(r"^where\b|\blocat|\bsituated\b", ql) and re.search(r"\b(kdu|university)\b", ql) and \
                not re.search(r"faculty|foc|computing", ql) and c.get("university_address"):
            return Answer(f"KDU is located at {c['university_address']} The Faculty of Computing is at the Southern Campus, "
                          f"Sooriyawewa.", 0.9, "university_location")
        if re.search(r"^where\b.*\b(faculty|foc|located|situated)\b|\blocat", ql) and re.search(r"faculty|foc|computing", ql):
            sec = next((s for s in self.kb["sections"] if s["title"] == "Faculty of Computing"), None)
            if sec:
                m = re.search(r"In 2015.*?Sooriyawewa\.", sec["text"].replace("\n", " "))
                if m:
                    return Answer(m.group(0), 0.88, "faculty_location", sec["page"])
        return None

    def h_height(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        h = self.kb["heights"]
        if not h or not re.search(r"\b(height|tall)\b", ql):
            return None
        gender = "female" if re.search(r"\b(female|women|woman|girl|girls|ladies)\b", ql) else \
            ("male" if re.search(r"\b(male|men|man|boy|boys)\b", ql) else None)
        service = next((s for s in ("army", "navy", "air force") if s in ql), None)
        items = [(k, v) for k, v in h.items() if (not gender or k.startswith(gender + "|")) and (not service or k.endswith("|" + service))]
        txt = "; ".join(f"{k.split('|')[0].title()} {k.split('|')[1].title()}: {v}" for k, v in items)
        return Answer(f"Minimum height (military students) - {txt}.", 0.95, "height")

    def h_max_duration(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        md = self.kb["max_duration"]
        if not md or not re.search(r"\b(max|maximum|longest|within|time limit|allowed duration|how many years.*complete)\b"
                                   r"|how long can|take to (finish|complete)|deadline to (finish|complete)", ql):
            return None
        if not re.search(r"\b(duration|years?|time|period|complete|study|degree|semesters?)\b", ql):
            return None
        if re.search(r"\bhonou?rs\b|\bacademic years\b", ql):
            return None  # "within four academic years" (Hons rule) is answered from the text
        if re.search(r"\b(who|extend\w*|extension|discontinu\w*)\b", ql):
            return None  # authority / discontinuation questions are answered from the rule text
        sel = {p: v for p, v in md.items() if not ctx["programs"] or p in ctx["programs"]}
        vals = set(sel.values())
        if len(vals) == 1:
            v = vals.pop()
            who = "any degree programme" if len(sel) == len(md) else join_list([PROGRAMS[p] for p in sel])
            return Answer(f"The maximum allowed duration for {who} is {v} years ({int(v) * 2} semesters). "
                          f"Extensions may be granted on medical or exceptional grounds.", 0.95, "max_duration")
        return Answer("Maximum durations: " + "; ".join(f"{p}: {v} years" for p, v in sel.items()) + ".", 0.9, "max_duration")

    def h_glossary(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        acr = None
        pats = [r"^(?:what\s+(?:is|are)\s+(?:the\s+|an?\s+)?|define\s+|meaning of\s+(?:the\s+)?|full form of\s+)([a-z/]{2,6})\s*\??$",
                r"^what\s+(?:does|do)\s+(?:the\s+)?([a-z/]{2,6})\s+(?:mean|stand for)\s*\??$",
                r"^([a-z/]{2,6})\s+(?:means?|stands? for)\s*\??$"]
        for p in pats:
            m = re.search(p, ql.strip())
            if m:
                acr = m.group(1).upper()
                break
        if not acr or acr not in self.kb["glossary"]:
            return None
        full = self.kb["glossary"][acr]
        extra = ""
        if acr in ("GPA", "SGPA", "YGPA", "FGPA", "MGPA", "NGPA"):
            hits = [p for p in self.passages if acr in p["text"] and len(p["text"]) < 300 and
                    re.search(r"consider|calculat|assigned", p["text"])]
            if hits:
                extra = " " + hits[0]["text"]
        return Answer(f"{acr} stands for {full}.{extra}", 0.9, "glossary")

    def h_section_count(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        """Generic: 'how many X' / 'list X' where X is a section whose children enumerate items."""
        if ctx["qtype"] not in ("count", "list"):
            return None
        qs = set(stems(ql))
        best, best_s = None, 0.0
        for s in self.kb["sections"]:
            if len(s["children"]) < 2:
                continue
            ts = set(stems(s["title"]))
            if not ts:
                continue
            ov = len(ts & qs) / len(ts)
            if ov > best_s:
                best, best_s = s, ov
        q_content = qs - {"many", "number", "list", "name", "offer", "total"}
        back = len(set(stems(best["title"])) & q_content) / max(len(q_content), 1) if best else 0
        if best and best_s >= 0.66 and back >= 0.6:
            names = [self.sections[c]["title"] for c in best["children"]]
            if ctx["qtype"] == "count":
                return Answer(f"{len(names)} - {join_list(names)}.", 0.75 * best_s + 0.1, "section_count", best["page"])
            return Answer(f"{best['title']}: {join_list(names)}.", 0.7 * best_s + 0.1, "section_list", best["page"])
        return None

    # ------------------------------------------------------------------
    # retrieval fallback with answer-type re-ranking
    # ------------------------------------------------------------------
    def retrieve_answer(self, ql: str, ctx: dict[str, Any]) -> Answer | None:
        qw = self.expand_query(ql)
        for p in ctx.get("programs", []):
            qw["prog" + p.lower()] = 1.0
        if not qw:
            return None
        hits = self.bm25.search(qw, k=25)
        if not hits:
            return None
        qtype = ctx["qtype"]
        rescored = []
        max_possible = sum(self.bm25.idf(t) * w for t, w in qw.items()) * (self.bm25.k1 + 1) / 2 or 1.0
        q_terms = {t for t, w in qw.items() if w >= 1.0}
        oov = [t for t in q_terms if self.is_oov(t)]  # topic words absent from the whole handbook
        n_topic = len({t for t in q_terms if t not in self.VERBS})
        threshold = 0.67 if oov else (0.5 if n_topic <= 2 else 0.34)
        for s, i in hits:
            p = self.passages[i]
            txt = p["text"]
            bonus = 1.0
            wants_number = qtype in ("count", "amount") or re.search(
                r"\b(minimum|maximum|least|limit|pass mark|percentage|how long|how old|age|weight|weightage)\b", ql) or re.search(
                r"\bwhat\b.*\b(gpa|fgpa|sgpa|marks?|percentage|grade point)\b.*\b(need\w*|required|requirement)\b", ql)
            has_num = re.search(r"\d", txt) or (re.search(r"\bwithin\b|\bhow many (years?|semesters?|months?|times?)\b", ql) and re.search(
                r"\b(one|two|three|four|five|six|seven|eight|nine|ten|twelve|sixteen)\b", txt, re.I))
            if wants_number and has_num:
                bonus += 0.25
            if wants_number and "%" in txt and re.search(r"\b(marks?|percentage|pass|minimum|attendance|weight\w*)\b", ql):
                bonus += 0.3
            elif wants_number and qtype != "yesno" and not has_num:
                bonus -= 0.3  # the expected answer type (a number) is missing
            if qtype == "when" and re.search(r"\b(19|20)\d{2}\b", txt):
                bonus += 0.3
            elif qtype == "when" and re.search(r"\b(at the (completion|end)|upon|after|following the|within|"
                                               r"before|(first|second|third|fourth|final) year|"
                                               r"\d+(st|nd|rd|th) semester)\b", txt, re.I):
                bonus += 0.35
            if qtype in ("when", "who", "how") and (p["kind"] == "table_row" or re.match(r"Table\s*\d+", txt)) and \
                    not re.search(r"\b(19|20)\d{2}\b|Dr\.|Mr\.|Mrs\.", txt):
                bonus -= 0.25  # numeric table rows rarely answer when/who/how questions
            if qtype == "who" and re.search(r"\b(students?|Board|Senate|BOM|BoS|Dean|Registrar|Council|"
                                            r"Vice Chancellor|Committee|registry)\b", txt, re.I):
                bonus += 0.3
            if qtype == "where" and re.search(r"\b(refer|section|By-?Laws?|Road|located|campus)\b", txt, re.I):
                bonus += 0.3
            if qtype == "email" and "@" in txt:
                bonus += 0.4
            if qtype == "phone" and re.search(r"\d{7,}", txt):
                bonus += 0.4
            if qtype == "how" and re.search(r"formula|calculated|shall be|by re-?sitting|as follows|procedure", txt, re.I):
                bonus += 0.35
            if re.search(r"%|\d", ql) and re.search(r"%|\d", txt):
                bonus += 0.1
            if len(txt) < 40:
                bonus -= 0.2
            if p["section"] == "0":
                bonus -= 0.15  # front matter repeats facts stated in the body
            hm = re.match(r"([A-Z][a-z]+)\s+(?:To|The|In|A|An|This|Our)\b", txt)  # "Vision To be ...", "Mission To ..."
            if hm and stem(hm.group(1).lower()) in q_terms:
                bonus += 0.3  # passage starts with the asked concept, e.g. "Vision ...", "Mission ..."
            if re.match(r"(However|But|Although|Nevertheless)\b", txt) and \
                    not re.search(r"\b(medical|exception|except|valid|reason|sick|illness|special)\b", ql):
                bonus -= 0.3  # prefer the main rule over its exception
            if re.search(r"\bwhat (activities|things|items|components|topics|parts)\b|\bwhich (activities|topics)\b", ql) \
                    and txt.count(",") >= 3:
                bonus += 0.3
            # coverage of the original (non-expanded) query terms
            cov = self.coverage(q_terms, self.bm25.tf[i])
            # subject mismatch: the sentence is about another entity than the one asked about
            subj = re.split(r"\s(?:was|is|has|are|were)\s", txt, maxsplit=1)[0]
            subj_ent = set(stems(subj)) & self.ENTITY_TERMS
            if subj_ent and (set(qw) & self.ENTITY_TERMS) and (subj_ent - set(qw)):
                bonus -= 0.3
            rescored.append((cov >= threshold, s * bonus * (0.6 + 0.6 * cov), cov, i))
        # passages with enough evidence always outrank passages without it
        rescored.sort(reverse=True)
        if self.debug:
            for r in rescored[:5]:
                print("   ", r[0], round(r[1], 2), round(r[2], 2), self.passages[r[3]]["text"][:70])
        _, score, cov, i = rescored[0]
        p = self.passages[i]
        text = p["text"]
        limit = 420
        # complete list-introducing sentences ("... the following conditions.") with their items
        nxt = self.passages[i + 1] if i + 1 < len(self.passages) else None
        if p["kind"] != "list_item" and \
                (text.rstrip().endswith(":") or re.search(r"\bfollowing\b[^.]*[.:,]?\s*$", text, re.I)) and nxt \
                and nxt["section"] == p["section"]:
            if nxt.get("group"):
                items = [x["text"] for x in self.passages if x.get("group") == nxt["group"]]
            else:
                items = [x["text"] for x in self.passages[i + 1:i + 4] if x["section"] == p["section"]]
            text = text + " " + " ".join(self.pick_items(items, q_terms, text))
            limit = 700
        elif p.get("group") and qtype == "count" and not re.search(
                r"how many\s+(years?|months?|weeks?|days?|hours?|credits?|marks?|semesters?|times?|percent)", ql):
            sib = [x for x in self.passages if x.get("group") == p["group"]]
            text = f"{len(sib)}: " + " ".join(x["text"] for x in sib)
            limit = 600
        elif p.get("group"):
            sib = [x for x in self.passages if x.get("group") == p["group"]]
            own = set(stems(text))
            ctx_stems = set(stems(p.get("context", "")))
            ctx_only = {t for t in q_terms if not ({t} | self._syn_of.get(t, set())) & own
                        and ({t} | self._syn_of.get(t, set())) & ctx_stems}
            extra = {t for t in q_terms if t not in ctx_stems and t not in self.VERBS}
            if ctx_only and p.get("context"):
                # the item matched only through its intro sentence: answer with the intro + the items
                text = p["context"] + " " + " ".join(self.pick_items([x["text"] for x in sib], q_terms, p["context"]))
                limit = 700
            elif p.get("context") and extra & set(stems(text)) and len(sib) > 3 and \
                    re.search(r"\bfollowing\b|:$", p["context"]):
                text = p["context"] + " " + text  # one matching item of a long list: keep its intro
            elif len(sib) <= 3 and sum(len(x["text"]) for x in sib) < 420:
                text = " ".join(x["text"] for x in sib)
        elif len(text) < 70 and nxt and nxt["section"] == p["section"]:
            text = text + " " + nxt["text"]
        elif nxt and nxt["section"] == p["section"] and re.match(r"However\b", nxt["text"]) and \
                len(text) + len(nxt["text"]) < 520:
            text = text + " " + nxt["text"]  # keep the qualifying exception with the main rule
            limit = 540
        conf = min(0.95, 0.35 + 0.5 * cov + 0.1 * min(score / max_possible * 4, 1))
        # money questions need an actual amount in the evidence
        if re.search(r"\b(fees?|cost|costs|price|payment|pay|how much)\b", ql) and \
                re.search(r"\b(fees?|cost|price|pay|payment|money|rs|rupees|charge)\b", ql) and \
                not re.search(r"\b(rs\.?|lkr|rupees)\s*\d|\$\s*\d|\d{1,3},\d{3}", text, re.I):
            related = f" The only related rule it gives is: {self.shorten(text, 200)}" if cov >= threshold else ""
            return Answer("I couldn't find that amount in the Student Handbook; it does not list fees." + related,
                          0.7, "not_found_amount")
        if qtype == "when" and cov < 0.9 and not re.search(
                r"\b(19|20)\d{2}\b|\b(january|february|march|april|may|june|july|august|september|october|november|"
                r"december|semesters?|years?|weeks?|days?|after|before|within|upon|end|completion|following)\b",
                text, re.I):
            cov = 0.0  # a "when" question needs a time expression in the answer
        if cov < threshold:
            return Answer("I couldn't find a clear answer to that in the Student Handbook. "
                          "Try rephrasing using handbook terms (e.g. attendance, re-sit, FGPA, module code).",
                          0.2, "not_found")
        return Answer(self.shorten(text, limit), conf, "retrieval", p["page"], evidence=p["text"],
                      extra={"section": f"{p['section']} {p['section_title']}"})

    def pick_items(self, items: list[str], q_terms: set[str], intro: str) -> list[str]:
        """Keep the list items that match question words not already in the intro;
        when none match specifically, keep the whole list (max 8)."""
        extra = {t for t in q_terms if t not in set(stems(intro)) and t not in self.VERBS}
        scored = [(len(extra & set(stems(it))), it) for it in items]
        best = max((sc for sc, _ in scored), default=0)
        if best > 0 and len(items) > 2:
            return [it for sc, it in scored if sc == best][:3]
        return items[:8]

    def coverage(self, q_terms: set[str], dterms) -> float:
        """IDF-weighted share of question terms found in the passage. A term also counts as
        found when one of its thesaurus synonyms is present (exam -> examination)."""
        tot = got = 0.0
        terms = {t for t in q_terms if t not in self.VERBS} or set(q_terms)
        for t in terms:
            syns = self._syn_of.get(t, set())
            in_corpus = [f for f in {t} | syns if self.bm25.df.get(f, 0)]
            w = max((self.bm25.idf(f) for f in in_corpus), default=self.bm25.idf(t))
            if t.isdigit():
                w = min(w, 1.0)             # a bare number ("semester 1") is weak evidence on its own
            tot += w
            if t in dterms:
                got += w                    # exact term: full credit
            elif len(t) >= 5 and any(d.startswith(t) or (len(d) >= 5 and t.startswith(d)) for d in dterms):
                got += 0.8 * w              # same word family: lanka / lankan
            elif syns & set(dterms):
                got += 0.75 * w             # thesaurus synonym: partial credit
        literal = any((t in dterms or (len(t) >= 5 and any(d.startswith(t) for d in dterms)))
                      for t in terms if not t.isdigit())
        if not literal:
            return 0.0  # evidence must contain at least one of the asked concepts itself, not only synonyms
        return got / tot if tot else 0.0

    VERBS = {"happen", "get", "got", "sit", "need", "mean", "take", "want", "go", "come", "do", "make", "allow",
             "know", "find", "stay", "remain", "becom", "keep", "still", "giv", "given", "say", "said", "tell",
             "use", "used", "able"}

    ENTITY_TERMS = {"faculty", "foc", "department", "kdu", "university", "campus"}

    GENERIC = {"happen", "get", "got", "sit", "need", "mean", "take", "want", "go", "come", "do", "make", "allow",
               "know", "find", "requir", "student", "faculty", "university", "kdu", "stay", "remain", "becom",
               "keep", "still", "giv", "given", "say", "said", "tell", "use", "used", "able"}

    def is_oov(self, term: str) -> bool:
        """True when neither the term, its thesaurus synonyms, nor a word of the same family
        (5-letter prefix) occurs anywhere in the handbook."""
        if self.bm25.df.get(term, 0) or term in self.GENERIC or len(term) < 3 or not term.isalpha():
            return False
        if not hasattr(self, "_thes_by_stem"):
            self._thes_by_stem = defaultdict(list)
            for k, v in THESAURUS.items():
                self._thes_by_stem[stem(k)] += v
                self._thes_by_stem[k] += v
        for syn in self._thes_by_stem.get(term, []):
            if any(self.bm25.df.get(x, 0) for x in stems(syn)):
                return False
        if len(term) <= 5:
            return True  # short stem never seen in the handbook (gym, bus, centr, wifi)
        prefix = term[:6]
        if not hasattr(self, "_df_terms"):
            self._df_terms = list(self.bm25.df)
        return not any(t.startswith(prefix) for t in self._df_terms)

    def retrieve_scoped(self, ql: str, keep, k: int = 2) -> tuple[float, str] | None:
        """BM25F restricted to passages that satisfy keep(passage); returns (score, text)."""
        qw = self.expand_query(ql)
        cand = [(self.bm25.score(qw, i), i) for i, p in enumerate(self.passages) if keep(p)]
        cand = [c for c in cand if c[0] > 0]
        if not cand:
            return None
        cand.sort(reverse=True)
        s, i = cand[0]
        return s, self.shorten(self.passages[i]["text"], 420)

    @staticmethod
    def shorten(text: str, limit: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= limit:
            return text
        cut = text[:limit]
        dot = cut.rfind(". ")
        return (cut[: dot + 1] if dot > limit * 0.5 else cut.rsplit(" ", 1)[0] + " ...")


# ======================================================================
# 5. PERSISTENCE HELPERS
# ======================================================================

def build_kb(pdf_path: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    kb = HandbookExtractor(pdf_path).run()
    if out_path:
        Path(out_path).write_text(json.dumps(kb, ensure_ascii=False, indent=1), encoding="utf-8")
    return kb


def load_kb(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    kb = json.loads(p.read_text(encoding="utf-8"))
    return kb if kb.get("version") == KB_VERSION else None


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "Student-Handbook-2022.pdf"
    kb = build_kb(pdf, "handbook_kb.json")
    qa = HandbookQA(kb)
    print(f"Indexed {kb['page_count']} pages, {len(kb['sections'])} sections, {len(kb['courses'])} courses, "
          f"{len(kb['staff'])} staff, {len(kb['passages'])} passages")
    while True:
        try:
            q = input("\nQ> ")
        except EOFError:
            break
        if not q.strip():
            break
        print("A>", qa.answer(q)["answer"])
