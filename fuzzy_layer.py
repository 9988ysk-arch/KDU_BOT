"""
fuzzy_layer.py - fuzzy rule base for the handbook assistant (v11, no LLM, CPU only).

The handbook's regulations are crisp (FGPA >= 3.70 is First Class, 45 marks is C), so they stay crisp in the
Prolog layer. Fuzzy logic is used where the *input* or the *decision* is imprecise:

1. FuzzyMatcher      typo-tolerant matching. Every unknown word gets a membership degree mu(w, t) in each
                     handbook term t (string similarity). A word is corrected only when mu >= a length-dependent
                     threshold, the best term clearly beats the runner-up, and the word is not a common English
                     word. "hed" -> head, "departmnt" -> department, "suplementary" -> supplementary.

2. Linguistic terms  "close to first class", "is 3.1 good?", "how far from second upper?", "almost a pass".
                     The crisp verdict always comes from the rules (class band / grade table); fuzzy sets
                     (VERY_CLOSE, CLOSE, MODERATE, FAR on the distance to the next boundary) describe the gap.

3. Decision rules    Mamdani-style rule base that decides ANSWER / VERIFY / FAIL for a retrieved sentence from
                     three inputs: retrieval confidence, question-evidence coverage and entity agreement.
                       R1 IF confidence is HIGH AND coverage is HIGH                     THEN answer
                       R2 IF confidence is HIGH AND entity is MATCH                      THEN answer
                       R3 IF confidence is MEDIUM AND coverage is MEDIUM                 THEN verify
                       R4 IF confidence is MEDIUM AND entity is MISMATCH                 THEN verify
                       R5 IF confidence is LOW AND coverage is LOW                       THEN fail
                       R6 IF entity is MISMATCH AND coverage is LOW                      THEN fail
                       R7 IF topic is MISMATCH AND coverage is not HIGH                  THEN fail
                       R8 IF answer-type is MISMATCH ("how many" without a number)       THEN verify (0.6)
                       R9 IF topic is MISMATCH AND coverage is HIGH                      THEN verify
                     (topic = the evidence comes from the handbook chapter the question is about)
                     Rule strength = min of the antecedents (fuzzy AND), aggregation = max, defuzzification =
                     weighted average of the output singletons (answer 1.0, verify 0.5, fail 0.0).
"""
from __future__ import annotations

import difflib
import re
from typing import Any

# ------------------------------------------------------------------------------------------------ membership
def trap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership: 0 below a, rises to 1 at b, 1 until c, falls to 0 at d."""
    if x <= a or x >= d:
        return 1.0 if (b <= x <= c) else 0.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    if x <= c:
        return 1.0
    return (d - x) / (d - c) if d > c else 1.0


def tri(x: float, a: float, b: float, c: float) -> float:
    return trap(x, a, b, b, c)


# Common English words that must never be "corrected" into handbook terms (the handbook vocabulary itself is
# also protected). Kept short on purpose: only words that appear in student questions.
COMMON = set("""
a about above after again against all almost alone along already also although always am among an and another
any anyone anything are around as ask at away back bad badly barely be because been before being below best
better between big both but by can cannot close closer come could did do does doing done down during each
early easy either else enough even ever every exactly fail far few find fine first for found from full get
gets give given go going good got great had half hard has have having he hello help her here high him his
how however i if in into is it its just keep kind know last late least less let like little long look lot
low make many may maybe me mean might mine more most much must my near nearly need never new next nice no
none nor not nothing now of off often ok okay old on once one only or other our out over own please pretty
quite rather really right said same say see seem should since so some something soon still such sure take
tell than thank thanks that the their them then there these they thing think this those though through to
too toward true try under until up upon us use very want was way we well were what when where whether which
while who whom whose why will wish with within without would yes yet you your quick quickly enough okay
gap sem fee fees due set reset weather bored visa password register change switch laptop phone number address
office room building campus bring wear food bus train late early form forms apply applied apply where card
id loan job jobs intern salary friend friends parent parents home house rent money cost price free paid
hour hours minute minutes week weeks month months today tomorrow yesterday night morning evening class men red bed led fed age ago add aid aim art bit box boy car cut dog ear eat end eye fit fix fly fun gas guy hat hit hot ice job key kid law lay leg lie lip map mix mom net nor oil pay pen pet pin pot put ran rat raw row run sad sat sea sit six sky son sun tax ten tie tip top toy two war wet win won yet zoo
fine poor average excellent decent worried worry scored score marks mark result results level levels
""".split())

SHORT_TARGETS = {"head", "dean", "fgpa", "sgpa", "gpa", "exam", "exams", "staff", "email", "mark", "marks",
                 "grade", "pass", "fail", "level"}
EXTRA_TARGETS = {"supplementary", "attendance", "requirement", "requirements", "department", "departments",
                 "examination", "examinations", "semester", "semesters", "elective", "electives", "compulsory",
                 "credits", "credit", "research", "lecturer", "lecturers", "programme", "degree", "module", "modules",
                 "prerequisite", "repeat", "resit", "weak", "class", "upper", "lower", "division", "graduate",
                 "graduation", "military", "civil", "entry", "admission", "duration", "software", "engineering",
                 "information", "technology", "systems", "computer", "science", "computational", "mathematics"}


def edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein distance (insert, delete, substitute, swap adjacent letters)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


class FuzzyMatcher:
    def __init__(self, qa):
        self.qa = qa
        targets: set[str] = set(getattr(qa, "correction_targets", [])) | EXTRA_TARGETS | SHORT_TARGETS
        for c in qa.courses.values():
            targets |= {w for w in re.findall(r"[a-z]+", c["title"].lower()) if len(w) > 3}
        self.surnames = set()
        for st in qa.kb.get("staff", []):
            for w in re.findall(r"[A-Za-z]+", re.sub(r"\(.*?\)", "", st["name"])):
                if len(w) > 4 and not w.isupper():
                    self.surnames.add(w.lower())
        targets |= self.surnames
        self.targets = sorted(t for t in targets if t.isalpha())
        self.protected = set(getattr(qa, "vocab", set())) | COMMON
        self._tset = set(self.targets)

    @staticmethod
    def mu(a: str, b: str) -> float:
        """Membership of word a in term b: similarity ratio, with an adjacent-swap bonus (cerdit/credit)."""
        r = difflib.SequenceMatcher(None, a, b).ratio()
        if len(a) == len(b) and sorted(a) == sorted(b):
            r = max(r, 0.9)
        return r

    def threshold(self, n: int) -> float:
        return 0.85 if n <= 3 else 0.8 if n <= 5 else 0.78

    def best(self, w: str) -> tuple[str, float, float] | None:
        pool = sorted(SHORT_TARGETS) if len(w) <= 3 else self.targets
        cands = difflib.get_close_matches(w, pool, n=4, cutoff=0.7)
        scored = sorted(((self.mu(w, c), c) for c in cands if c[0] == w[0] and abs(len(c) - len(w)) <= 2),
                        reverse=True)
        if not scored:
            return None
        second = scored[1][0] if len(scored) > 1 else 0.0
        return scored[0][1], scored[0][0], second

    def correct(self, text: str) -> tuple[str, list[tuple[str, str, float]]]:
        """Returns (corrected text, [(word, term, membership)])."""
        fixes: list[tuple[str, str, float]] = []

        def fix(m: re.Match) -> str:
            w = m.group(0)
            lw = w.lower()
            if lw in self.protected or len(lw) < 3 or not lw.isalpha() or w.isupper():
                return w
            # inflections of known words are not typos ("headed", "offering", "teachers", "barely")
            for suf in ("s", "es", "ed", "d", "ing", "er", "ers", "ly", "ion", "ions", "al"):
                base = lw[:-len(suf)]
                if lw.endswith(suf) and len(base) >= 3 and (base in self.protected or base in self._tset or
                                                            base + "e" in self.protected or base + "e" in self._tset):
                    return w
            b = self.best(lw)
            if not b:
                return w
            term, mu1, mu2 = b
            max_edits = 1 if len(lw) <= 7 else 2  # a typo is a small edit, not a different word
            if mu1 >= self.threshold(len(lw)) and mu1 - mu2 >= 0.04 and term != lw and \
                    edit_distance(lw, term) <= max_edits:
                fixes.append((w, term, round(mu1, 2)))
                return term.capitalize() if w[0].isupper() else term
            return w
        out = re.sub(r"[A-Za-z]+", fix, text)
        return out, fixes


# ------------------------------------------------------------------------------------------------ linguistic
VAGUE = re.compile(r"\b(close|closer|near|nearly|almost|far|how far|good|bad|ok|okay|enough|decent|high|low|"
                   r"barely|just|borderline|safe|safely|comfortably|risky|short|too low|too high|well)\b")

CLASS_BOUNDS = [(3.70, "First Class"), (3.30, "Second Class (Upper Division)"),
                (3.00, "Second Class (Lower Division)"), (2.00, "Pass")]


def gap_label(d: float) -> tuple[str, float]:
    """Fuzzy description of a positive gap d (in FGPA points) to a boundary."""
    sets = {"very close": trap(d, -1, 0, 0.05, 0.1), "close": tri(d, 0.05, 0.12, 0.25),
            "moderately far": tri(d, 0.15, 0.35, 0.6), "far": trap(d, 0.4, 0.7, 5, 6)}
    name = max(sets, key=sets.get)
    return name, round(sets[name], 2)


def standing_label(x: float) -> tuple[str, float]:
    """Assistant's own fuzzy reading of an FGPA (not a handbook term)."""
    sets = {"weak": trap(x, -1, 0, 1.8, 2.2), "fair": trap(x, 1.8, 2.2, 2.8, 3.1),
            "good": trap(x, 2.8, 3.1, 3.4, 3.6), "very good": trap(x, 3.4, 3.6, 3.75, 3.9),
            "excellent": trap(x, 3.7, 3.9, 4.2, 5)}
    name = max(sets, key=sets.get)
    return name, round(sets[name], 2)


class FuzzyLinguistic:
    def __init__(self, qa, logic=None):
        self.qa, self.logic = qa, logic
        self.grade_rows = [r for r in qa.kb.get("grades", {}).get("rows", []) if r.get("low") is not None]
        self.class_page = qa.kb.get("classes", {}).get("page")

    def crisp_class(self, x: float) -> str:
        if self.logic is not None:  # crisp verdict from the Prolog facts class_band(Class, Low, High)
            try:
                for r in self.logic.pl.query("class_band(C, Lo, Hi)"):
                    hi = r["Hi"] if isinstance(r["Hi"], (int, float)) else 99
                    if r["Lo"] - 1e-9 <= x < hi:
                        return str(r["C"])
            except Exception:
                pass
        for b, name in CLASS_BOUNDS:
            if x >= b - 1e-9:
                return name
        return "below the 2.00 needed to graduate"

    def target_class(self, ql: str) -> str | None:
        if re.search(r"\bfirst\b", ql):
            return "First Class"
        if re.search(r"\bupper\b", ql):
            return "Second Class (Upper Division)"
        if re.search(r"\blower\b|\bsecond class\b", ql):
            return "Second Class (Lower Division)"
        if re.search(r"\b(pass|graduate)\b", ql):
            return "Pass"
        return None

    def answer(self, q: str, fgpa_ctx: float | None = None) -> dict | None:
        ql = q.lower()
        if not VAGUE.search(ql):
            return None
        page = f" (class bands, Student Handbook p. {self.class_page})" if self.class_page else ""
        # ---- FGPA
        m = re.search(r"\b([0-4]\.\d{1,2})\b", ql)
        if m and re.search(r"\b(f?gpa|class|first|upper|lower|second|honou?rs?|have|got|is)\b", ql) and \
                not re.search(r"\bmarks?\b", ql):
            x = float(m.group(1))
            if x > 4.2:
                return None
            cls = self.crisp_class(x)
            tgt = self.target_class(ql)
            parts = [f"An FGPA of {x:.2f} is {cls}{page}."]
            if tgt:
                bound = next(b for b, n in CLASS_BOUNDS if n == tgt)
                if re.search(r"\b(enough|ok|okay|good enough|sufficient)\b", ql):  # yes/no question: crisp first
                    parts[0] = ("Yes. " if x >= bound else "No. ") + parts[0]
                if x >= bound:
                    parts.append(f"You are already in the {tgt} band (minimum {bound:.2f}).")
                else:
                    d = round(bound - x, 2)
                    lab, mu = gap_label(d)
                    parts.append(f"{tgt} needs {bound:.2f}, so you are {d:.2f} short - {lab} (fuzzy degree {mu}).")
            else:
                nxt = next(((b, n) for b, n in reversed(CLASS_BOUNDS) if b > x), None)
                lab, mu = standing_label(x)
                parts.append(f"The handbook doesn't define \"good\"; on my fuzzy scale {x:.2f} reads as {lab} "
                             f"(degree {mu}).")
                if nxt:
                    parts.append(f"The next class, {nxt[1]}, starts at {nxt[0]:.2f} ({nxt[0] - x:.2f} higher).")
            if tgt in ("First Class", "Second Class (Upper Division)") or "class" in ql:
                parts.append("The class also depends on failures and weak passes, not only on the FGPA.")
            return {"answer": " ".join(parts), "kind": "fuzzy_class", "fgpa": x}
        # ---- marks
        m = re.search(r"\b(\d{1,3})\s*(?:marks?|%|percent)\b", ql)
        if m and 0 <= int(m.group(1)) <= 100:
            v = int(m.group(1))
            row = next((r for r in self.grade_rows if r["low"] <= v <= r["high"]), None)
            g = row["grade"] if row else ("a failure grade (Ie/Ia/Ib)" if v < 35 else "?")
            if row and g in ("C-", "D+"):
                d = 45 - v
                status = (f"{v} marks is {g} (GPV {row['gpv']}), a weak pass: {d} mark{'s' if d != 1 else ''} short of "
                          f"a full pass (C, 45). At most one D+/C- is allowed per semester.")
            elif v >= 45:
                margin = v - 45
                lab = "just above" if margin <= 2 else "comfortably above" if margin >= 10 else "above"
                status = f"{v} marks is {g} (GPV {row['gpv'] if row else ''}) - a pass, {lab} the 45 mark for C."
            else:
                status = (f"{v} marks is below 35, so it is a failure grade (Ie/Ia/Ib depending on which component "
                          f"was under 35%); the course must be repeated.")
            return {"answer": status + " (Every component also needs at least 35%.)", "kind": "fuzzy_marks"}
        return None


# ------------------------------------------------------------------------------------------------ decision
class FuzzyDecision:
    """Decides what to do with a retrieved sentence. Returns (action, score, rule strengths)."""

    @staticmethod
    def coverage(question: str, evidence: str) -> float:
        from handbook_qa import stems
        qs = {s for s in stems(question.lower()) if len(s) > 2}
        if not qs:
            return 0.5
        es = set(stems(evidence.lower()))
        return len(qs & es) / len(qs)

    def decide(self, confidence: float, coverage: float, entity: float | None, topic: float = 0.6,
               atype: float = 0.7) -> tuple[str, float, dict[str, float]]:
        c_hi, c_med, c_lo = trap(confidence, 0.55, 0.75, 1, 1.1), tri(confidence, 0.3, 0.55, 0.8), \
            trap(confidence, -0.1, 0, 0.3, 0.5)
        v_hi, v_med, v_lo = trap(coverage, 0.45, 0.7, 1, 1.1), tri(coverage, 0.2, 0.45, 0.7), \
            trap(coverage, -0.1, 0, 0.15, 0.35)
        # entity None = the question names no module/person: neutral, neither match nor mismatch
        e_match, e_mis = (0.6, 0.0) if entity is None else (entity, 1 - entity)
        t_match, t_mis = trap(topic, 0.4, 0.8, 1, 1.1), trap(topic, -0.1, 0, 0.2, 0.5)
        a_match, a_mis = trap(atype, 0.5, 0.8, 1, 1.1), trap(atype, -0.1, 0, 0.3, 0.5)
        rules = {
            "R1 conf HIGH & coverage HIGH & topic MATCH -> answer": min(c_hi, v_hi, max(t_match, 0.6)),
            "R2 conf HIGH & entity MATCH & topic MATCH -> answer": min(c_hi, e_match, max(t_match, 0.6)),
            "R3 conf MEDIUM & coverage MEDIUM -> verify": min(c_med, v_med),
            "R4 conf MEDIUM & entity MISMATCH -> verify": min(c_med, e_mis),
            "R5 conf LOW & coverage LOW -> fail": min(c_lo, v_lo),
            "R6 entity MISMATCH & coverage LOW -> fail": min(e_mis, v_lo),
            "R7 topic MISMATCH & coverage not HIGH -> fail": min(t_mis, 1 - v_hi),
            "R8 answer-type MISMATCH -> verify": 0.6 * a_mis,
            "R9 topic MISMATCH & coverage HIGH -> verify": min(t_mis, v_hi),
        }
        vals = list(rules.values())
        out = {"answer": max(vals[0], vals[1]), "verify": max(vals[2], vals[3], vals[7], vals[8]),
               "fail": max(vals[4], vals[5], vals[6])}
        tot = sum(out.values())
        score = (out["answer"] * 1.0 + out["verify"] * 0.5) / tot if tot else 0.5
        action = "answer" if score >= 0.7 else "verify" if score >= 0.3 else "fail"
        return action, round(score, 2), {k.split()[0]: round(v, 2) for k, v in rules.items()}


# ------------------------------------------------------------------------------------------------ v11 inputs
# Which handbook chapters a question about a topic should be answered from (page ranges come from the KB).
TOPIC_FAMILIES = [
    ("regulations", r"\b(get|got|gets|getting) an? [a-f][+-]?\b|\b(grades?|grading|graded|pass(ing)?|fail\w*|repeat\w*|re-?sit\w*|attendance|exams?|examinations?|marks?|"
                    r"gpa|sgpa|gpv|weak|supplementary|medical|absent|excuse\w*|discontinu\w*|duration|class|"
                    r"graduat\w*|award\w*|trophy|fgpa|first class|upper|lower)\b", {"2", "4", "5", "6"}),
    ("curriculum", r"\b(modules?|courses?|subjects?|credits?|semesters?|electives?|compulsory|syllabus|curriculum|"
                   r"level|year)\b", {"3"}),
    ("people", r"\b(lecturers?|staff|head|dean|e-?mail|research\w*|professor|dr|mr|mrs|ms|qualifications?)\b",
     {"1", "7", "8"}),
    ("admission", r"\b(entry|admission|a/?l|o/?l|stream|eligib\w*|apply|military|civil|height|age)\b", {"2", "1"}),
]


class Signals:
    """Extra fuzzy inputs for the decision rule base: topic agreement and answer-type agreement."""

    def __init__(self, qa):
        self.chapters = []
        for sec in qa.kb.get("sections", []):
            if sec.get("level", 9) <= 1 and sec["number"] != "0":
                self.chapters.append((sec["page"], sec["number"]))
        self.chapters.sort()

    def chapter_of(self, page: int | None) -> str | None:
        if not page:
            return None
        ch = None
        for p, num in self.chapters:
            if page >= p:
                ch = num
        return ch

    def topic_agreement(self, question: str, page: int | None) -> float:
        ql = question.lower()
        want = set()
        for _, rx, chs in TOPIC_FAMILIES:
            if re.search(rx, ql):
                want |= chs
        ch = self.chapter_of(page)
        if not want or ch is None:
            return 0.6  # nothing to compare: neutral
        return 1.0 if ch in want else 0.0

    @staticmethod
    def type_agreement(question: str, answer: str) -> float:
        ql = question.lower()
        if re.search(r"\bhow (many|much|long)\b|\bwhat (percentage|percent)\b", ql):
            return 1.0 if re.search(r"\d|\b(one|two|three|four|five|six|seven|eight|first|second)\b", answer.lower()) else 0.2
        if re.search(r"^\s*who\b", ql):
            return 1.0 if re.search(r"\b(Dr|Mr|Mrs|Ms|Miss|Prof|Eng)\.|\b(head|dean|registrar|board|senate|faculty)\b",
                                    answer, re.I) else 0.3
        if re.search(r"^\s*when\b", ql):
            return 1.0 if re.search(r"\b(semester|year|end|after|before|during|week|month|\d)\b", answer.lower()) else 0.3
        return 0.7
