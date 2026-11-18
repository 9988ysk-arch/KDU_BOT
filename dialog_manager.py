"""
dialog_manager.py - conversation layer on top of the non-LLM handbook QA engine.

Pipeline for every user message (rule-based, no LLM):

  1. Reset            "start over", "new chat"                -> clear the session
  2. Pending state    the bot asked a clarifying / verification question last turn
                      -> map the reply (option label, ordinal, CS/SE, yes/no) to a resolved question
  3. Small talk       greetings, thanks, who-are-you, help, bye (English + Singlish);
                      "hi, who is the dean?" -> greeting is stripped and the question is answered
  4. Feedback         "that's wrong" -> apology + alternative evidence + topic options
  5. User context     "I'm a software engineering student", "my FGPA is 3.4" -> stored in the session
  6. Context          pronouns (it / he / his / that department / the first one / there),
                      ellipsis ("what about the credits?"), remembered programme of the student
  7. Scope check      doubtful scope -> verification question with options instead of a guess:
                        - curriculum details asked for IT / IS / CE (handbook only has CS + SE tables)
                        - a module title shared by CS and SE (Industrial Training, ...)
                        - a surname shared by several staff members
  8. Answer           HandbookQA.answer()
  9. Verification     low-confidence retrieval answers ask "Did this answer your question?"
 10. State update     entities from the question and the answer (module, person, programme,
                      department, semester, FGPA, list of people) with recency for pronouns
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import small_talk
from conversation_rules import ConversationRules
from fuzzy_layer import FuzzyDecision, FuzzyLinguistic, FuzzyMatcher, Signals
try:
    from logic_layer import LogicLayer
except Exception:  # the Prolog layer is optional: without it the assistant still works
    LogicLayer = None
from handbook_qa import DEPARTMENTS, PROGRAMS, HandbookQA, fmt_num as fmt, normalize, programs_in, stem, tokens

PROG_NAMES = {"CS": "Computer Science", "SE": "Software Engineering", "CE": "Computer Engineering",
              "IT": "Information Technology", "IS": "Information Systems"}
CURRICULUM_PROGS = ("CS", "SE")


def PNAME(code: str) -> str:
    """'IT' -> 'the information technology degree' (programme words the parser always recognises)"""
    return f"the {PROG_NAMES[code].lower()} degree" if code in PROG_NAMES else code  # programmes whose module tables are in the handbook
ORDINALS = {"first": 0, "1st": 0, "one": 0, "1": 0, "second": 1, "2nd": 1, "two": 1, "2": 1,
            "third": 2, "3rd": 2, "three": 2, "3": 2, "fourth": 3, "4th": 3, "4": 3, "last": -1}

EXAMPLES = ["How many credits is CS2082?", "What are the electives in semester 6 for SE?",
            "What grade do I get for 72 marks?", "Who is the head of the IT department?",
            "What is the attendance requirement?"]


@dataclass
class Session:
    entities: dict[str, Any] = field(default_factory=dict)  # name -> value
    seen: dict[str, int] = field(default_factory=dict)      # name -> turn number (recency)
    user_programme: str | None = None                        # "I'm a SE student"
    last_question: str | None = None                         # resolved question of the last QA turn
    last_answer: str = ""
    last_people: list[str] = field(default_factory=list)     # people listed in the last answer
    pending: dict[str, Any] | None = None                    # clarification / verification state
    last_topics: list[str] = field(default_factory=list)     # topic suggestions of the last "not found"
    last_reply: dict[str, Any] | None = None                 # full last answer (for "repeat that")
    last_kind: str = ""                                      # answer type of the last QA turn
    last_page: int | None = None                             # handbook page of the last answer
    last_evidence: str = ""                                  # sentence(s) the last answer came from
    last_list: tuple[str, list[str]] | None = None           # ("module"|"person"|"programme", items)
    turn: int = 0
    updated: float = field(default_factory=time.time)
    # v10 context
    people_hist: list[str] = field(default_factory=list)     # people mentioned, most recent last (for he/she)
    topic: str | None = None                                 # "class" | "grade" | "repeat" (for ellipsis)
    last_failed: str | None = None                           # question of the last turn that found nothing
    user_name: str | None = None                             # "my name is Kasun"

    def remember(self, name: str, value: Any) -> None:
        if value is None:
            return
        self.entities[name] = value
        self.seen[name] = self.turn
        if name == "person":
            if value in self.people_hist:
                self.people_hist.remove(value)
            self.people_hist.append(value)
            del self.people_hist[:-6]


class DialogManager:
    SESSION_TTL = 60 * 60 * 6  # forget idle sessions after 6 hours

    def __init__(self, qa: HandbookQA):
        self.qa = qa
        self.sessions: dict[str, Session] = {}
        # shared module titles (CS + SE copies) and shared staff surnames, for clarification
        by_title: dict[str, list[str]] = {}
        for code, c in qa.courses.items():
            by_title.setdefault(c["title"].lower(), []).append(code)
        self.shared_titles = {t: codes for t, codes in by_title.items() if len(codes) > 1}
        by_surname: dict[str, list[dict[str, Any]]] = {}
        for st in qa.kb["staff"]:
            parts = [p for p in re.sub(r"\(.*?\)", "", st["name"]).replace(".", " ").split() if len(p) > 2]
            if parts:
                by_surname.setdefault(parts[-1].lower(), []).append(st)
        self.shared_surnames = {k: v for k, v in by_surname.items() if len(v) > 1}
        self.rules = ConversationRules(qa)  # v10: handbook-grounded rule answers + "not covered" catalogue
        # v11 fuzzy layer: typo-tolerant matching + fuzzy decision rule base (linguistic part needs Prolog below)
        self.fuzzy = FuzzyMatcher(qa)
        self.fuzzy_dec = FuzzyDecision()
        self.signals = Signals(qa)
        # Prolog reasoning layer (facts generated from this handbook + handbook_rules.pl)
        self.logic = None
        if LogicLayer is not None:
            try:
                self.logic = LogicLayer(qa)
            except Exception:
                self.logic = None
        self.fuzzy_ling = FuzzyLinguistic(qa, self.logic)
        # v11 ontology generated from this handbook + Prolog rules over it (multi-hop relational questions)
        self.onto = None
        try:
            from ontology_query import OntologyQuery
            self.onto = OntologyQuery(qa)
        except Exception as exc:  # optional: without it the assistant still works
            print(f"[ontology] disabled: {exc}")

    # ------------------------------------------------------------------ public API
    def handle(self, session_id: str, message: str) -> dict[str, Any]:
        self._expire()
        s = self.sessions.setdefault(session_id or "default", Session())
        s.turn += 1
        s.updated = time.time()
        try:
            return self._handle(s, (message or "").strip())
        except Exception as exc:  # graceful failure: never show a stack trace to a student
            print(f"[dialog] {type(exc).__name__}: {exc}")
            return self._reply("Sorry, something went wrong while I was looking that up. Could you ask it "
                               "again in a slightly different way?", "error", options=EXAMPLES[:3])

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    # ------------------------------------------------------------------ main pipeline
    def _handle(self, s: Session, text: str) -> dict[str, Any]:
        self._prefix = ""
        low = normalize(text).lower().strip()

        # 0a. Sinhala / Tamil script: the handbook index is English
        if re.search(r"[\u0D80-\u0DFF\u0B80-\u0BFF]", text) and not re.search(r"[A-Za-z]{3,}", text):
            return self._reply("Sorry, I can only read questions typed in English or Singlish (Sinhala in English "
                               "letters, e.g. \"dean kawda?\"). The handbook itself is in English.",
                               "unsupported_script", options=EXAMPLES[:3])
        # 0. empty / punctuation only (emoji-only messages are small talk)
        if not re.search(r"[a-z0-9]", low) and not small_talk.classify(text)[0]:
            return self._reply("Please type a question about the Student Handbook, for example: "
                               f"\"{EXAMPLES[0]}\"", "empty", options=EXAMPLES[:3])
        # 1. reset
        if re.fullmatch(r"(please )?(start over|reset|new chat|new conversation|clear( the)? (chat|context)|"
                        r"forget (it|everything)|restart)[.!]?", low):
            self.sessions[self._sid_of(s)] = Session(turn=s.turn)
            return self._reply("Okay, starting fresh. I've cleared what I remembered from this chat. "
                               "What would you like to know?", "reset", options=EXAMPLES[:3])
        # 2. pending clarification / verification
        if s.pending:
            res = self._resolve_pending(s, text, low)
            if res is not None:
                return res
        # 2b. meta questions about the last answer (repeat / source / why / tell me more)
        meta = self._meta(s, low)
        if meta:
            return meta
        # 3. small talk (may be followed by a real question)
        intent, rest, prefix = small_talk.classify(text)
        if intent == "confused" and s.last_reply and s.last_question and \
                not s.last_reply.get("kind", "").startswith(("small_talk", "not_")):
            return self._simpler(s)
        if intent:
            reply, examples = small_talk.reply_for(intent, small_talk.normalise(text))
            if s.user_name and intent in ("greeting", "greeting_time", "how_are_you", "bye", "good_night", "thanks"):
                reply = re.sub(r"^(Hello|Good \w+|Goodbye|You're welcome|I'm doing well)([!,])",
                               rf"\1, {s.user_name}\2", reply, count=1)
            opts = None
            if intent == "emotion_negative":
                opts = ["How do I re-sit a course?", "When are supplementary exams held?",
                        "What is the attendance rule on medical grounds?"]
            elif examples:
                opts = EXAMPLES[:4]
            return self._reply(reply, "small_talk_" + intent, opts)
        if rest:
            text, low = rest, normalize(rest).lower()
        self._prefix = prefix or ""
        if self._prefix and s.user_name and self._prefix in ("Hi!", "Good morning!", "Good afternoon!", "Good evening!"):
            self._prefix = self._prefix[:-1] + f", {s.user_name}!"
        # 4. feedback on the previous answer
        fb = self._feedback(s, low)
        if fb:
            return fb
        # 4b. several questions in one message -> answer each in turn
        parts = self._split_questions(text)
        if len(parts) > 1:
            replies = [self._one_question(s, p) for p in parts]
            out = self._reply("\n\n".join(r["answer"] for r in replies), "multi",
                              options=next((r.get("options") for r in replies if r.get("options")), None))
            s.last_reply = out
            return out
        out = self._one_question(s, text)
        if getattr(self, "_prefix", "") and not out["answer"].startswith(self._prefix):
            out = dict(out, answer=f"{self._prefix} {out['answer']}")
        self._prefix = ""
        return out

    def _one_question(self, s: Session, text: str) -> dict[str, Any]:
        low = normalize(text).lower().strip()
        # 5. statements about the user
        ctx_reply = self._user_context(s, text, low)
        if ctx_reply:
            return ctx_reply
        # 5'. v11 fuzzy matching: correct typos towards handbook terms ("hed of the IT departmnt")
        self._fixes = []
        fixed, fixes = self.fuzzy.correct(text)
        if fixes:
            self._fixes = fixes
            text, low = fixed, normalize(fixed).lower().strip()
        # 5a. v10 topic follow-ups: "and second upper?" after a class answer, "what about C-?" / "can I repeat
        #     it?" after a grade answer, "how many times?" after a repeat answer
        follow = self._topic_followup(s, text, low)
        if follow:
            text, low = follow, normalize(follow).lower().strip()
        # 5a'. a he/she pronoun right after a turn that found nobody: don't silently reuse an older person
        stale = self._stale_pronoun(s, text, low) or self._ambiguous_pronoun(s, text, low)
        if stale:
            return stale
        # 5b. plural references ("their emails", "how long are they?") -> one answer per item
        plural = self._plural(s, text, low)
        if plural:
            return plural
        # 5c. vague one-word questions ("credits", "grades") -> ask what exactly
        vague = self._vague(s, low)
        if vague:
            return vague
        # 6. context resolution
        resolved = self._possessive(self._resolve_context(s, text, low), s.entities.get("person"))
        # 6b. impossible values (150 marks, FGPA 5.2, semester 10) and unknown staff names
        bad = self._validate(resolved)
        if bad:
            return bad
        # 6a0. v10: a department that does not exist ("head of the music department")
        unk = self._unknown_department(s, resolved)
        if unk:
            return unk
        # 6a. v10 graceful failing: topics the handbook never covers (hostel, library, timetable ...)
        nc = self.rules.not_covered(resolved)
        if nc:
            s.last_failed = None
            return self._reply(nc["answer"], nc["kind"], options=nc.get("options"))
        # 6a-ii. v11 vague wording with a number ("close to first class", "is 3.1 good?", "almost a pass"):
        #        crisp class/grade from the rules, fuzzy sets describe the gap
        if not re.search(r"\b(fail\w*|weak|repeat\w*|re-?sit\w*|d\+|c-|failures?)\b", resolved.lower()):
            fz = self.fuzzy_ling.answer(resolved)
            if fz:
                if fz.get("fgpa") is not None:
                    s.remember("fgpa", fz["fgpa"])
                return self._finish(s, resolved, text, fz["answer"], fz["kind"],
                                    topic="class" if fz["kind"] == "fuzzy_class" else "grade")
        # 6b'. reasoning over several handbook rules (GPA calculator, semester/class/graduation checks,
        #      eligibility, repeat rules): the Prolog layer answers these exactly
        if self.logic is not None:
            lr = self.logic.answer(resolved)
            if lr:
                self._update_state(s, resolved, lr["answer"], lr["kind"])
                s.last_kind, s.last_page, s.last_evidence = lr["kind"], lr.get("page"), ""
                out = self._reply(lr["answer"], lr["kind"], confidence=lr.get("confidence"))
                out["reasoning"] = "prolog"
                s.last_reply = out
                return out
        # 6b''. v10 rule answers grounded in the handbook text (pass mark, grade letters, weak passes, repeat
        #       limits, Pass class, prerequisites, difficulty, head of a degree without its own department)
        rr = self._rule_answer(s, resolved, text)
        if rr:
            return rr
        # 6b-iii. v11 relational / multi-hop questions: Prolog over the automatically generated ontology
        if self.onto is not None:
            orr = self.onto.answer(resolved)
            if orr:
                return self._finish(s, resolved, text, orr["answer"], orr["kind"], confidence=orr.get("confidence"),
                                    reasoning="ontology+prolog")
        # 6c. under-specified questions: ask for the missing slot (degree, year/semester, department, ...)
        slot = self._slots(s, resolved)
        if slot:
            return slot
        # 7. scope verification
        clar = self._scope_check(s, resolved)
        if clar:
            return clar
        # 8. answer
        return self._answer(s, resolved, original=text)

    # ------------------------------------------------------------------ v10: topics, rules, stale pronouns
    CLASS_FOLLOW = re.compile(
        r"^(?:(?:and|or|so|then)\s+)?(?:(?:what|how)\s+about\s+|what\s+of\s+|for\s+)?(?:the\s+|a\s+)?(?:requirements? (?:for|of) "
        r"(?:the |a )?)?(first(?: class)?|1st(?: class)?|(?:second|2nd)(?: class)?\s*\(?(?:upper|lower)(?: division)?\)?|"
        r"(?:upper|lower)(?: division| second)?(?: class)?|pass(?: class)?|general(?: degree)?|ordinary(?: pass)?)"
        r"(?:\s+class)?\s*[?.!]*$")
    GRADE_FOLLOW = re.compile(r"^(?:(?:and|or)\s+)?(?:(?:what|how)\s+about\s+)?(?:if i get\s+)?(?:an?\s+)?(?:grade\s+)?"
                              r"[\"'‘“]?(a\+|a-|b\+|b-|c\+|c-|d\+|ie|ia|ib|ne|ab|ex|a|b|c|d|e|f)[\"'’”]?\s*[?.!]*$", re.I)

    def _topic_followup(self, s: Session, text: str, low: str) -> str | None:
        l = low.strip()
        if s.topic == "class":
            m = self.CLASS_FOLLOW.match(l)
            if m:
                w = m.group(1)
                if "upper" in w:
                    return "what are the requirements for second class upper?"
                if "lower" in w:
                    return "what are the requirements for second class lower?"
                if w.startswith(("first", "1st")):
                    return "what are the requirements for first class?"
                return "what are the requirements for a pass class?"
        if s.topic in ("grade", "repeat") or (s.topic == "class" and s.entities.get("grade")):
            m = self.GRADE_FOLLOW.match(text.strip())
            if m and (len(l) > 1 or s.topic == "grade"):
                g = m.group(1)
                if not (g == "a" and not re.search(r"\b(an|grade)\b", l)):
                    return f"what if I get {self.rules.label(g.lower())}?"
            g = s.entities.get("grade")
            if g and re.search(r"\b(it|that|this|that grade|this grade)\b", l) and \
                    re.search(r"\b(repeat|re-?sit|redo|retake|upgrade|improve|pass|fail|mean|gpv|count|allowed|ok)\b", l) and \
                    not re.search(r"\b[a-z]{2}\s?\d{4}\b", l):
                label = self.rules.label(g)
                if re.search(r"\b(repeat|re-?sit|redo|retake|upgrade|improve)\b", l):
                    return f"can I repeat a module with {label}?"
                return f"what if I get {label}?"
        if s.topic in ("repeat", "grade") and re.fullmatch(
                r"(and |so )?(how many (times|attempts|chances)|is there a limit|any limit|how often)"
                r"( can i (do|repeat|re-?sit|take) (it|that|this))?( then)?\s*[?.!]*", l):
            return "how many times can I repeat a module?"
        return None

    def _stale_pronoun(self, s: Session, text: str, low: str) -> dict[str, Any] | None:
        if not s.last_failed or not s.entities.get("person"):
            return None
        if not re.search(self.PRONOUN_PERSON, low) or self.qa.match_person(low):
            return None
        if not re.search(r"\b(head|dean|lecturer|staff|teacher|coordinator|dr|mr|mrs|ms|prof|professor|who)\b",
                         s.last_failed.lower()):
            return None
        person = self._pick_person(s, low)
        failed = s.last_failed
        s.last_failed = None
        resolved = re.sub(self.PRONOUN_PERSON, person, text, count=1, flags=re.I)
        resolved = self._possessive(resolved, person)
        return self._clarify(s, f"I couldn't answer your last question (\"{failed.rstrip('?')}\"), so I'm not sure who "
                                f"you mean. Do you mean {person}?", "confirm",
                             [f"Yes, {person}", "No, someone else"], [resolved, None],
                             keys=[["yes", "yeah", "yep"], ["no", "someone else"]], default=0)

    KNOWN_DEPT_WORDS = re.compile(r"^(computer science|information technology|computer engineering|computational "
                                  r"mathematics|software engineering|information systems|computing|cs|it|ce|se|is|cm|"
                                  r"maths?|mathematics|comp(uter)? sci|that|this|the|same|which|each|every|what|your|my|"
                                  r"our|a|an|any|one|other|whole|other|respective|relevant|their|his|her|its|of)$")

    def _unknown_department(self, s: Session, q: str) -> dict[str, Any] | None:
        ql = normalize(q).lower()
        m = re.search(r"\b(?:the\s+)?([a-z]+(?:\s[a-z]+)?)\s+(?:department|dept)\b", ql) or \
            re.search(r"\b(?:department|dept)\.? of\s+(?:the\s+)?([a-z]+(?:\s[a-z]+)?)\b", ql)
        if not m:
            return None
        name = m.group(1).strip()
        words = name.split()
        if self.KNOWN_DEPT_WORDS.match(name) or (words and self.KNOWN_DEPT_WORDS.match(words[-1])) or \
                self.qa.match_department(ql) or programs_in(name):
            return None
        if words[0] in ("head", "staff", "lecturers", "email", "of", "in", "about", "which", "for", "who", "is", "are"):
            return None
        s.last_failed = q
        depts = ", ".join(DEPARTMENTS.values())
        return self._reply(f"I couldn't find a \"{name.title()}\" department in the Student Handbook. The Faculty of "
                           f"Computing has four departments: {depts}.", "not_found_department",
                           options=[f"Who is the head of the {d.lower()}?" for d in list(DEPARTMENTS.values())[:3]])

    @staticmethod
    def _gender(name: str) -> str:
        if re.match(r"(Mrs|Ms|Miss)\b", name):
            return "f"
        if re.match(r"Mr\b", name):
            return "m"
        return "?"

    def _ambiguous_pronoun(self, s: Session, text: str, low: str) -> dict[str, Any] | None:
        """'her' when the latest person is a Dr./Prof. (gender unknown) but an earlier one is a Mrs./Ms.: ask."""
        if len(s.people_hist) < 2 or self.qa.match_person(low):
            return None
        fem, masc = re.search(r"\b(she|her|hers)\b", low), re.search(r"\b(he|him|his)\b", low)
        if bool(fem) == bool(masc):
            return None
        latest = s.people_hist[-1]
        if self._gender(latest) != "?":
            return None
        want = "f" if fem else "m"
        earlier = next((n for n in reversed(s.people_hist[-3:-1]) if self._gender(n) == want), None)
        if not earlier:
            return None
        res = [self._possessive(re.sub(self.PRONOUN_PERSON, n, text, count=1, flags=re.I), n) for n in (latest, earlier)]
        return self._clarify(s, f"Just to be sure - do you mean {latest} or {earlier}?", "choose", [latest, earlier], res,
                             keys=[[latest.split()[-1]], [earlier.split()[-1]]])

    def _pick_person(self, s: Session, low: str) -> str:
        """he/him/his -> the most recent man (Mr.) or person of unknown gender (Dr., Prof.);
        she/her/hers -> the most recent woman (Mrs., Ms., Miss) or person of unknown gender."""
        hist = s.people_hist or ([s.entities["person"]] if s.entities.get("person") else [])
        if not hist:
            return s.entities.get("person", "")
        fem = re.search(r"\b(she|her|hers)\b", low)
        masc = re.search(r"\b(he|him|his)\b", low)
        if fem and masc:
            return hist[-1]

        def gender(name: str) -> str:
            if re.match(r"(Mrs|Ms|Miss)\b", name):
                return "f"
            if re.match(r"Mr\b", name):
                return "m"
            return "?"
        want = "f" if fem else "m"
        for name in reversed(hist):
            if gender(name) in (want, "?"):
                return name
        return hist[-1]

    def _entity_agreement(self, q: str, ans: str) -> float | None:
        """1.0 if every module / person the question names appears in the answer, 0.0 if none does,
        None if the question names no specific entity."""
        ql, al = normalize(q).lower(), ans.lower()
        named = [c.replace(" ", "").lower() for c in re.findall(r"\b([a-z]{2,3}\s?\d{4})\b", ql)]
        t = self.qa.match_course_title(ql)
        if t:
            named.append(self.qa.courses[t[0]]["title"].lower())
        p = self.qa.match_person(ql)
        if p:
            named.append(re.sub(r"\(.*?\)", "", p["name"]).split()[-1].lower())
        if not named:
            return None
        return sum(1 for n in named if n in al) / len(named)

    def _finish(self, s: Session, q: str, original: str, ans: str, kind: str, topic: str | None = None,
                confidence: float | None = None, reasoning: str | None = None) -> dict[str, Any]:
        """Common bookkeeping for answers produced outside the engine (fuzzy, ontology)."""
        self._update_state(s, q, ans, kind)
        if topic:
            s.topic = topic
        m = re.search(r"\(p\. (\d+)\)", ans)
        s.last_kind, s.last_page, s.last_evidence = kind, (int(m.group(1)) if m else None), ""
        if getattr(self, "_prefix", ""):
            ans = f"{self._prefix} {ans}"
            self._prefix = ""
        out = self._reply(ans, kind, confidence=confidence)
        if reasoning:
            out["reasoning"] = reasoning
        if normalize(q).lower() != normalize(original).lower() or getattr(self, "_fixes", None):
            out["interpreted_question"] = q
        if getattr(self, "_fixes", None):
            out["corrections"] = [{"typed": a, "read_as": b, "membership": m} for a, b, m in self._fixes]
        s.last_reply = out
        return out

    def _rule_answer(self, s: Session, q: str, original: str) -> dict[str, Any] | None:
        recent_module = s.entities.get("module") if self._recent(s, ["module", "person", "programme",
                                                                       "department"]) == "module" else None
        last_modules = s.last_list[1] if s.last_list and s.last_list[0] == "module" else []
        r = self.rules.answer(q, module=recent_module, last_modules=last_modules)
        if not r:
            return None
        st = r.get("state", {})
        if st.get("grade"):
            s.remember("grade", st["grade"])
        if st.get("module"):
            s.remember("module", st["module"])
        if st.get("person"):
            s.remember("person", st["person"])
            s.last_people = [st["person"]]
        if st.get("dept"):
            code = next((c for c, n in DEPARTMENTS.items() if n == st["dept"]), None)
            if code:
                s.remember("department", code)
        kind = r["kind"]
        s.topic = ("class" if kind.startswith("class") else "repeat" if kind in ("rule_repeat_limit",
                   "rule_grade_repeat") else "grade" if kind.startswith(("rule_grade", "rule_pass", "rule_weak"))
                   else s.topic)
        s.last_question, s.last_answer, s.last_kind = q, r["answer"], kind
        m = re.search(r"p\. (\d+)", r["answer"])
        s.last_page, s.last_evidence = (int(m.group(1)) if m else None), ""
        s.last_failed = None
        ans = r["answer"]
        if getattr(self, "_prefix", ""):
            ans = f"{self._prefix} {ans}"
            self._prefix = ""
        out = self._reply(ans, kind, options=r.get("options"))
        if normalize(q).lower() != normalize(original).lower():
            out["interpreted_question"] = q
        s.last_reply = out
        return out

    # ------------------------------------------------------------------ 2b. meta questions
    def _meta(self, s: Session, low: str) -> dict[str, Any] | None:
        l = low.strip(" ?.!")
        if not s.last_reply:
            return None
        if re.fullmatch(r"((can|could) you |please )?(repeat( that| it| the answer| again)?|say (that|it) again|"
                        r"come again|once more|one more time|what did you say|pardon|again)( please)?", l):
            r = dict(s.last_reply)
            r["kind"] = "repeat"
            return r
        if not s.last_question:
            return None
        if re.fullmatch(r"(where did you (get|find|read) (that|this|it)( from)?|what('?s| is) (the|your) source|"
                        r"source|sources|which page( is (that|it|this) (on|in))?|where is (that|this|it) "
                        r"(written|mentioned|stated|in the handbook)|reference|how do you know( that)?|"
                        r"where('?s| is) (that|this) from|is (that|this) (from|in) the handbook)", l):
            return self._reply(self._source_text(s), "source")
        if re.fullmatch(r"(why|why( is| was)? (that|it|so)|why not|how come|explain( why)?|reason|"
                        r"what('?s| is) the reason|why is that so)", l):
            return self._reply(self._why_text(s), "why")
        if re.fullmatch(r"(tell me more|more( details| info(rmation)?)?|explain( it)? more|elaborate|go on|"
                        r"(give|show) me more( details| info(rmation)?)?|details|more about (it|that|this)|"
                        r"can you (explain|elaborate)( more| further)?|anything else about (it|that))( please)?", l):
            return self._more(s)
        return None

    def _simpler(self, s: Session) -> dict[str, Any]:
        """\"I don't understand\" after an answer: restate its first point plainly and offer next steps."""
        ans = re.sub(r"\s*\(Student Handbook, p\.[^)]*\)", "", s.last_reply.get("answer", ""))
        ans = ans.split("\n\nDid this answer")[0]
        first = re.split(r"(?<=[.!?])\s+(?=[A-Z])", ans.strip())[0]
        bits = [f"Sorry, let me put it more simply. The key point: {first}"]
        if s.topic == "grade" or s.last_kind.startswith(("rule_pass", "rule_grade", "rule_weak", "grade")):
            bits.append("In short: 45% or more overall (with at least 35% in the exam and in the coursework) "
                        "is a pass.")
        elif s.topic == "class":
            bits.append("In short: your class depends on your final GPA (FGPA) and on how many failures you had.")
        bits.append("You can also ask \"which page?\" to read it yourself, or \"tell me more\".")
        return self._reply(" ".join(bits), "simplify", options=["Which page?", "Tell me more"])

    def _section_of_page(self, page: int | None) -> str:
        if not page:
            return ""
        secs = [x for x in self.qa.kb["sections"] if page in x.get("pages", [x["page"]]) and x["number"] != "0"]
        if not secs:
            return ""
        sec = max(secs, key=lambda x: x["level"])
        return f"Section {sec['number']} ({sec['title']})"

    def _source_text(self, s: Session) -> str:
        bits = ["That comes from the KDU Faculty of Computing Student Handbook 2022"]
        if s.last_page:
            sec = self._section_of_page(s.last_page)
            bits.append(f", page {s.last_page}" + (f", {sec}" if sec else ""))
        text = "".join(bits) + "."
        if s.last_evidence and s.last_evidence.strip() not in s.last_answer:
            text += f" The handbook says: \"{s.last_evidence[:300].strip()}\""
        return text

    def _why_text(self, s: Session) -> str:
        k, q = s.last_kind, s.last_question or ""
        if k.startswith("class"):
            res = self.qa.answer("what are the class requirements?")["answer"]
            return "Because classes are awarded by FGPA bands. " + res + " (Student Handbook, page 58.)"
        if k.startswith("grade"):
            res = self.qa.answer("what are the grades?")["answer"]
            return "Because the grade comes from the marks table. " + res + " The pass mark is 45% (grade C)."
        return "That is what the handbook states. " + self._source_text(s)

    def _more(self, s: Session) -> dict[str, Any]:
        target = self._recent(s, ["module", "person", "programme", "department"])
        if target == "module":
            code = s.entities["module"]
            c = self.qa.courses.get(code, {})
            descs = c.get("descriptions") or []
            if descs:
                d = descs[0]["text"]
                return self._reply(f"{code} - {c['title']}: {d}", "more_module")
        if target == "person":
            p = self.qa.match_person(s.entities["person"].lower())
            if p:
                bits = [f"{p['name']} - {p.get('designation', '')}".strip(" -")]
                if p.get("role"):
                    bits.append(f"Role: {p['role']}")
                if p.get("department"):
                    bits.append(f"Department: {DEPARTMENTS.get(p['department'], p['department'])}")
                for key, label in (("qualifications", "Qualifications"), ("research_areas", "Research areas"),
                                   ("email", "Email")):
                    if p.get(key):
                        bits.append(f"{label}: {p[key]}")
                return self._reply(". ".join(bits) + ".", "more_person")
        if target == "programme":
            res = self.qa.answer(f"compare {PROG_NAMES[s.entities['programme']]} degree and "
                                 f"{PROG_NAMES[s.entities['programme']]} degree")
            if res["kind"] == "compare_programmes":
                return self._reply(res["answer"].split(" | ")[0], "more_programme")
        alts = self.qa.alternatives(s.last_question, exclude=s.last_answer, k=2)
        if alts:
            return self._reply("Here is more related information from the handbook: " + " ".join(alts),
                               "more_related")
        return self._reply("That is all the handbook says about it. " + self._source_text(s), "more_none")

    # ------------------------------------------------------------------ 4b. several questions
    WH = r"(what|who|whom|whose|how|when|where|which|why|is|are|can|do|does|did|will|should)"

    def _split_questions(self, text: str) -> list[str]:
        t = text.strip()
        parts = [p.strip() for p in re.split(r"\?\s*(?:(?:and|also|plus|then)\b[,]?\s*)?(?=\S)", t) if p.strip()]
        if len(parts) == 1:
            parts = [p.strip() for p in re.split(rf"[,;]?\s+(?:and )?also\s+(?={self.WH}\b)", t, flags=re.I)
                     if p.strip()]
        if len(parts) == 1:
            m = re.match(rf"^(.{{12,}}?)\s*,?\s+and\s+({self.WH}\b.{{8,}})$", t, flags=re.I)
            if m and re.match(self.WH + r"\b", m.group(1), re.I) and \
                    not re.search(r"\b(between|both|compare|difference)\b", m.group(1), re.I):
                parts = [m.group(1), m.group(2)]
        parts = [p for p in parts if len(re.findall(r"[a-z0-9]+", p.lower())) >= 2]
        return parts if len(parts) > 1 else [text]

    # ------------------------------------------------------------------ 5b. plural references
    def _plural(self, s: Session, text: str, low: str) -> dict[str, Any] | None:
        if not s.last_list or not re.search(r"\b(they|them|their|theirs|these|those|both|each|all of them|"
                                            r"each of them)\b", low):
            return None
        kind, items = s.last_list
        if len(items) < 2 or self.qa.match_person(low) or programs_in(text) and kind != "programme":
            return None
        items = items[:6]
        if kind == "person" and re.search(r"\b(e-?mails?|research|designations?|departments?|qualifications?|"
                                           r"contact|positions?)\b", low):
            field_ = re.search(r"\b(e-?mail|research|designation|department|qualification|contact|position)",
                               low).group(1)
            field_ = {"contact": "email", "position": "designation"}.get(field_, field_)
            qs = [f"what is {n}'s {field_}?" for n in items]
        elif kind == "programme" and re.search(r"\b(how long|duration|years?|stream|credits?|semesters?|"
                                              r"departments?|entry|requirements?)\b", low):
            body = re.sub(r"\b(are|do|does|is) (they|these|those|both|them)\b", "is it", low)
            body = re.sub(r"\b(they|them|these|those|both|each of them|all of them)\b", "it", body)
            if re.search(r"\bmore credits|less credits|fewer credits|longer\b", low):
                return None
            qs = [re.sub(r"\bit\b", f"the {PROG_NAMES[p].lower()} degree", body, count=1) for p in items]
        elif kind == "module" and re.search(r"\b(credits?|semesters?|compulsory|electives?|about|level)\b", low):
            body = re.sub(r"\b(they|them|these|those|both|each of them|all of them)\b", "it", low)
            body = re.sub(r"\b(are|do) it\b", lambda m: ("is" if m.group(1) == "are" else "does") + " it", body)
            qs = [re.sub(r"\bit\b", c, body, count=1) for c in items]
        else:
            return None
        answers = []
        for q in qs:
            r = self.qa.answer(q)
            if not r.get("kind", "").startswith("not_found"):
                answers.append(r["answer"])
        if not answers:
            return None
        out = self._reply("\n".join("• " + a for a in answers), "plural_" + kind)
        s.last_reply = out
        s.last_answer = out["answer"]
        return out

    # ------------------------------------------------------------------ 5c. vague one-word questions
    VAGUE = {
        "credits": (["Credits of a module", "Credits in a semester", "Total credits of a degree"],
                    ["How many credits is CS2082?", "__semester_credits__",
                     "How many credits are in the computer science degree?"]),
        "modules": (["Modules in a semester", "Electives in a semester", "Find a module by name"],
                    ["What are the modules in semester 1?", "What are the electives in semester 6?",
                     "What is the module code of Machine Learning?"]),
        "grades": (["Grading scheme (marks to grades)", "Grade for my marks", "Pass mark"],
                   ["What are the grades?", "What grade do I get for 65 marks?", "What is the pass mark?"]),
        "exams": (["Attendance requirement", "Re-sit / repeat exams", "Supplementary exams"],
                  ["What is the attendance requirement?", "How do I re-sit a course?",
                   "When are supplementary exams held?"]),
        "gpa": (["How GPA is calculated", "Class for an FGPA", "GPA credits in a semester"],
                ["How is GPA calculated?", "What are the class requirements?", "__semester_credits__"]),
        "staff": (["Dean of the faculty", "Head of a department", "Staff by research area"],
                  ["Who is the dean?", "Who is the head of the computer science department?",
                   "Who researches machine learning?"]),
        "requirements": (["Entry requirements", "Attendance to sit exams", "Pass mark of a module",
                          "Class (FGPA) requirements"],
                         ["__ask__:what are the entry requirements?", "What is the attendance requirement?",
                          "What is the pass mark?", "What are the class requirements?"]),
        "departments": (["Departments of the faculty", "Head of a department", "Degrees of a department"],
                        ["What are the departments of the faculty?", "__ask__:who is the head of the department?",
                         "__ask__:what degrees does the department offer?"]),
        "degrees": (["Degrees offered", "Duration of a degree", "Entry requirements"],
                    ["What degrees are offered by the faculty?", "How long is the software engineering degree?",
                     "What are the entry requirements?"]),
    }
    VAGUE_WORDS = {"credit": "credits", "credits": "credits", "grade": "grades",
                   "grades": "grades", "marks": "grades", "exam": "exams", "exams": "exams",
                   "examination": "exams", "gpa": "gpa", "fgpa": "gpa", "staff": "staff", "lecturers": "staff",
                   "degree": "degrees", "degrees": "degrees", "programmes": "degrees", "programs": "degrees",
                   "requirements": "requirements", "requirement": "requirements", "rules": "requirements",
                   "criteria": "requirements", "regulations": "requirements", "lecturer": "staff",
                   "lecturers": "staff", "teachers": "staff", "department": "departments",
                   "head": "departments", "hod": "departments"}

    def _vague(self, s: Session, low: str) -> dict[str, Any] | None:
        w = re.sub(r"^((what|which) are (the |my )?|how many |list (the |all )?|show (me )?(the )?|"
                   r"tell me about (the )?|about (the )?|info on |information (on|about) |the |all )+", "",
                   low.strip(" ?.!"))
        topic = self.VAGUE_WORDS.get(w)
        if not topic:
            return None
        # with a recent module / person / programme the f2 ellipsis rule gives a better answer
        if topic in ("credits", "modules") and self._recent(s, ["module", "semester", "programme"]):
            return None
        labels, res = self.VAGUE[topic]
        return self._clarify(s, f"What would you like to know about {topic}: " +
                             "; ".join(l.lower() for l in labels) + "?", "choose", labels, res,
                             [[] for _ in labels], None)

    def _semester_credit_table(self) -> str:
        rows, seen = [], set()
        for t in self.qa.kb["semester_totals"]:
            prog = "SE" if t["section"] in ("3.3.5", "3.3.6") else ("CS" if t["semester"] >= 5 else "CS/SE")
            key = (t["semester"], prog)
            if key in seen:
                continue
            seen.add(key)
            rows.append(f"Semester {t['semester']} ({prog}): {fmt(t['gpa'])} GPA, {fmt(t['ngpa'])} NGPA, "
                        f"{fmt(t['mgpa'])} MGPA")
        return "Credits per semester - " + "; ".join(rows) + "."

    # ------------------------------------------------------------------ 6c. slot filling
    CURR = (r"\b(subjects?|modules?|courses?|course units?|electives?|optional (?:modules?|subjects?|courses?)|"
            r"compulsory (?:modules?|subjects?|courses?)|core (?:modules?|subjects?|courses?)|curriculum|syllabus|"
            r"course (?:structure|content|list))\b")
    CURR_NOT = (r"\b(fail\w*|re-?sit\w*|repeat\w*|pass(ed)?|grades?|regist\w*|drop\w*|attendance|exams?|"
                r"examinations?|gpv|gpa|ngpa|mgpa|exempt\w*|change|credit value|minimum|maximum|limit|"
                r"offered by the faculty|degree programmes?|programmes? (are )?offered|how many (courses|degrees?) "
                r"(are )?offered|department|what is an? (module|course|elective)|mean|definition|"
                r"choose between|select between|(choose|select) (a|my|the|your) (degree|programme|program|stream)|"
                r"can i (take|do)|prerequisite|a/?l|o/?l|stream|lecturer|teach\w*|who|module code|code of|"
                r"compulsory or elective|elective or compulsory)\b|^(is|are|does|do|can|was|has)\b")

    @staticmethod
    def _period(ql: str) -> tuple[str, str] | None:
        """('semester 5', 'semester') / ('year 3', 'year') / ('whole', 'whole') / None"""
        words = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
                 "final": 4, "last": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7,
                 "eighth": 8, "8th": 8, "one": 1, "two": 2, "three": 3, "four": 4}
        m = re.search(r"\bsem(?:ester)?\s*0?([1-8])\b|\b([1-8])(?:st|nd|rd|th)? sem(?:ester)?\b", ql)
        if m:
            return f"semester {m.group(1) or m.group(2)}", "semester"
        m = re.search(r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|seventh|7th|eighth|8th|"
                      r"final|last) sem(?:ester)?\b", ql)
        if m:
            return f"semester {words[m.group(1)]}", "semester"
        m = re.search(r"\b(?:year|level|lvl|yr)\s*0?([1-4]|one|two|three|four)\b|\b([1-4])(?:st|nd|rd|th)? "
                      r"(?:year|level)\b", ql)
        if m:
            v = m.group(1) or m.group(2)
            return f"year {words.get(v, v)}", "year"
        m = re.search(r"\b(first|1st|second|2nd|third|3rd|fourth|4th|final|last) (?:year|level)\b", ql)
        if m:
            return f"year {words[m.group(1)]}", "year"
        if re.search(r"\b(all|whole|entire|full|complete|overall|every)\b.{0,25}\b(degree|programme|program|"
                     r"course|years?|semesters?|curriculum|modules|subjects|courses)\b|\boverview\b|\bfull list\b|"
                     r"\ball of them\b", ql):
            return "whole", "whole"
        return None

    def _slot_value(self, slot: str, text: str, low: str) -> str | None:
        l = low.strip(" ?.!")
        if slot == "prog":
            if re.fullmatch(r"(both|both of them|cs and se|cs ?/ ?se|both cs and se|both please|dekama)", l):
                return "BOTH"
            if l in ("cs", "se", "ce", "it", "is"):
                return l.upper()
            ps = programs_in(text) or programs_in(text.upper() if len(l) <= 3 else text)
            return ps[0] if len(ps) == 1 else None
        if slot == "period":
            if re.fullmatch(r"[1-4]", l):
                return f"year {l}"
            per = self._period(l if re.search(r"year|level|sem|all|whole|overview|entire|full", l)
                               else "year " + l if re.fullmatch(r"(first|second|third|fourth|final|1st|2nd|3rd|4th)",
                                                                l) else l)
            return per[0] if per and per[1] != "whole" else None
        if slot == "sem":
            m = re.fullmatch(r"(?:sem(?:ester)?\s*)?([5-7])", l)
            if m:
                return f"semester {m.group(1)}"
            per = self._period(l)
            return per[0] if per and per[1] == "semester" else None
        if slot == "dept":
            d = self.qa.match_department(l)
            if not d:
                for code, name in DEPARTMENTS.items():
                    if l in (code.lower(), name.lower(), name.lower().replace("department of ", "")):
                        d = code
            return DEPARTMENTS[d] if d else None
        if slot == "student":
            if re.search(r"\b(civil|civilian|normal|day scholar|non[- ]?military)\b", l):
                return "civil"
            if re.search(r"\b(military|army|navy|air ?force|cadet|officer cadets?)\b", l):
                return "military"
        return None

    def _ask_slot(self, s: Session, msg: str, slot: str, template: str, labels: list[str],
                  resolutions: list[str | None], keys: list[list[str]] | None = None,
                  values: dict[str, str] | None = None) -> dict[str, Any]:
        out = self._clarify(s, msg, "choose", labels, resolutions, keys=keys)
        s.pending["slot"], s.pending["template"] = slot, template
        if values:
            s.pending["values"] = values
        s.pending["question"] = template
        return out

    def _slots(self, s: Session, q: str) -> dict[str, Any] | None:
        q = self.qa.correct_spelling(q)
        ql = normalize(q).lower()
        return (self._curriculum_slots(s, q, ql) or self._head_slot(s, ql) or self._entry_slot(s, ql)
                or self._teacher_reply(s, q, ql) or self._staff_slot(s, ql) or self._fail_slot(s, ql)
                or self._duration_reply(s, q, ql) or None)

    def _staff_slot(self, s: Session, ql: str) -> dict[str, Any] | None:
        if not re.search(r"^(who are|list|show( me)?|tell me|name)( all)? (the |all the )?(lecturers|academic staff|"
                         r"staff( members)?|teachers|instructors|lecturer list|staff list)\s*\??$", ql):
            return None
        labels = [DEPARTMENTS[d] for d in ("CS", "IT", "CE", "CM")]
        res = [f"who are the lecturers in the {DEPARTMENTS[d].lower()}?" for d in ("CS", "IT", "CE", "CM")]
        return self._ask_slot(s, "The handbook lists the academic staff by department. Which department?", "dept",
                              "who are the lecturers in the {}?", labels, res,
                              keys=[["cs", "computer science"], ["it", "information technology"],
                                    ["ce", "computer engineering"], ["cm", "maths", "mathematics", "computational"]])

    def _fail_slot(self, s: Session, ql: str) -> dict[str, Any] | None:
        if not re.search(r"^(what (happens|will happen|if)|what should i do|what do i do|what can i do)\b.*\b(fail|failed|"
                         r"failing|flunk\w*)\s*(it|that)?\??$|^(if|when) i fail\??$|^i (failed|fail)\b.{0,15}what now\??$",
                         ql):
            return None
        return self._clarify(s, "It depends on what you failed. Which of these is it?", "choose",
                             ["A module (re-sit)", "An exam I missed through illness", "A low SGPA (below 2.0)",
                              "Weak passes (D+ / C-)"],
                             ["How do I re-sit a course?", "what happens if I miss the exam because I'm sick",
                              "What happens if my SGPA is below 2.0?", "How many modules can I fail?"],
                             keys=[["module", "course", "subject", "re-sit", "resit"], ["ill", "sick", "medical",
                                   "missed"], ["sgpa", "gpa", "warning"], ["weak", "d+", "c-"]])

    def _duration_reply(self, s: Session, q: str, ql: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"(what is |whats |what's )?(the )?(duration|length|time period|period)( of (the )?(degree|"
                            r"course|programme|program|study|studies))?\s*\??", ql.strip()):
            return None
        if self._recent(s, ["programme"]) and s.entities.get("programme"):
            return None
        a = self.qa.answer("how long is the degree?")["answer"]
        b = self.qa.answer("what is the maximum duration to complete the degree")["answer"]
        out = self._reply(a + " " + b, "programme_duration",
                          options=["How many semesters are there?", "What is the maximum time to finish?"])
        s.last_reply = out
        return out

    def _curriculum_slots(self, s: Session, q: str, ql: str) -> dict[str, Any] | None:
        if not re.search(self.CURR, ql) or re.search(self.CURR_NOT, ql):
            return None
        if re.search(r"\b[a-z]{2,3}\s?\d{4}\b", ql) or self.qa.match_course_title(ql) or \
                self.qa.match_course_acronym(q):
            return None
        if not re.search(r"\b(list|what|which|show|give|provide|tell|name|all|display|see|know|available|are there|"
                         r"how many|number of|do (i|we) (have|take|study|learn|do|follow)|will (i|we)|study|learn|"
                         r"taught|need to (take|do|follow)|have to (take|do|follow)|offered|in|for|of)\b", ql) \
                and len(ql.split()) > 3:
            return None
        if re.search(r"\b(the|this|that|a|my|one|which) (module|subject|course|course unit)\b", ql) and \
                not re.search(r"\b(modules|subjects|courses|electives|list|all|every)\b", ql) and \
                not self._period(ql):
            if s.entities.get("module"):
                return None  # "the module" = the module talked about a moment ago
            return self._reply("Which module do you mean? Type its name or code, for example \"What is CS2082?\" "
                               "or \"Tell me about machine learning\".", "clarify_module",
                               options=["What is CS2082?", "Tell me about machine learning",
                                        "List the modules of a degree"])
        progs = self.qa.parse(normalize(q), q)["programs"] or ([s.user_programme] if s.user_programme else [])
        if progs and progs[0] not in CURRICULUM_PROGS:
            return None  # IT / IS / CE: the scope check explains that the handbook has no tables for them
        per = self._period(ql)
        elective = bool(re.search(r"\belectives?\b|\boptional\b", ql))
        kind = "elective modules" if elective else (
            "compulsory modules" if re.search(r"\b(compulsory|core|mandatory)\b", ql) else "modules")
        if progs and per and per[1] != "whole":
            return None  # fully specified: the engine answers
        if progs and per:
            return self._degree_overview(s, progs[0], False, kind)
        if per and per[1] != "whole" and self.logic is not None and kind == "modules":
            same = self.logic.same_modules(per[0])  # Prolog: same_modules(Sem) for every semester of the period
            if same:
                return None
        if per and per[1] != "whole":
            # period known, degree not: answer directly when CS and SE have exactly the same modules
            a_cs, a_se = (self.qa.answer(f"{kind} in {per[0]} for {p}")["answer"] for p in ("CS", "SE"))
            c_cs, c_se = set(re.findall(r"\b[A-Z]{2,3}\d{4}\b", a_cs)), set(re.findall(r"\b[A-Z]{2,3}\d{4}\b", a_se))
            if c_cs and c_cs == c_se:
                return None
            what = f"{kind} in {per[0].title()}"
            labels = ["Computer Science", "Software Engineering", "Both"]
            return self._ask_slot(s, f"The {kind} in {per[0].title()} are different for each degree. Which degree "
                                     f"programme do you mean?", "prog", f"{kind} in {per[0]} for {{}}", labels,
                                  [f"{kind} in {per[0]} for CS", f"{kind} in {per[0]} for SE",
                                   f"__multi__:{kind} in {per[0]} for CS||{kind} in {per[0]} for SE"],
                                  keys=[["cs", "computer science"], ["se", "software engineering"], ["both", "all"]])
        if not progs:
            labels = ["Computer Science", "Software Engineering", "Computer Engineering", "Information Technology",
                      "Information Systems"]
            tail = f" in {per[0]}" if per and per[1] != "whole" else ""
            res = [f"__ask__:{kind}{tail} for {PNAME(p)}" for p in ("CS", "SE", "CE", "IT", "IS")]
            if per and per[1] == "whole":
                res = [f"__overview__:{p}:0" if p in CURRICULUM_PROGS else f"__ask__:{kind} for {PNAME(p)}"
                       for p in ("CS", "SE", "CE", "IT", "IS")]
            tmpl = f"all {kind} of {{}}" if per and per[1] == "whole" else f"{kind} for {{}}"
            return self._ask_slot(s, f"Which degree programme do you mean? The {kind} are different for each "
                                     f"degree.", "prog", tmpl, labels, res,
                                  keys=[["cs"], ["se"], ["ce"], ["information technology"], ["information systems"]])
        # degree known, period missing
        p = progs[0]
        if elective:
            sems = sorted({o["semester"] for c in self.qa.courses.values() for o in c["offerings"]
                           if o["program"] == p and o["category"] == "Elective"})
            labels = [f"Semester {n}" for n in sems] + ["All electives"]
            res = [f"elective modules in semester {n} for {p}" for n in sems] + \
                  ["__multi__:" + "||".join(f"elective modules in semester {n} for {p}" for n in sems)]
            return self._ask_slot(s, f"{PROG_NAMES[p]} has electives in Semesters "
                                     f"{', '.join(str(n) for n in sems[:-1])} and {sems[-1]}. Which semester?",
                                  "sem", f"elective modules in {{}} for {p}", labels, res,
                                  keys=[[str(n)] for n in sems] + [["all", "every", "whole"]])
        labels = ["Year 1", "Year 2", "Year 3", "Year 4", "Whole degree overview"]
        res = [f"{kind} in year {n} for {p}" for n in range(1, 5)] + [f"__overview__:{p}:0"]
        return self._ask_slot(s, f"Which year of the BSc (Hons) in {PROG_NAMES[p]}? You can also type a semester "
                                 f"(1-8).", "period", f"{kind} in {{}} for {p}", labels, res,
                              keys=[["1", "first", "1st"], ["2", "second", "2nd"], ["3", "third", "3rd"],
                                    ["4", "fourth", "final", "4th"], ["whole", "all", "overview", "entire", "full"]])

    def _degree_overview(self, s: Session, p: str, full: bool, kind: str = "modules") -> dict[str, Any]:
        by_sem: dict[int, list[tuple[str, str, str]]] = {}
        for code, c in self.qa.courses.items():
            for o in c["offerings"]:
                if o["program"] == p:
                    by_sem.setdefault(o["semester"], []).append((code, c["title"], o["category"]))
        lines, total = [], 0
        for sem in sorted(by_sem):
            mods = sorted(set(by_sem[sem]))
            if kind == "elective modules":
                mods = [m for m in mods if m[2] == "Elective"]
            elif kind == "compulsory modules":
                mods = [m for m in mods if m[2] == "Compulsory"]
            if not mods:
                continue
            total += len(mods)
            if full:
                lines.append(f"Semester {sem}: " + "; ".join(f"{c} {t}" for c, t, _ in mods))
            else:
                el = sum(1 for m in mods if m[2] == "Elective")
                lines.append(f"Semester {sem}: {len(mods)} {kind}" + (f" ({el} electives)" if el and
                                                                       kind == "modules" else ""))
        head = f"BSc (Hons) in {PROG_NAMES[p]} - {kind} across the degree ({total} in total):"
        if full:
            out = self._reply(head + "\n" + "\n".join(lines), "degree_modules_full")
        else:
            out = self._reply(head + "\n" + "\n".join(lines) + "\nPick a year to see the module names.",
                              "degree_overview",
                              options=["Year 1", "Year 2", "Year 3", "Year 4", "Show all module names"])
            s.pending = {"type": "choose", "options": out["options"],
                         "resolutions": [f"{kind} in year {n} for {p}" for n in range(1, 5)] +
                                        [f"__overview__:{p}:1"],
                         "keys": [["1", "first"], ["2", "second"], ["3", "third"], ["4", "fourth", "final"],
                                  ["all", "names", "full", "everything"]],
                         "default": None, "slot": "period", "template": f"{kind} in {{}} for {p}",
                         "question": f"{kind} for {p}"}
        s.remember("programme", p)
        s.last_question, s.last_answer = f"{kind} for {p}", out["answer"]
        return out

    def _head_slot(self, s: Session, ql: str) -> dict[str, Any] | None:
        if not re.search(r"\b(who is|who's|whos|name of|tell me) (the )?(head|hod|head of (the )?department)\b|"
                         r"^(the )?(head|hod)( of (the )?department)?\??$", ql):
            return None
        if self.qa.match_department(ql) or programs_in(ql) or re.search(r"\b(faculty|dean)\b", ql):
            return None
        if s.entities.get("department") and self._recent(s, ["department", "person", "module"]) == "department":
            return None  # "who is the head?" right after a department question: context resolution handles it
        labels = [DEPARTMENTS[d] for d in ("CS", "IT", "CE", "CM")] + ["The Faculty (Dean)"]
        res = [f"who is the head of the {DEPARTMENTS[d].lower()}?" for d in ("CS", "IT", "CE", "CM")] + \
              ["who is the dean?"]
        return self._ask_slot(s, "The faculty has four departments, each with its own head. Which one do you mean?",
                              "dept", "who is the head of the {}?", labels, res,
                              keys=[["cs", "computer science"], ["it", "information technology"],
                                    ["ce", "computer engineering"], ["cm", "maths", "mathematics", "computational"],
                                    ["dean", "faculty"]])

    def _entry_slot(self, s: Session, ql: str) -> dict[str, Any] | None:
        if not re.search(r"\b(entry|entrance|admission|eligibility|selection|enrol\w*) (requirements?|criteria|"
                         r"qualifications?|conditions?)\b|\brequirements? (to|for) (enter|join|apply|get in|admission|"
                         r"entry|enrol\w*)\b|\bhow (can|do) i (apply|join|enter|get in|get selected)( (to|for) (kdu|the faculty|a degree|the degree|"
                         r"the programme|admission|this degree))?\s*\??$|\bwho can "
                         r"(apply|join)\b|\bqualifications? (needed|required) (to|for) (join|enter|apply)\b", ql):
            return None
        if re.search(r"\b(civil|military|army|navy|air ?force|cadets?|stream|a/?l|o/?l|english|age|height|weight|"
                     r"married|unmarried|citizen\w*|loans?|fees?|scholarships?|hostels?|medical|physical|"
                     r"exams?|re-?sit\w*|transfer\w*|leave|deferr?\w*)\b", ql):
            return None
        return self._ask_slot(s, "Entry requirements differ for civil and military students. Which one do you mean?",
                              "student", "what are the entry requirements for {} students?",
                              ["Civil students", "Military students", "Both"],
                              ["what A/L stream is needed for each degree?",
                               "what are the additional requirements for military students?",
                               "__multi__:what A/L stream is needed for each degree?||what are the additional "
                               "requirements for military students?"],
                              keys=[["civil", "civilian"], ["military", "army", "navy", "air force"], ["both", "all"]],
                              values={"civil": "what A/L stream is needed for each degree?",
                                      "military": "what are the additional requirements for military students?"})

    def _teacher_reply(self, s: Session, q: str, ql: str) -> dict[str, Any] | None:
        if not re.search(r"\bwho (teaches|is teaching|will teach|conducts|lectures|takes|handles|coordinates|is "
                         r"coordinating|runs|is in charge of)\b|\b(lecturer|"
                         r"teacher|instructor|module coordinator|coordinator) (for|of|in|who teaches)\b|\bwho is the "
                         r"(lecturer|teacher|instructor|coordinator)\b", ql):
            return None
        code = (re.search(r"\b([a-z]{2,3}\s?\d{4})\b", ql) or [None])[0]
        mt = self.qa.match_course_title(ql)
        if not code and not mt and self.qa.match_department(ql):
            return None  # "who teaches in the computer engineering department" -> staff list
        code = (code.upper().replace(" ", "") if code else None) or (mt[0] if mt else None) or \
            self.qa.match_course_acronym(q)
        msg = ("The Student Handbook does not say which lecturer teaches each module. It lists the academic staff "
               "of each department with their research areas.")
        opts = []
        if code and code in self.qa.courses:
            title = self.qa.courses[code]["title"]
            msg = f"The Student Handbook does not say who teaches {code} {title}. " + msg.split(". ", 1)[1]
            opts.append(f"Who researches {title.lower()}?")
            prog = self.qa.courses[code]["offerings"][0]["program"]
            opts.append(f"Who are the lecturers in the {DEPARTMENTS.get('CS' if prog in ('CS', 'SE') else prog, 'Department of Computer Science').lower()}?")
        else:
            opts += ["Who are the lecturers in the department of computer science?",
                     "Who researches machine learning?"]
        return self._reply(msg + " Your timetable or the LMS will name the lecturer for your module.", "not_found",
                           options=opts + ["Who is the dean?"])

    # ------------------------------------------------------------------ 6b. validation
    def _validate(self, q: str) -> dict[str, Any] | None:
        ql = normalize(q).lower()
        m = re.search(r"(?<![\w.])(-\d+)\s*(%|marks?|percent)", q.lower()) or \
            re.search(r"\b(\d{3,})\s*(%|marks?|percent)", ql) or \
            re.search(r"\b(?:marks?|score|got|get)\s+(?:of\s+)?(\d{3,}|-\d+)\b", ql)
        if m:
            v = float(m.group(1))
            if v > 100 or v < 0:
                return self._reply(f"{m.group(1)} marks is not possible: final marks are between 0 and 100. "
                                   "Please check the number and ask again.", "invalid_value",
                                   options=["What grade do I get for 72 marks?", "What are the grades?"])
        m = re.search(r"\b(?:f?gpa|gpv)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\b", ql) or \
            re.search(r"\b(\d+(?:\.\d+)?)\s*(?:f?gpa)\b", ql)
        if m and float(m.group(1)) > 4.2:
            return self._reply(f"An FGPA of {m.group(1)} is not possible: the highest grade point value is 4.20 "
                               "(grade A+), so a GPA/FGPA is between 0.00 and 4.20.", "invalid_value",
                               options=["What class is an FGPA of 3.5?", "What are the class requirements?"])
        m = re.search(r"\bsem(?:ester)?\s*0?(\d+)\b", ql)
        if m and not 1 <= int(m.group(1)) <= 8 and not re.search(r"\b\d+\s+semesters\b", ql):
            return self._reply(f"There is no semester {m.group(1)}: the degree programmes run from Semester 1 "
                               "to Semester 8 (4 levels, 2 semesters each).", "invalid_value",
                               options=["What modules are in semester 1?", "What are the electives in semester 7?"])
        m = re.search(r"\blevel\s*(\d+)\b", ql)
        if m and not 1 <= int(m.group(1)) <= 4:
            return self._reply(f"There is no level {m.group(1)}: the degrees have Levels 1 to 4.", "invalid_value")
        m = re.search(r"\b(?:dr|mr|mrs|ms|miss|prof|professor|eng)\.?\s+([a-z][a-z]{2,})\b", ql)
        if m and not self.qa.match_person(ql):
            surname = m.group(1)
            known = any(surname in st["name"].lower() for st in self.qa.kb["staff"]) or \
                any(surname in str(a).lower() for a in self.qa.kb.get("appointments", []))
            if not known:
                return self._reply(f"I couldn't find a staff member named \"{surname.title()}\" in the Student "
                                   "Handbook staff list. Check the spelling, or ask by role (e.g. \"head of the "
                                   "IT department\").", "unknown_person",
                                   options=["Who is the dean?", "Who is the head of the IT department?"])
        return None

    # ------------------------------------------------------------------ 2. pending states
    def _resolve_pending(self, s: Session, text: str, low: str) -> dict[str, Any] | None:
        p = s.pending
        s.pending = None
        if small_talk.classify(text)[0] == "uncertain" and p.get("type") != "verify":
            if p.get("slot") == "prog":
                low = text = "both"
            elif p.get("default") is not None:
                low = text = "yes"
            elif p.get("options"):
                s.pending = p
                return self._reply("No problem. Just pick the closest one below, or type your question in other "
                                   "words.", "clarify_help", options=p["options"])
        if p.get("slot"):
            val = self._slot_value(p["slot"], text, low)
            if val:
                if p["slot"] == "prog" and val == "BOTH":
                    qs = [p["template"].format(PNAME(x)) for x in ("CS", "SE")]
                    return self._reply("\n".join("• " + self.qa.answer(q)["answer"] for q in qs), "clarify_all")
                if p.get("values"):
                    return self._answer(s, p["values"][val], original=text, skip_checks=True)
                return self._one_question(s, p["template"].format(PNAME(val) if p["slot"] == "prog" else val))
        yes = re.fullmatch(r"(yes|yeah|yep|ok(ay)?|sure|please|correct|right|y|ow|ow ow|hari)[.! ]*(please)?[.!]*", low)
        no = re.fullmatch(r"(no|nope|nah|not really|not that|wrong|n|naha|epa)[.!,]*( thanks?| thank you)?[.!]*", low)
        if p["type"] == "verify":
            if yes:
                return self._reply("Great! Anything else you'd like to know?", "verify_ok")
            if no:
                alts = self.qa.alternatives(p["question"], exclude=p["answer"])
                topics = self.qa.suggest_topics(p["question"].lower())
                if alts:
                    return self._reply("Sorry about that. Here is other related information from the handbook: "
                                       + alts[0] + (" Closest topics:" + topics.split(":", 1)[1] if topics else ""),
                                       "verify_alternative", options=self._topic_options(topics))
                return self._reply("Sorry about that. Could you rephrase the question?" + topics,
                                   "verify_rephrase", options=self._topic_options(topics))
            return None  # a new question: handle normally
        # choose / confirm
        opts = p["options"]
        idx = None
        if yes and p.get("default") is not None:
            idx = p["default"]
        elif no and p.get("type") == "confirm":
            return self._reply("Okay. Ask me anything else about the handbook.", "clarify_declined",
                               options=EXAMPLES[:3])
        elif re.fullmatch(r"(both|all|all of them|both of them|each|every one|both please|all please|"
                          r"show both|show all|dekama|okkoma)[.!]*", low):
            qs = [q.replace("__ask__:", "") for q in p["resolutions"]
                  if q and (not q.startswith("__") or q.startswith("__ask__:"))]
            if len(qs) >= 2 and p.get("type") == "choose":
                answers = [self.qa.answer(q)["answer"] for q in qs]
                out = self._reply("\n".join("• " + a for a in answers), "clarify_all")
                s.last_reply = out
                return out
        else:
            m = re.search(r"\b(first|second|third|fourth|last|1st|2nd|3rd|4th|one|two|three)\b", low)
            if m and len(low.split()) <= 4:
                idx = ORDINALS[m.group(1)]
                idx = idx if idx >= 0 else len(opts) - 1
            elif re.fullmatch(r"[1-4]", low.strip(" .")):
                idx = int(low.strip(" .")) - 1
            else:
                for i, label in enumerate(opts):
                    ll = label.lower()
                    keys = p.get("keys", [[]] * len(opts))[i]
                    if low == ll or (len(low) > 2 and (low in ll or ll in low)) or any(
                            re.search(rf"\b{re.escape(k.lower())}\b", low) for k in keys):
                        idx = i
                        break
        if idx is not None and 0 <= idx < len(opts):
            q = p["resolutions"][idx]
            if q is None:
                return self._reply("Okay. Ask me anything else about the handbook.", "clarify_declined")
            if q.startswith("__ask__:"):
                return self._one_question(s, q[len("__ask__:"):])
            if q.startswith("__overview__:"):
                _, prog, full = q.split(":")
                out = self._degree_overview(s, prog, full == "1")
                s.last_reply = out
                return out
            if q.startswith("__multi__:"):
                answers = [self.qa.answer(x)["answer"] for x in q[len("__multi__:"):].split("||")]
                out = self._reply("\n".join("• " + a for a in answers), "clarify_all")
                s.last_reply = out
                return out
            if q.startswith("__text__:"):
                return self._reply(q[len("__text__:"):], "info")
            if q == "__semester_credits__":
                out = self._reply(self._semester_credit_table(), "semester_credit_table")
                s.last_reply = out
                return out
            return self._answer(s, q, original=text, skip_checks=True)
        return None  # not an answer to the clarification -> treat as a new question

    # ------------------------------------------------------------------ 4. feedback
    def _feedback(self, s: Session, low: str) -> dict[str, Any] | None:
        if not s.last_question:
            if re.fullmatch(r"(no|nope|nah|not really|naha|yes|yeah|yep)[.!]*", low):
                if s.last_topics:
                    return self._reply("No problem. These handbook topics are the closest I found: "
                                       + "; ".join(t.replace("Tell me about ", "") for t in s.last_topics)
                                       + ". Pick one, or rephrase your question.", "prompt", options=s.last_topics)
                return self._reply("Okay. What would you like to know? You can pick one of these or type your "
                                   "own question.", "prompt", options=EXAMPLES[:4])
            return None
        if re.fullmatch(r"(yes|yeah|yep|correct|right)[.!]*", low):
            return self._reply("Great! Anything else you'd like to know?", "verify_ok")
        bare_no = re.fullmatch(r"(no|nope|nah|not really|naha)[.!]*", low)
        if not bare_no and not re.fullmatch(r"(no[,.]? )?(that'?s|that is|this is|it'?s|your answer is)? ?(wrong|not (right|correct|"
                            r"what i (asked|meant|wanted))|incorrect|not true)( answer)?[.!]*|wrong answer[.!]*|"
                            r"not helpful[.!]*|that'?s not it[.!]*", low):
            return None
        alts = self.qa.alternatives(s.last_question, exclude=s.last_answer)
        topics = self.qa.suggest_topics(s.last_question.lower())
        msg = "Sorry about that."
        if alts:
            msg += f" Here is other related information I found: {alts[0]}"
        msg += " If that's still not it, could you rephrase or add a detail (module code, semester, degree)?"
        return self._reply(msg, "feedback", options=self._topic_options(topics))

    # ------------------------------------------------------------------ 5. user context statements
    NOT_NAMES = {"a", "an", "the", "student", "fine", "good", "ok", "okay", "here", "back", "confused", "lost", "not",
                 "so", "very", "in", "from", "doing", "going", "sorry", "stressed", "worried", "done", "new", "also"}

    def _user_context(self, s: Session, text: str, low: str) -> dict[str, Any] | None:
        # v10: "my name is Kasun" / "call me Nimal" -> used in greetings; "what is my name?" recalls it
        m = re.fullmatch(r"(?:hi|hello|hey)?[,! ]*(?:my name is|my name's|call me|this is|mage nama)\s+([a-z][a-z'-]{1,20})"
                         r"(?:\s+[a-z][a-z'-]{1,20})?[.! ]*", low)
        if m and m.group(1) not in self.NOT_NAMES:
            s.user_name = m.group(1).title()
            return self._reply(f"Nice to meet you, {s.user_name}! What would you like to know from the Student "
                               "Handbook?", "context_name", options=EXAMPLES[:3])
        if re.fullmatch(r"(what('?s| is) my name|do you (know|remember) my name|who am i)\s*\??", low):
            if s.user_name:
                return self._reply(f"You told me your name is {s.user_name}.", "context_name")
            return self._reply("You haven't told me your name yet. You can say \"my name is ...\".", "context_name")
        m = re.search(r"\b(i'?m|i am|im|mama)\s+(an?\s+)?(in\s+)?(the\s+)?(.+?)\s*(student|undergraduate|batch|"
                      r"degree|program+e?)?[.!]*$", low)
        if m and not re.search(r"\?|\b(what|which|how|who|when|where|can|is)\b", low):
            progs = programs_in(m.group(5)) + [p for p in ("CS", "SE", "CE", "IT", "IS")
                                                if re.search(rf"\b{p}\b", text)]
            if progs:
                p = progs[0]
                s.user_programme = p
                s.remember("programme", p)
                note = ""
                if p not in CURRICULUM_PROGS:
                    note = (" Note: the handbook lists module tables only for Computer Science and Software "
                            "Engineering, so module questions will be answered for those degrees.")
                return self._reply(f"Got it, you're in BSc (Hons) in {PROG_NAMES[p]}. I'll use that for "
                                   f"programme-specific questions.{note}", "context_set")
        m = re.fullmatch(r"(my )?(fgpa|gpa|final gpa)( is| =)? ?(\d(\.\d{1,2})?)[.!]*", low)
        if m:
            s.remember("fgpa", float(m.group(4)))
            return self._reply(f"Noted, FGPA {float(m.group(4)):.2f}. Ask me which class it gives or what you "
                               f"need for the next class.", "context_set",
                               options=["What class will I get?", "What do I need for first class?"])
        return None

    # ------------------------------------------------------------------ 6. context resolution
    PRONOUN_PERSON = r"\b(he|she|his|her|him|hers)\b"
    PRONOUN_THING = r"\b(it|its|it's|this|that|this one|that one)\b"

    SINGLISH_PRONOUNS = {r"\b(eyage|eyaage|eyge|eyagey)\b": "his", r"\b(eya|eyaa)\b": "he",
                         r"\b(ekata|eke)\b(?=.*\?)?": "its"}

    def _resolve_context(self, s: Session, text: str, low: str) -> str:
        q = text
        for pat, eng in list(self.SINGLISH_PRONOUNS.items())[:2]:
            q = re.sub(pat, eng, q, flags=re.I)
        lq = (s.last_question or "").lower()
        # (a0) number only ("58" after a marks question, "3.5" after an FGPA question)
        m = re.fullmatch(r"(?:and |what about |how about )?(\d+(?:\.\d+)?)\s*(marks?|%)?\s*\??", text.strip().lower())
        if m and s.last_question:
            v = m.group(1)
            if re.search(r"\b\d+(\.\d+)?\s*(marks?|%)", lq):
                return re.sub(r"\b\d+(\.\d+)?(?=\s*(marks?|%))", v, s.last_question, count=1)
            if re.search(r"\bf?gpa\b", lq) and re.search(r"\d\.\d", lq) and "." in v:
                return re.sub(r"\d\.\d+", v, s.last_question, count=1)
            if re.search(r"\bsemester \d", lq) and "." not in v and int(v) <= 8:
                return re.sub(r"(semester )\d", rf"\g<1>{v}", s.last_question, count=1, flags=re.I)
        # (a1) remembered marks: "did I pass?", "what grade is that?"
        if s.entities.get("marks") is not None and not re.search(r"\d", text) and \
                re.search(r"\b(pass(ed)?|fail(ed)?|grade|gpv)\b", text.lower()) and \
                re.search(r"\b(i|my|me|that|it)\b", text.lower()):
            mk = s.entities["marks"]
            if re.search(r"\bpass|fail", text.lower()):
                return f"Do I pass with {mk} marks?"
            return f"What grade do I get for {mk} marks?"
        # (a2) "which one has more credits?" after a comparison / list of programmes or modules
        if s.last_list and re.search(r"\b(which|what) (one|of them|degree|programme|module|course)?\s*(has|have|is|"
                                     r"carries|gives)\b.*\b(more|most|less|fewer|least|longer|shorter)\b",
                                     text.lower()):
            kind, items = s.last_list
            if kind == "programme" and len(items) >= 2:
                return (f"compare {' and '.join(items[:3])} degrees, "
                        + re.sub(r"^.*?\b(more|most|less|fewer|least|longer|shorter)\b", r"which has \1",
                                 text.lower()).rstrip("?") + "?")
            if kind == "module" and len(items) >= 2:
                return f"compare {' and '.join(items[:3])}, " + \
                    re.sub(r"^.*?\b(more|less|fewer)\b", r"which has \1", text.lower()).rstrip("?") + "?"
        # (a3) "the first one / the second one" -> n-th module / programme of the last answer list
        m = re.search(r"\b(?:the )?(first|second|third|fourth|last|1st|2nd|3rd|4th) (one|module|course|subject|"
                      r"degree|programme|program)\b", text.lower())
        if m and s.last_list and s.last_list[0] in ("module", "programme"):
            kind, items = s.last_list
            i = ORDINALS[m.group(1)]
            if abs(i) <= len(items) and i < len(items):
                item = items[i]
                name = item if kind == "module" else f"the {PROG_NAMES[item].lower()} degree"
                return re.sub(re.escape(m.group(0)), name, text, count=1, flags=re.I)
        # (a0') v10: he/she/his/her -> the most recent person of matching gender, before the engine's generic
        #       follow-up rewrite (which always takes the person of the previous question)
        if re.search(self.PRONOUN_PERSON, q.lower()) and s.entities.get("person") and \
                not self.qa.match_person(normalize(q).lower()) and len(s.people_hist) > 1:
            person = self._pick_person(s, q.lower())
            if person != s.entities["person"]:
                s.remember("person", person)
                q = re.sub(self.PRONOUN_PERSON, person, q, count=1, flags=re.I)
                return self._possessive(q, person)
        # (a) slot / pronoun follow-ups relative to the last question (engine rules)
        if s.last_question:
            q = self.qa.resolve_followup(q, s.last_question)
        ql = normalize(q).lower()
        own = self.qa.parse(normalize(q), q)
        has_subject = bool(own["codes"] or self.qa.match_course_title(ql) or self.qa.match_person(ql))
        # (b) "the first one / the second one" -> the n-th person of the last answer list
        m = re.search(r"\bthe (first|second|third|last|1st|2nd|3rd) (one|person|lecturer|name)\b", ql)
        if m and s.last_people:
            i = ORDINALS[m.group(1)]
            name = s.last_people[i] if -len(s.last_people) <= i < len(s.last_people) else None
            if name is None:  # "the second one" when only one person was listed
                return text
            q = re.sub(m.group(0), name, q, flags=re.I)
            return q
        # (c) person pronouns -> last person (question or answer)
        if re.search(self.PRONOUN_PERSON, ql) and not self.qa.match_person(ql) and s.entities.get("person"):
            person = self._pick_person(s, ql)  # v10: gender agreement with the recent people
            if person != s.entities["person"]:
                s.remember("person", person)
            q = re.sub(self.PRONOUN_PERSON, person, q, count=1, flags=re.I)
            q = re.sub(r"\b(" + re.escape(person) + r")\s+(email|research|designation)", r"\1's \2",
                       q, flags=re.I)
            return q
        # (d) "that / this / the department", "there" -> last department
        dept = s.entities.get("department")
        if dept and re.search(r"\b(that|this|the|same) (department|dept)\b", ql):
            q = re.sub(r"\b(that|this|the|same) (department|dept)\b", DEPARTMENTS[dept], q, flags=re.I)
            return q
        if dept and re.search(r"\b(staff|lecturers?|instructors?|members?|head)\b", ql) and \
                re.search(r"\bthere\b|\bin it\b", ql) and not self.qa.match_department(ql):
            return re.sub(r"\b(there|in it)\b", f"in the {DEPARTMENTS[dept]}", q, flags=re.I)
        # (e) "it / this" -> most recent of module / programme / department
        if re.search(self.PRONOUN_THING, ql) and not has_subject and not re.search(
                r"\bit\s+(department|dept|degree|programme|program|stream|students?)\b", ql):
            target = self._recent(s, ["module", "semester", "department", "programme"])
            if target == "semester" and re.search(r"\b(credits?|modules?|subjects?|electives?|courses?)\b", ql):
                prog = next(iter(programs_in(s.last_question or "")), None)
                q = re.sub(r"\b(does it have|it has|in it|does it carry|is it)\b",
                           f"in semester {s.entities['semester']}", q, count=1, flags=re.I)
                if f"semester {s.entities['semester']}" not in q.lower():
                    q = re.sub(r"\b(it|this semester|that semester)\b", f"semester {s.entities['semester']}", q,
                               count=1, flags=re.I)
                if prog and not programs_in(q):
                    q = f"{q.rstrip('?. ')} for {prog}?"
                return q
            if target == "semester":
                target = self._recent(s, ["module", "department", "programme"])
            if target == "module":
                q = re.sub(r"\b(it's|its|it|this module|that module|this course|that course|this one|that one)\b",
                           s.entities["module"], q, count=1, flags=re.I)
            elif target == "programme":
                name = f"the {PROG_NAMES[s.entities['programme']].lower()} degree"
                q = re.sub(r"\b(it's|its|it|this degree|that degree|this programme|that programme|this one|"
                           r"that one)\b", name, q, count=1, flags=re.I)
            elif target == "department":
                q = re.sub(r"\b(it's|its|it)\b", DEPARTMENTS[s.entities["department"]], q, count=1, flags=re.I)
            ql = normalize(q).lower()
        # (f) ellipsis: "what about the credits?" -> keep the last semester / module / programme
        m = re.match(r"^(?:what|how) about (?:the )?(.+?)\??$", ql) or re.match(r"^and (?:the )?(.+?)\??$", ql)
        if m and q == text:
            topic = m.group(1)
            new_own = self.qa.parse(normalize(topic), topic)
            if not (new_own["codes"] or new_own["semester"] or new_own["programs"] or self.qa.match_person(topic)):
                if s.entities.get("semester") and re.search(r"\b(credits?|modules?|subjects?|electives?|gpa|"
                                                            r"ngpa|mgpa|compulsory)\b", topic):
                    q = f"{'how many ' if re.search(r'credits?|gpa', topic) else ''}{topic} in semester " \
                        f"{s.entities['semester']}"
                elif s.entities.get("module") and self._recent(s, ["module", "person"]) == "module":
                    q = f"{topic} of {s.entities['module']}"
                elif s.entities.get("person") and self._recent(s, ["module", "person"]) == "person":
                    q = f"{s.entities['person']} {topic}"
        # (f2) very short question without its own subject: "which semester?", "credits?", "email?"
        if q == text and len(ql.split()) <= 3 and not has_subject and not own["semester"] and not own["programs"]:
            target = self._recent(s, ["module", "person", "programme"])
            if target == "module" and re.search(r"\b(semester|credits?|elective|compulsory|code|level|year|"
                                                r"category|about|description)\b", ql):
                q = f"{q.rstrip('?. ')} of {s.entities['module']}?"
            elif target == "person" and re.search(r"\b(e-?mail|research|designation|department|qualifications?|"
                                                  r"phd|position)\b", ql):
                q = f"{s.entities['person']} {q.rstrip('?. ')}?"
            elif target == "programme" and re.search(r"\b(duration|how long|stream|department|credits?|"
                                                     r"semesters?)\b", ql):
                q = f"{q.rstrip('?. ')} of the {PROG_NAMES[s.entities['programme']].lower()} degree?"
        # (g) remembered programme for programme-specific questions
        ql = normalize(q).lower()
        if s.user_programme and not programs_in(q) and not re.search(r"\b(CS|SE|CE|IT|IS)\b", q) and \
                re.search(r"\b(semester|electives?|level|year modules|degree credits|total credits)\b", ql) and \
                not re.search(r"\bsemesters? (and|vs)\b", ql):
            q = f"{q.rstrip('?. ')} for {PROG_NAMES[s.user_programme].lower()}"
        # (h) FGPA remembered: "what class will I get?"
        if s.entities.get("fgpa") is not None and re.search(r"\bclass\b", ql) and not re.search(r"\d\.\d", ql) and \
                re.search(r"\b(i|my|me)\b", ql) and not re.search(r"\b(need|required|requirement)\b", ql):
            q = f"What class do I get with an FGPA of {s.entities['fgpa']:.2f}?"
        return q

    @staticmethod
    def _possessive(q: str, person: str | None) -> str:
        if not person:
            return q
        return re.sub(r"(" + re.escape(person) + r")\s+(email|e-mail|research|designation|qualifications?|department)",
                      r"\1's \2", q, flags=re.I)

    @staticmethod
    def _recent(s: Session, names: list[str]) -> str | None:
        have = [(s.seen.get(n, -1), -i, n) for i, n in enumerate(names) if s.entities.get(n) is not None]
        return max(have)[2] if have else None

    # ------------------------------------------------------------------ 7. scope verification
    def _scope_check(self, s: Session, q: str) -> dict[str, Any] | None:
        ql = normalize(q).lower()
        ctx = self.qa.parse(normalize(q), q)
        progs = [p for p in ctx["programs"]]
        curriculum = re.search(r"\b(modules?|subjects?|electives?|course units?|credits?|semester|syllabus|"
                               r"curriculum)\b", ql) or (re.search(r"\b(first|second|third|fourth|final) year\b|"
                                                                   r"\blevel [1-4]\b", ql))
        # (a) curriculum details for a programme the handbook has no tables for
        if curriculum and progs and not set(progs) & set(CURRICULUM_PROGS) and not ctx["codes"] and \
                not re.search(r"\b(how long|duration|years? (is|does)|stream|department|offer(s|ed)? by|head|"
                              r"a/?l|o/?l|advanced level|ordinary level|entry|admission|eligib\w*|apply|"
                              r"required to (enter|join)|needed (for|to) (enter|join|apply))\b", ql) and \
                not re.search(r"\b(subjects?|passes) (needed|required)\b.*\b(a/?l|o/?l)\b|\bat a/?l\b", ql):
            p = progs[0]
            base = re.sub(r"\b(bsc\s*(\(hons\)\s*)?(in\s+)?)?(the\s+)?(" + "|".join(
                [PROG_NAMES[x] for x in progs] + [x for x in progs]) + r")(\s+(degree|programme|program|students?))?",
                          "", q, flags=re.I)
            base = re.sub(r"\b(of|for|in) the\s*(\?|$)|\s+(of|for|in)\s*(\?|$)", " ", base).strip(" ?")
            msg = (f"The Student Handbook lists the module and credit tables only for BSc (Hons) in Computer "
                   f"Science and Software Engineering; it does not give them for BSc (Hons) in {PROG_NAMES[p]}.")
            if p in ("IT", "IS"):
                msg += " (IT and IS share a common curriculum in the first two years, but their modules are not listed.)"
            msg += " Would you like the Computer Science / Software Engineering information instead?"
            return self._clarify(s, msg, "confirm",
                                 ["Yes, show CS and SE", "Computer Science", "Software Engineering", "No thanks"],
                                 [base, f"{base} for computer science", f"{base} for software engineering", None],
                                 keys=[["both", "cs and se", "cs/se"], ["cs", "computer science"],
                                       ["se", "software engineering"], ["no"]], default=0)
        # (b) module title shared by CS and SE copies
        if not ctx["codes"] and not (set(progs) & set(CURRICULUM_PROGS)) and not s.user_programme:
            for title, codes in self.shared_titles.items():
                if re.search(rf"\b{re.escape(title)}\b", ql):
                    labels, res = [], []
                    for code in codes:
                        prog = self.qa.courses[code]["offerings"][0]["program"]
                        labels.append(f"{PROG_NAMES[prog]} ({code})")
                        res.append(re.sub(re.escape(title), code, q, flags=re.I))
                    msg = (f"\"{self.qa.courses[codes[0]]['title']}\" exists in more than one degree: "
                           + " and ".join(labels) + ". Which one do you mean?")
                    keys = [[self.qa.courses[c]["offerings"][0]["program"], c,
                             PROG_NAMES[self.qa.courses[c]["offerings"][0]["program"]]] for c in codes]
                    return self._clarify(s, msg, "choose", labels, res, keys=keys)
        # (c) staff surname shared by several people
        if not re.search(r"\b(dr|mr|mrs|ms|miss|eng|maj)\.?\s+[a-z]{2,}\b", ql):
            for surname, people in self.shared_surnames.items():
                if re.search(rf"\b{surname}\b", ql):
                    # a first name / initials in the question that picks one person
                    picks = [st for st in people if any(
                        p.lower() in ql.split() for p in st["name"].replace(".", " ").split()[1:-1] if len(p) > 2)]
                    if len(picks) == 1:
                        return None
                    labels = [f"{st['name']} ({DEPARTMENTS.get(st['department'], '')})" for st in people]
                    res = [re.sub(rf"\b{surname}\b", st["name"], q, flags=re.I) for st in people]
                    return self._clarify(s, f"There are {len(people)} staff members named {surname.title()}. "
                                            f"Which one do you mean?", "choose", labels, res,
                                         keys=[[st["email"].split("@")[0]] for st in people])
        return None

    def _clarify(self, s: Session, msg: str, typ: str, labels: list[str], resolutions: list[str | None],
                 keys: list[list[str]] | None = None, default: int | None = None) -> dict[str, Any]:
        s.pending = {"type": typ, "options": labels, "resolutions": resolutions, "keys": keys or [[]] * len(labels),
                     "default": default}
        return self._reply(msg, "clarify", options=labels)

    # ------------------------------------------------------------------ 8-10. answer + verification + state
    def _answer(self, s: Session, q: str, original: str, skip_checks: bool = False) -> dict[str, Any]:
        if self._gibberish(q):
            intent = small_talk.fallback(original) or small_talk.classify(original)[0]
            if intent:
                reply, examples = small_talk.reply_for(intent, small_talk.normalise(original))
                return self._reply(reply, "small_talk_" + intent, EXAMPLES[:4] if examples else None)
            return self._reply("Sorry, I didn't understand that. You can ask me things like:", "not_understood",
                               options=EXAMPLES[:4])
        res = self.qa.answer(q)
        if self.logic is not None:
            res = self.logic.verify(q, res)  # grade / class answers cross-checked with the Prolog rules
        ans = res.get("answer", "")
        kind = res.get("kind", "")
        options = None
        if (kind.startswith("not_found") or (kind == "retrieval" and res.get("confidence", 1) < 0.6)) and \
                not skip_checks:
            intent = small_talk.fallback(original)
            if intent:  # the handbook has nothing for it, and it reads like small talk
                reply, examples = small_talk.reply_for(intent, small_talk.normalise(original))
                return self._reply(reply, "small_talk_" + intent, EXAMPLES[:4] if examples else None)
        if kind == "not_found" and not skip_checks:
            fb = self.rules.section_fallback(q)
            if fb:
                res, ans, kind = dict(res, kind=fb["kind"], page=None), fb["answer"], fb["kind"]
                options = fb.get("options")
        if kind.startswith("not_found"):
            topics = self.qa.suggest_topics(normalize(q).lower())
            options = self._topic_options(topics) or EXAMPLES[:3]
            s.last_topics = self._topic_options(topics) or []
            if "I can help with" not in ans and kind == "not_found":
                ans += " I can help with modules, credits, grades, exams, degree programmes and staff details."
        elif kind == "retrieval" and re.match(r"^[A-Z][^.]{2,80}? [A-Z]{2,3}\d{4} (This|The) (module|course)", ans) \
                and not re.search(r"\b[a-z]{2,3}\s?\d{4}\b", normalize(q).lower()) \
                and not self.qa.match_course_title(normalize(q).lower()) and not self.qa.match_course_acronym(q):
            # the best evidence is one module's description, but the question named no module: ask first
            return self._clarify(s, "I'm not sure what you're asking. Do you mean one of these?", "choose",
                                 ["Modules of a degree", "About a specific module", "Rules for modules (re-sit, "
                                  "attendance, pass mark)"],
                                 ["__ask__:list the modules", "__text__:Type the module name or code, for example "
                                  "\"What is CS2082?\" or \"Tell me about machine learning\".",
                                  "__multi__:What is the pass mark?||What is the attendance requirement?||"
                                  "How do I re-sit a course?"],
                                 keys=[["degree", "list", "all"], ["specific", "one module", "name"],
                                       ["rules", "re-sit", "attendance", "pass"]])
        elif kind == "retrieval":
            # v11 fuzzy decision rule base: answer / verify / fail from five fuzzy inputs
            ev = res.get("evidence") or ans
            action, score, rules = self.fuzzy_dec.decide(
                res.get("confidence", 1), self.fuzzy_dec.coverage(q, ev), self._entity_agreement(q, ans),
                self.signals.topic_agreement(q, res.get("page")), self.signals.type_agreement(q, ans))
            self._fuzzy_trace = {"action": action, "score": score, "rules": rules}
            if action == "fail":
                ans = ("I couldn't find a reliable answer to that in the handbook. The closest passage I found, with "
                       f"low confidence, is: \"{ans.strip()}\"\n\nDid this answer your question?")
            elif action == "verify":
                ans += "\n\nDid this answer your question?"
            if action != "answer":
                options = ["Yes", "No"]
                s.pending = {"type": "verify", "question": q, "answer": ans}
        self._update_state(s, q, ans, kind)
        if not kind.startswith("not_found"):
            s.last_kind, s.last_page, s.last_evidence = kind, res.get("page"), res.get("evidence") or ""
        if getattr(self, "_prefix", ""):
            ans = f"{self._prefix} {ans}"
            self._prefix = ""
        out = self._reply(ans, kind, options=options, confidence=res.get("confidence"))
        if getattr(self, "_fuzzy_trace", None):
            out["fuzzy_decision"] = self._fuzzy_trace
            self._fuzzy_trace = None
        if getattr(self, "_fixes", None):
            out["corrections"] = [{"typed": a, "read_as": b, "membership": m} for a, b, m in self._fixes]
            out.setdefault("interpreted_question", q)
        understood = res.get("interpreted_question") or (q if normalize(q).lower() != normalize(original).lower()
                                                           else None)
        if understood:
            out["interpreted_question"] = understood
        s.last_reply = out
        return out

    def _update_state(self, s: Session, q: str, ans: str, kind: str) -> None:
        if kind.startswith("not_found"):
            s.last_failed = q
            return
        if kind in ("clarify",):
            return
        s.last_failed = None
        if kind.startswith(("class", "logic_class")):
            s.topic = "class"
        elif kind.startswith(("grade", "logic_grade")):
            s.topic = "grade"
        elif kind.startswith("logic_repeat"):
            s.topic = "repeat"
        elif not kind.startswith(("small_talk", "retrieval")):
            s.topic = None
        s.last_question = q
        s.last_answer = ans
        ql = normalize(q).lower()
        ctx = self.qa.parse(normalize(q), q)
        code = next((c for c in ctx["codes"] if c in self.qa.courses), None)
        if not code:
            m = self.qa.match_course_title(ql)
            code = m[0] if m else self.qa.match_course_acronym(q)
        if not code and kind.startswith("course"):
            m = re.match(r"([A-Z]{2,3}\d{4})\b", ans)
            code = m.group(1) if m else None
        if code:
            s.remember("module", code)
        progs = [p for p in ctx["programs"]]
        if len(progs) == 1:
            s.remember("programme", progs[0])
        if ctx["semester"]:
            s.remember("semester", ctx["semester"])
        m = re.search(r"\b(\d{1,3})\s*(marks?|%)", ql) or (re.match(r"(?:Yes\. |No\. )?(\d{1,3}) marks", ans)
                                                          if kind.startswith("grade_lookup") else None)
        if m and re.search(r"\b(i|my|me|i'm)\b", ql):
            s.remember("marks", int(m.group(1)))
        codes = list(dict.fromkeys(re.findall(r"\b([A-Z]{2}\d{4})\b", ans)))
        progs_ans = [c for c, n in PROGRAMS.items() if n.lower() in ans.lower()] if kind.startswith(
            ("department", "compare_programmes", "programme")) else []
        progs_ans = sorted(progs_ans, key=lambda c: ans.lower().find(PROGRAMS[c].lower()))
        person = self.qa.match_person(ql) or (self.qa.match_person(normalize(ans).lower())
                                              if kind in ("person_role", "appointment", "person_info") or
                                              kind.startswith("person") else None)
        if person:
            s.remember("person", person["name"])
        names = re.findall(r"((?:Dr|Mr|Mrs|Ms|Miss|Eng|Maj|Prof)\.\s?(?:\(Eng\.\)\s)?[A-Z][\w.]*(?:\s[A-Z][\w-]+)*)",
                           ans)
        s.last_people = names
        if len(names) >= 2:
            s.last_list = ("person", names)
        elif len(progs_ans) >= 2:
            s.last_list = ("programme", progs_ans)
        elif len(codes) >= 2:
            s.last_list = ("module", codes)
        elif s.last_list and any(it.lower() in ql or PROG_NAMES.get(it, "~").lower() in ql
                                 for it in s.last_list[1]):
            pass  # follow-up on one item of the list: keep the list for "the second one"
        else:
            s.last_list = None
        if len(names) == 1 and not person:
            s.remember("person", names[0])
        dept = None
        if re.search(r"\bdepartment\b", ql):
            dept = self.qa.match_department(ql)
        m = re.search(r"offered by the (Department of [A-Za-z ]+?)\.", ans) or \
            re.search(r"^The (Department of [A-Za-z ]+?) offers", ans)
        if m:
            dept = next((c for c, n in DEPARTMENTS.items() if n == m.group(1).strip()), dept)
        if dept:
            s.remember("department", dept)

    # ------------------------------------------------------------------ helpers
    def _gibberish(self, q: str) -> bool:
        toks = [t for t in tokens(q, drop_stop=False) if t.isalpha()]
        if not toks:
            return not re.search(r"\d", q)
        known = sum(1 for t in toks if t in self.qa.vocab or stem(t) in self.qa.bm25.df or len(t) <= 2)
        if known or self.qa.parse(normalize(q), q)["codes"]:
            return False
        # unknown words that still look like real words ("library hours") are a "not found", not gibberish
        wordlike = [t for t in toks if re.search(r"[aeiou]", t) and not re.search(r"[^aeiouy]{4,}", t)
                    and not re.search(r"(asdf|qwer|zxcv|hjkl|jkl;|sdfg|dfgh|ghjk)", t)]
        return len(wordlike) < max(1, len(toks) // 2 + 1) if len(toks) > 1 else not wordlike

    @staticmethod
    def _topic_options(topics: str) -> list[str] | None:
        if not topics:
            return None
        items = [re.sub(r"\s*\([\d.]+\)", "", t).strip(" .") for t in topics.split(":", 1)[1].split(";")]
        return [f"Tell me about {t.lower()}" for t in items if t][:3] or None

    @staticmethod
    def _reply(text: str, kind: str, options: list[str] | None = None, confidence: float | None = None) -> dict:
        # the PDF contains broken Word cross-references ("section Error! Reference source not found.")
        text = re.sub(r"(section|clause)\s+Error! Reference source not found\.\s*\(\s*(Faculty of Computing "
                      r"By-Laws)\s*\)?", r"the \2", text)
        text = re.sub(r"Error! Reference source not found\.", "(see the Faculty of Computing By-Laws)", text)
        out = {"answer": text, "kind": kind, "llm_used": False}
        if options:
            out["options"] = options
        if confidence is not None:
            out["confidence"] = confidence
        return out

    def _sid_of(self, s: Session) -> str:
        return next((k for k, v in self.sessions.items() if v is s), "default")

    def _expire(self) -> None:
        now = time.time()
        for k in [k for k, v in self.sessions.items() if now - v.updated > self.SESSION_TTL]:
            del self.sessions[k]
