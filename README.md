# Student Handbook Assistant v11 (CPU-only, no LLM: fuzzy rule base + auto-generated ontology + Prolog)

Answers questions about the Faculty of Computing Student Handbook without any LLM,
embedding model, ONNX model or GPU. Typical answer time: ~1.5 ms on a normal CPU.

## Host it online for free
See **DEPLOY.md** (Render free tier: GitHub -> Render Blueprint, about 10 minutes).

## Run
    pip install -r requirements.txt
    python app.py            # open http://127.0.0.1:8000

Put `Student-Handbook-2022.pdf` next to `app.py` to index it automatically, or upload it
from the page. The knowledge base is cached in `data/handbook_kb.json`.


## What's new in v11: fuzzy rule base, automatic ontology generation, Prolog over the ontology
Design principle: the handbook's regulations are **crisp** (FGPA >= 3.70 is First Class; 45 marks is C), so
they stay crisp in Prolog. Fuzzy logic is used only where the **input** or the **decision** is imprecise.

### 1. Fuzzy rule base (`fuzzy_layer.py`)
**a) Fuzzy matching of typos.** Every unknown word has a membership degree in each handbook term (string
similarity). It is corrected only if: membership >= a length-dependent threshold, the best term clearly beats the
runner-up, the edit distance (Damerau-Levenshtein) is at most 1 (2 for words longer than 7 letters), the word
is not a known English word or an inflection of one ("headed", "teaching"), and the term keeps the first
letter. Examples: hed -> head, departmnt -> department, suplementary -> supplementary, machin lerning -> machine
learning, Kathriarachi -> Kathriarachchi. Corrections and their memberships are returned in the API
(`corrections`) and shown as "Understood as".

**b) Fuzzy linguistic terms.** "Am I close to first class with 3.65?", "is 3.1 good?", "how far is 3.28 from
second upper?", "is 44 marks almost a pass?". The crisp class comes from the Prolog facts `class_band/3`; fuzzy
sets on the gap (very close / close / moderately far / far) and an FGPA standing scale (weak / fair / good /
very good / excellent, labelled as the assistant's own scale) describe it. Yes/no questions ("is 3.2 enough for
second upper?") start with a crisp Yes/No.

**c) Fuzzy decision rule base** for retrieved sentences: ANSWER / VERIFY / FAIL from five fuzzy inputs:
retrieval confidence, question-evidence coverage, entity agreement (do the module/person in the question appear
in the answer?), topic agreement (does the evidence come from the handbook chapter the question is about?) and
answer-type agreement ("how many" -> a number, "who" -> a person). Nine Mamdani-style rules, min for AND, max
aggregation, weighted-average defuzzification. The trace is returned in the API (`fuzzy_decision`).

Calibration (175 benchmark retrieval answers, all correct, plus 11 known-wrong retrieval sentences):

| | v10 (confidence < 0.6) | v11 fuzzy rules |
|---|---|---|
| correct answers hidden | 0 | 0 |
| correct answers with an extra "Did this answer?" | 18 | 13 |
| known-wrong sentences flagged | 0/11 | 5/11 |

Finding: the engine's own confidence is ~0.95 even for wrong sentences, so the topic and answer-type inputs
carry most of the discrimination. The 6 unflagged wrong sentences come from the right chapter (partly relevant
rules); in the full pipeline those questions are answered earlier by the v10 rule layer.

### 2. Automatic ontology generation (`ontology.py`)
A hand-designed schema (Faculty, Department, Programme, Module, Staff, Grade, ClassAward, ResearchTopic and 20
relations) is **populated automatically** every time a handbook PDF is indexed:

| source | confidence | examples |
|---|---|---|
| tables | 1.00 | module tables, staff list, grade and class tables |
| role strings | 0.95 | "Head - Department of Information Technology" -> head_of |
| text patterns | 0.85 | "The department offers two degrees, namely BSc (Hons) in ..." -> offers |
| derived | 0.90 | research topics split from each research-area text |

Every fact carries provenance `prov(Fact, Page, Source, Confidence)`; a fact found by two independent sources is
confirmed. A validation pass checks consistency (one department per programme, one head per department,
offerings point to known programmes, text and tables agree).

Result for the 2022 handbook: **2,013 facts, all 5 `offers` facts confirmed by table + text, 0 validation
issues.** The validator proved useful while building it: a wider text window first produced false facts
("CS offers CE/IT/IS"); the check flagged them. Reading the text showed a real, different relation - the CS
department "offers a large percentage of computing courses for BSc (Hons) in Computer Engineering, IT and IS"
- so v11 models service teaching as its own relation, `teaches_for(Department, Programme)`.

Outputs (regenerated at start-up and after every upload): `data/handbook_ontology.pl` (Prolog, also loads in
SWI-Prolog), `data/handbook_ontology.ttl` (RDF/Turtle, opens in Protege), `data/ontology_report.json`.
`GET /api/ontology` returns the counts and the validation result. Run `python ontology.py data/handbook_kb.json`
to regenerate by hand.

### 3. Prolog over the ontology (`ontology_rules.pl`, `ontology_query.py`)
General rules replace hand-coded special cases, e.g.

```prolog
head_of_programme(P, H)   :- offers(D, P), head_of(H, D).
department_of_module(M, D):- in_programme(M, P), offers(D, P).
only_in(M, P, Other, Sem) :- offered(M, P, Sem), \+ offered(M, Other, Sem).
researcher_for_programme(T, P, S) :- offers(D, P), research_topic(S, T), member_of(S, D).
```

`ontology_query.py` recognises the entities in a question (module, programme, department, staff, research
topic, semester), runs the goal, and shows the reasoning chain and page:

> **which department offers machine learning?** CS4042 - Machine Learning is offered by the Department of
> Computer Science (it is part of CS). (p. 35) How I worked it out: CS4042 -> in_programme CS -> offers ->
> Department of Computer Science.

Question types: department of a module; head of the department that offers a module/degree; degrees of the
department headed by a person; does department X offer (degree Y / a degree at all); is a module in CS / SE /
both; modules only in one degree (per semester); modules shared by CS and SE (list or count); which degrees
include a module; who researches topic T in the department that offers degree P.

### Tests (blind sets written before the code, first run reported)

| set | v10 | v11 first run | v11 final |
|---|---|---|---|
| **blind V11 (24 turns: typos, vague wording, multi-hop)** | 14/24 (58.3%) | used for development | 24/24 |
| **blind V11b (17 turns, fresh wording, not used for tuning)** | 11/17 (64.7%) | **16/17 (94.1%)** | 17/17 |
| all earlier dialog, small-talk and logic suites | 100% | - | 100% (unchanged) |
| all single-question sets through the dialog manager | 1,934/1,935 | - | 1,934/1,935 |
| fuzz test (960 messages) | - | - | 0 errors, 3.5 ms/message |

Start-up of fuzzy layer + Prolog + ontology: about 0.1 s.

Fixes found by testing: the corrector first changed real words (visa -> visual, bored -> board, reset -> resit,
headed -> head) - fixed with the edit-distance limit and inflection protection. The fuzz test found two crashes
(both caught, no stack trace shown): "the second one" after a one-person answer (an index bug from v5), and
"electives in semester 1" (there are none; the engine indexed an empty list). Both are fixed. The v11b miss
("am I safely in first class?") was a missing vague word ("safely").

Run: `python evaluate_dialog.py x tests/blind_v11_tests.json --fails` and `tests/blind_v11b_tests.json`.

## What was new in v10: logical answers, topic context, gender-aware pronouns, graceful failing
New probes showed answers that were found by keyword but were wrong for the question: "what if I get a D?"
returned a sentence about PhD careers, "what is the passing grade?" returned a broken sentence with the PDF's
"Error! Reference source not found.", and "and second upper?" lost the class topic.

**New file `conversation_rules.py`** - rule answers built from the knowledge base and the exact text of
Sections 4-6, each with its page:

| Question | Answer |
|---|---|
| what is the passing grade? / marks to pass it? | C: 45% overall and at least 35% in every component (module-specific when a module is in context) |
| what if I get a D / E / F? | no such grade; D+ and C- are weak passes, below that Ie / Ia / Ib |
| what if I get C- / D+ / Ie / Ia / Ib / Ab / Ne / Ex? | meaning, GPV, pass / weak pass / fail, what to do next |
| what is a weak pass? / repeat to improve a weak pass? | D+/C-, one per semester, repeat only up to C, effect on classes |
| how many times can I repeat? | no fixed number; 16 semesters / 8 years, re-sit charge, supplementary only after S7-S8 |
| requirements for a pass (degree)? | Pass FGPA band + degree-award criteria (6.1.1) |
| head of software engineering / who heads IS? | no separate department: the CS / IT department and its head |
| prerequisites? / who coordinates it? / which is hardest? | not in the handbook (credits shown instead of difficulty) |

**Small talk.** New intents: uncertain ("maybe", "idk", "not sure", "nvm"), compare_ai ("are you smarter
than ChatGPT?"), wait ("hold on a sec"). General knowledge ("capital of France", "who discovered
penicillin") is out of scope. Name memory: "my name is Kasun" / "call me Nimesha", used in greetings and
thanks; "what's my name?". "I don't understand" after an answer restates its key point simply.

**Pronoun handling.** he/his -> most recent Mr. (or Dr./Prof.), she/her -> most recent Mrs./Ms./Miss (or
Dr./Prof.), resolved before the engine's older follow-up rewrite. If the latest person is a Dr. and an
earlier one matches the pronoun, the bot asks which one. A pronoun right after a turn that found nobody
asks "Do you mean X?" instead of silently reusing an older person. "it" can now be a grade ("can I repeat it?").

**Context management.** The session tracks the topic (class / grade / repeat): "and second upper?",
"what about first?", "what about C-?", "and Ia?", "how many times?" are answered in that topic.

**Dialog management.** "I don't know" / "not sure" in reply to a clarification takes the default
(e.g. both CS and SE) or repeats the options.

**Graceful failing.** Topics the handbook never covers (hostel, library, canteen, scholarships, dress
code, timetables and dates, transport, Wi-Fi/LMS, clubs, counselling, plagiarism, internships) say so and
give the faculty contacts printed in the handbook. Unknown departments ("music department") list the four
real ones. Section fallback: when nothing is found but a section title matches ("attendance rule" ->
4.3 Attendance), the opening of that section is quoted and labelled as such. Broken cross-references
("Error! Reference source not found.") are removed from every answer.

**Tests (written before running, first run reported honestly).**

| set | first run | final (v10) |
|---|---|---|
| **blind v10 dialogs (40 turns / 18 conversations)** | **35/40 (87.5%)** | 40/40 |
| **blind v10b dialogs (fresh wording)** | **22/23 (95.7%)** | 24/24 |
| all earlier dialog, small-talk and logic suites | - | unchanged (100%) |
| all single-question sets through the dialog manager | 1,934/1,935 | 1,934/1,935 |
| fuzz test (960 messages incl. HTML, SQL, 3,000 chars, emoji) | - | 0 errors, ~3.9 ms/message |

Disclosed test change: in blind v10b, "what are her research areas?" after Mrs. Sirisuriya and then
Dr. Hettige originally expected Sirisuriya; v10 asks which person is meant (the handbook gives no gender for
"Dr."), so the test now expects the question and then "the second one". Two old tests expect the phrase
"couldn't find" for hostel/library questions; the new not-covered message keeps that phrase.

Run the new sets: `python evaluate_dialog.py x tests/blind_v10_dialogs.json --fails`
(and `tests/blind_v10b_dialogs.json`).

## What was new in v8: a Prolog logic layer
Keyword retrieval can find a rule, but it cannot *combine* rules: "FGPA 3.8 with one failure" needs the
FGPA band and the failure-grade conditions of four classes at once. v8 adds a Prolog layer for that.

**Files**
- `handbook_rules.pl` – the handbook's rules as Prolog clauses (147 lines, each with its page):
  grades and components, semester completion, the SGPA formula, the four classes, graduation, repeat
  rules, admission streams, military-applicant checks, curriculum reasoning, slot requirements.
- `data/handbook_facts.pl` – generated from the PDF at start-up: 161 `unit/9` module facts, grade
  bands, class bands, programmes, departments, staff, heads, credit totals, heights, military limits.
- `prolog_engine.py` – a small pure-Python Prolog interpreter (unification, backtracking, negation as
  failure, if-then-else, findall, aggregate_all, arithmetic). Nothing to install; CPU only.
- `logic_layer.py` – builds the facts, loads the rules, turns reasoning questions into Prolog goals and
  verifies the engine's grade and class answers.

The two `.pl` files also run in SWI-Prolog unchanged:

```
swipl data/handbook_facts.pl handbook_rules.pl
?- best_class(3.8, 1, 0, 0, 0, C).          % C = 'Second Class (Upper Division)'
?- semester_problem(['A','Ie','C-','D+'], R).
?- gpa_of_modules([m('CS3023','A+'), m('CS3032','B')], S, _, _).   % S = 3.72
```

**What it answers that v7 could not**

| Question | Prolog goal | Answer |
|---|---|---|
| calculate my gpa: A in CS3023, B+ in CS3032 and C in CS3052 | `gpa_of_modules/4` | (4.00×3 + 3.30×2 + 2.00×2) / 7 = 3.22 |
| I got A, B, C- and D+ in semester 3, did I pass? | `semester_problem/2` | No: 2 weak passes, only one allowed |
| can I get first class with FGPA 3.8 and one failure? | `best_class/6`, `blocker/7` | Second Class (Upper Division), and why not First |
| FGPA 3.75, failed one module in semester 7 | same | Pass (no failure allowed in semesters 7-8 for any Second Class) |
| can I graduate with a GPA of 1.9? | `graduation_problem/3` | No: below 2.00 |
| I'm 23, can I apply as a military student? | `military_problem/1` + `military_unknown/1` | No (age 18-22) + the other checks |
| female, 155 cm, navy cadet? | `height_min/3` | No: minimum 160 cm |
| biology stream, can I do software engineering? | `stream_ok/2` | No (needs Maths) |
| CA 30 and exam 70, what grade? | `grade_of_components/5` | Ia: every component needs 35% |
| can I repeat a module with D+? | `repeat_rule/2` | weak pass: may repeat, up to C |
| how many electives in semester 5 for CS? | `elective_credits_needed/3` | 19 - 17 = 2 elective GPA credits |

The dialog manager asks Prolog first for these question types. The engine's grade and class
answers are cross-checked with `grade_of/3` and `best_class/6`; on a mismatch the rule-based
answer wins. The slot logic (eq. 23) asks Prolog whether CS and SE have the same modules in a semester
(`same_modules/1`), so it does not ask for the degree when the answer would be the same. A start-up
consistency check (`credit_mismatch/4`) confirms that the module tables add up to the credit summary
tables for every semester without electives (no mismatch found).

**Blind test of the logic layer (30 reasoning questions, gold worked out by hand from the rules, run
once):** 27/30 (90%) on the first run. One failure was a real bug: a module listed in both the CS and SE
tables was counted twice in the GPA. The other two were wording that wasn't recognised. After fixing:
30/30. Start-up of the Prolog layer: under 0.1 s. A 720-message fuzz test with reasoning questions gave
0 errors at 2.8 ms per message.

## What was new in v7: ask before answering a doubtful question
The v6 screenshot showed two bad answers. "list the subjects ?" returned the Computational Mathematics
department's subject areas, and "provide the subjects in the degree programs" returned one module's
description. Neither question says which degree or year, so v7 asks first.

**Slot filling for under-specified questions.** The dialog manager checks which details a question is
missing and asks for them one at a time, with clickable options. A typed reply ("SE", "year 2",
"semester 5", "final year", "both") fills the slot as well as a click does.

| Question | What is missing | The bot asks |
|---|---|---|
| list the subjects / what are the modules? | degree, then year | Which degree? → Which year? (or whole-degree overview) |
| modules in semester 5 / subjects in the 3rd year | degree (only when CS and SE differ) | CS / SE / Both |
| what are the electives? | degree, then semester | Semester 5 / 6 / 7 / All electives |
| who is the head? / who are the lecturers? | department | the four departments (+ Dean) |
| what are the entry requirements? | civil or military | Civil / Military / Both |
| what happens if I fail? | what was failed | a module / an exam missed through illness / low SGPA / weak passes |
| credits? / requirements / grades | which meaning | 3-4 options |
| tell me about the module | which module | type a name or code |

When the degree is known (typed in the question or said earlier, e.g. "I'm a software engineering
student"), only the year is asked. When the year's modules are the same for CS and SE (Years 1-2,
Semesters 1-4), the answer comes straight away with the label "CS/SE". IT, IS and CE still get the
"tables are only given for CS and SE" message.

**Whole-degree overview.** Module counts per semester (with electives), then "Year 1-4" or "Show all
module names".

**Awkward-answer guard.** If the best retrieval evidence is one module's description but the question
names no module, the bot asks what was meant instead of showing that description.

**New handbook answers (all checked against the PDF).** KDU vision and mission; awards and trophies;
how many D+/C- grades are allowed per semester; passing GPA (2.00); how students are assessed (CA 30% /
WE 70%); medium of instruction (English); the SGPA formula; how re-sitting works (Ie/Ia/Ib); changing
degree (not covered; the IT/IS specialisation rule is quoted); "who teaches X" (the handbook does not
say; offers the staff-by-research-area question instead); credit hours (not defined in the handbook).

**Small talk.** 33 intents now (+ celebration, good luck, "I'm fine", language). New phrases such as
istuti / bohoma istuti / nanri, gn all, ammo / aiyo, "all the best for me", "happy birthday", "can you
speak Sinhala". A keyword fallback catches new wording ("you are so dumb", "tell me a funny joke"), but
only after the handbook search has found nothing and only when the message has no handbook word in it,
so a real question is never swallowed. Real-looking unknown words ("library hours") now get "couldn't
find" instead of "didn't understand".

**First run of the new blind tests (written before running, run once):** dialogs 26/30 (86.7%), small
talk 21/40 (52.5%). After fixes: 30/30 and 40/40. A 720-message fuzz test gave 0 errors (3.0 ms per
message).

## What was new in v6
**Small talk (`small_talk.py`, 29 intents).** Greetings with the time of day (good morning / gud mrng /
gm / suba udesanak), good night, how are you, what's up, who are you, who made you, are you a bot /
ChatGPT, thanks, bye, praise, ok, sorry, lol, insults ("ur useless"), stress ("I'm worried I'll fail"
gets re-sit / supplementary / medical options), good news, jokes, "testing 123", time/weather/news
questions (politely out of scope). Messages are normalised first: elongations (hiiii, thankss),
chat spellings (thx, tnx, u, ur, hru, c u, nite), emoji, filler words (machan, bro, sir, please).
"good morning! what is the faculty email?" gets "Good morning!" plus the answer.

**Pronouns and lists.** they / them / their after a list of people ("what are their emails?" answers
each person), after a list of degrees ("how long are they?"), after a list of modules; "the first
one / the second one" for modules and degrees as well as people; "it" can now be a semester
("how many modules in semester 5 for SE?" -> "how many credits does it have?"), and the most specific
entity wins a tie (module > semester > department > programme).

**Context.** Remembers your marks ("I got 68 marks for databases" -> "did I pass?"), a number on its
own reuses the last question ("what grade for 72 marks?" -> "58"), "which one has more credits?"
after a comparison, "both" / "all" answers every option of a clarification.

**Dialog management.** Several questions in one message ("X? also Y", "who is the dean and what is
his email?") are answered in order, with pronouns resolved between them. Meta questions about the
last answer: "can you repeat that?", "where did you get that?" / "which page?" (page and section),
"why?" (the rule behind a class or grade), "tell me more" (full module description, full staff
record, or more related handbook text).

**Graceful failing.** Impossible values are caught before answering (150 marks, -5 marks, FGPA 5.2,
semester 10, level 6). Unknown staff names ("Dr. Perera") get "I couldn't find a staff member...".
One-word questions ("credits", "grades", "exams", "staff") get a choice of three meanings. Sinhala or
Tamil script gets a polite "please type in English or Singlish". Topics the handbook does not cover
(online / part-time degrees, part-time jobs) say so instead of returning an unrelated sentence. A
missed exam because of illness is mapped to the Excuse rule (grade Ex, re-sit as a first attempt).
A 480-message fuzz test (HTML, SQL, 3,000-character input, emoji, odd follow-ups) gave 0 errors,
2.6 ms per message on average.

## What was new in v5: dialog manager (`dialog_manager.py`)
Every chat message goes through a rule-based dialog pipeline with per-session state:
- **Small talk**: hi / how are you / who are you / help / thanks / bye (English + Singlish: ayubowan,
  kohomada, isthuthi ...). "hello, who is the dean?" answers the question.
- **Pronoun handling**: he/she/his/her (+ Singlish eya/eyage) -> last person (from the question *or*
  the answer); it/this -> most recent module / programme / department; "that department", "there",
  "the first one" (n-th person of the last answer list).
- **Context management**: remembers module, person, programme, department, semester, FGPA; "I'm a
  software engineering student" applies SE to later semester/elective/credit questions; ellipsis
  ("what about the credits?", "which semester?"); "start over" / New chat clears the context.
- **Dialog management**: pending clarification state; replies like "yes", "SE", "the second one", "2"
  resolve it; a new question simply replaces it.
- **Verification follow-up for doubtful scopes**: asks instead of guessing when
  (a) module/credit details are asked for IT, IS or CE (the handbook only has CS and SE tables),
  (b) a module title exists in both CS and SE (Industrial Training, Independent Study, Research Project),
  (c) several staff share a surname (Wijesinghe), (d) a retrieval answer has low confidence
  ("Did this answer your question?").
- **Graceful failing**: not-found answers with clickable topic suggestions, "didn't understand" for
  gibberish, apology + next-best evidence for "that's wrong", no stack traces.
- Engine: new "how many semesters in the degree" handler; raw table text removed from retrieval
  (it caused the "Table 40 GPA Summary ..." answer).
- API: `POST /api/chat {session_id, message}` -> {answer, options, interpreted_question, kind},
  `POST /api/reset {session_id}`. UI shows clickable options and a New chat button.
  `/api/ask` still works for single questions.

## What was new in v4
- Yes/No answers computed from the records: "Is Machine Learning compulsory?" -> "Yes. ...",
  "Is Automata Theory an elective?" -> "It depends on the degree: ..."
- Comparisons: "compare CS and SE degrees", "CS1012 vs CS1073", "which has more credits, X or Y?"
- "Closest handbook topics" suggestions when an answer is not in the handbook
- Module acronyms built from titles (AI, HCI, NLP, ...), more Singlish (keeyak, liyanna, thiyenne)
- Retrieval: answer-type filtering (number questions), IDF-weighted coverage with synonym/word-family
  credit, time expressions required for "when" questions, better list-item selection
- 831-question benchmark (`make_benchmark.py`, gold parsed independently from the PDF)

## What was new in v3
- Follow-up questions: "what about semester 5?", "how many credits is it?", "what is his email?"
  (the UI sends the previous question and shows "Understood as: ...")
- Singlish question words (kiyada, mokakda, kauda, wenna one, ganna, koheda ...)
- Transposition typo correction (cerdit -> credit), re-sat/resit normalisation
- Department -> degree records, module "aspect" answers (language, tools, topics)
- IDF-weighted, synonym-aware evidence coverage; answer-type cues for who/when/where
- Benchmark builder with gold answers parsed independently from the PDF

## Files
- `handbook_qa.py` – engine (PDF -> knowledge base -> question parsing -> answer)
- `app.py` – FastAPI server: /api/status, /api/upload, /api/ask {question, previous_question}
- `index.html` – chat UI
- `make_benchmark.py` – builds the 831-question benchmark (`tests/benchmark800.json`)
- `fuzzy_layer.py` – v11 fuzzy matching, fuzzy linguistic answers, fuzzy decision rule base
- `ontology.py` – v11 automatic ontology generation (schema, provenance, validation, Prolog + Turtle export)
- `ontology_rules.pl`, `ontology_query.py` – v11 Prolog rules over the ontology and question-to-goal mapping
- `conversation_rules.py` – v10 rule answers (grades, pass mark, repeats, Pass class), not-covered topics, section fallback
- `dialog_manager.py` – conversation layer (state, pronouns, lists, clarification, validation, meta questions)
- `small_talk.py` – small-talk normaliser, intent classifier and keyword fallback
- `logic_layer.py`, `prolog_engine.py`, `handbook_rules.pl`, `data/handbook_facts.pl` – Prolog reasoning layer (v8)
- `tests/blind_logic_tests.json` – 30 reasoning questions (run with `evaluate_dialog.py`)
- `evaluate_smalltalk.py` – small-talk intent tests (`tests/smalltalk_tests.json`, `tests/blind_v6_smalltalk.json`)
- `evaluate_dialog.py` – multi-turn conversation tests (`tests/dialog_tests.json`, `tests/dialog_blind_tests.json`)
- `evaluate_via_dialog.py` – runs single-question sets through the dialog manager
- `evaluate.py` – accuracy test (supports `re:` regex expectations, follow-up `prev`, `--session`), e.g. `python evaluate.py Student-Handbook-2022.pdf tests/bench500_questions.json --fails --session`
- `bench_build.py` – regenerates the benchmark from the PDF (`--blind` = new wording, 150 questions)
- `tests/` – all question sets + `benchmark_results.csv` (v2 vs v3 answer for every benchmark question)

## Accuracy
### Conversations, small talk and reasoning (v8)
| set | first run | final (v8) |
|---|---|---|
| **blind logic tests (30 reasoning questions, run once)** | **27/30 (90.0%)** | 30/30 |
| **blind v7 dialogs (doubtful questions, 30 turns / 15 conversations, run once)** | **26/30 (86.7%)** | 30/30 |
| **blind v7 small talk (40 messages, run once)** | **21/40 (52.5%)** | 40/40 |
| v7 clarification regression dialogs (34 turns / 18 conversations) | - | 34/34 |
| **blind v6 small talk (30 messages, written before running, run once)** | **23/30 (76.7%)** | 30/30 |
| **blind v6 dialogs (30 turns / 15 conversations, run once)** | **29/30 (96.7%)** | 30/30 |
| small talk intents (200 messages) | v5: 76/200 | 200/200 |
| v6 dialog features (51 turns / 30 conversations) | v5: 29/51 | 51/51 |
| dialog tests (60 turns / 22 conversations, incl. the 2 screenshot issues) | v4 engine: 28/59 | 60/60 |
| blind dialog tests from v5 (42 turns) | 39/42 (92.9%) | 42/42 |
| all single-question sets through the dialog manager | - | 1,934/1,935 |

The one remaining "failure" through the dialog manager is "What are the research areas of
Wijesinghe?": two staff members have that surname, so the assistant asks which one. That is the
intended behaviour.

### Single questions
| set | first run | final (v6) |
|---|---|---|
| **benchmark800 (831; 591 templated + 240 hand-written)** | **v3: 809/831 (97.4%)** | 831/831 |
| **blind60 v4 (casual / Singlish, run once)** | **48/60 (80.0%)** | 58/60 |
| advanced (yes/no + compare) | 5/24 before the features | 24/24 |
| bench500 (509) | v2: 489/509 (96.1%) | 509/509 |
| blind150 (new wording, run once) | 149/150 (99.3%) | 150/150 |
| final100 (unseen, hand-written) | 92/100 | 100/100 |
| test120 | 105/125 | 125/125 |
| holdout | 37/40 | 40/40 |
| gold (dev) | - | 81/81 |
| follow-up pairs | 12/15 | 15/15 |

Total over all 10 sets: 1,935/1,935. Mean latency ~2 ms.

First-run numbers are the honest estimates; final numbers are optimistic because those
questions were used to improve the system. 100% on these sets does not mean 0% errors on new
questions: the blind first runs (76.7% small talk, 96.7% dialogs, 80% casual single questions) are
the realistic estimate for wording nobody has tested yet. Casual phrasing and slang remain the
weakest area. Test-set corrections in v6 (disclosed): "Hons degree" now accepts "4 years" as well as
"four"; the blind v5 greeting test accepts "Good morning" (the time-aware reply); the small-talk
evaluator now accepts "a|b" intents, which it silently ignored before. In v7, the "reset_context" dialog
test expected both CS and SE electives at once for "what are the electives in semester 6?"; v7 asks
which degree first, so the test now expects the question and then "both".
`tests/benchmark800_results.csv` lists every benchmark question with the expected and the v4 answer.

## Handbook errors found while testing
Some SE Level 4 module descriptions in the PDF are copied from another degree: Software Evolution
(SE4022) is described as IS auditing, Formal Methods and Software Verification (SE4012) as IS
acquisition management, Software Quality (SE4032) as Business Process Re-engineering. The assistant
repeats the PDF faithfully, so these should be reported to the faculty.
