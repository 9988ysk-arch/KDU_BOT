"""
small_talk.py - rule-based small-talk understanding for the handbook assistant (no LLM).

classify(message) -> (intent | None, remaining_question | None, reply_text | None)

* Normalisation: lower case, emoji -> words, elongations collapsed ("hiiii" -> "hi",
  "thankss" -> "thanks"), chat abbreviations expanded ("thx", "tq", "gm"), filler words
  removed ("bro", "machan", "sir", "please" ...).
* Whole-message small talk: one intent ("good morning") or a chain of intents
  ("ok thanks bye") -> the most important intent answers.
* Small talk followed by a real question ("good morning! what is the faculty email?")
  -> a short courtesy prefix is returned together with the remaining question.
Intents cover English, Singlish (Sinhala in Latin letters) and a few Tamil words.
"""
from __future__ import annotations

import random
import re

EMOJI = {"👋": " hi ", "🙏": " thanks ", "👍": " ok ", "👌": " ok ", "😊": " nice ", "🙂": " nice ", "😀": " nice ",
         "😁": " nice ", "❤️": " nice ", "😂": " haha ", "🤣": " haha ", "😅": " haha ", "😢": " sad ", "😞": " sad ",
         "😭": " sad ", "😔": " sad ", "😡": " angry ", "🤔": " hmm ", "🎉": " yay ", "😴": " bored "}
ABBREV = {r"\bthx\b|\bthnx\b|\btnx\b|\bthanx\b|\btq\b|\bty\b|\btnq\b": "thanks", r"\bgm\b": "good morning",
          r"\bgn\b": "good night", r"\bu\b": "you", r"\br\b": "are", r"\bur\b": "your", r"\bpls\b|\bplz\b": "please",
          r"\bcya\b": "see you", r"\bttyl\b": "talk to you later", r"\bwassup\b|\bwazzup\b|\bwhatsup\b": "whats up",
          r"\bwhat's\b": "whats", r"\bdont\b": "don't", r"\bim\b": "i'm", r"\bi am\b": "i'm", r"\byou're\b": "you are",
          r"\bokk+\b|\bokie\b|\bokey\b|\bkk\b": "ok", r"\bhow's\b": "how is", r"\bthat's\b|\bthats\b": "that is",
          r"\bit's\b": "it is", r"\bwho's\b": "who is", r"\bthere's\b": "there is",
          r"\bgud\b|\bgd\b|\bgood+\b": "good", r"\bmrng\b|\bmrn\b|\bmornin\b|\bmorng\b|\bmorni\b": "morning",
          r"\bnite\b|\bnyt\b|\bnight+\b": "night", r"\bevng\b|\beve\b|\bevenin\b": "evening",
          r"\baftn\b|\baftanoon\b|\bafternun\b|\baftrnoon\b": "afternoon", r"\bdoin\b": "doing",
          r"\bgoin\b": "going", r"\bc you\b|\bc ya\b|\bsee ya\b": "see you", r"\bhru\b|\bhow ru\b": "how are you",
          r"\bwbu\b|\bhbu\b": "and you", r"\bthank youu+\b": "thank you", r"\bluv\b": "love",
          r"\bbye+\b": "bye", r"\bhw\b": "how", r"\bwat\b|\bwht\b": "what", r"\bplz+\b": "please"}
FILLER = r"\b(bro|bruh|machan|machang|malli|aiya|akka|sir|madam|miss|dear|friend|buddy|mate|bot|chatbot|assistant|" \
         r"there|again|everyone|guys|man|ya|yaa|please|so much|very much|a lot)\b"


def normalise(text: str) -> str:
    t = text.lower().strip().replace("\u2019", "'").replace("`", "'")
    for e, w in EMOJI.items():
        t = t.replace(e, w)
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)            # hiiii -> hi, sooo -> so
    t = re.sub(r"\bh+i+\b", "hi", t)                  # hii -> hi
    t = re.sub(r"\bhe+y+\b", "hey", t)
    t = re.sub(r"\b(hel+o+|hell?o+)\b", "hello", t)   # helloooo -> hello
    t = re.sub(r"\bthanks+\b", "thanks", t)
    t = re.sub(r"\b(ha){2,}h?\b|\bhah+a+\b", "haha", t)
    for pat, rep in ABBREV.items():
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()


def strip_punct(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[!?.,:;~*()\"]+", " ", t)).strip()


# intent, patterns (full-match after normalisation and filler removal), replies, show example buttons
INTENTS: list[tuple[str, list[str], list[str], bool]] = [
    ("greeting_time", [r"good (morning|afternoon|evening|day)", r"morning", r"suba (udesanak|sandyawak|dawasak|"
                       r"dahawalak)"], ["{time}! I'm the Student Handbook assistant. What would you like to know?"], True),
    ("good_night", [r"good ?night", r"night night", r"suba (rathriyak|rathrayak)", r"sweet dreams"],
     ["Good night! Come back any time you have a question about the handbook."], False),
    ("greeting", [r"hi", r"hello", r"hey", r"hey hey", r"hiya", r"howdy", r"yo", r"hola", r"greetings", r"hi hello",
                  r"ayubowan", r"aayubowan", r"ayubo", r"vanakkam", r"namaste", r"salaam", r"hey hello"],
     ["Hello! I'm the Student Handbook assistant. Ask me about modules, credits, grades, exams, staff or the "
      "degree programmes."], True),
    ("how_are_you", [r"(hi |hello |hey )?how are you( doing| today| doing today)?", r"how(s| is) it going",
                     r"how have you been", r"are you (ok|okay|fine|good|well)", r"how do you do",
                     r"(oyata )?kohomada( oyata)?", r"sapada", r"sapadda", r"hondin innawada", r"how you doing",
                     r"how is your day", r"hows your day", r"how is (your|ur) day going", r"how are things",
                     r"how are you and you", r"and you", r"you ok", r"you good", r"all good", r"how is life",
                     r"how are you today and you"],
     ["I'm doing well, thanks for asking! What would you like to know from the Student Handbook?"], False),
    ("whats_up", [r"whats up", r"sup", r"what are you doing", r"whats new", r"mokada wenne", r"mokada karanne",
                  r"what is up"], ["Just here, ready to answer questions about the Student Handbook. What do you need?"],
     True),
    ("identity", [r"who are you", r"what are you", r"whats your name", r"what is your name", r"your name",
                  r"introduce yourself", r"tell me about yourself", r"who am i (talking|chatting) (to|with)",
                  r"oya kawda", r"oyage nama mokakda", r"oyage nama"],
     ["I'm the Faculty of Computing Student Handbook assistant. I answer from the 2022 Student Handbook only, "
      "running locally on the CPU without an LLM."], True),
    ("creator", [r"who (made|created|built|developed|designed|programmed) (you|this|this bot|this system|this app)",
                 r"who is your (creator|developer|maker)", r"oyawa haduwe kawda"],
     ["I was built as a Faculty of Computing student project: a rule-based question answering system over the "
      "Student Handbook (pdfplumber + knowledge base + BM25F retrieval), with no LLM."], False),
    ("is_bot", [r"are you (a |an )?(human|person|robot|bot|real|ai|machine|computer|program)",
                r"are you (chat ?gpt|gpt|gemini|claude|an llm)", r"do you use (an |a )?(llm|ai|gpt|chat ?gpt|"
                r"neural network|gpu)", r"is this (ai|chat ?gpt|a bot|an llm)", r"am i talking to a (bot|human|person)"],
     ["I'm a bot, not a person, and not an LLM like ChatGPT. I'm a rule-based assistant that finds answers in the "
      "Student Handbook using a structured knowledge base and keyword search, running on the CPU."], False),
    ("bot_age", [r"how old are you", r"when were you (made|born|created)", r"whats your age"],
     ["I was created in 2026 for the Faculty of Computing. My knowledge comes from the 2022 Student Handbook."], False),
    ("bot_location", [r"where are you( from)?", r"where do you live", r"where are you located"],
     ["I run locally on this computer. The Faculty of Computing itself is at the KDU Southern Campus, "
      "Sooriyawewa."], False),
    ("affection", [r"i love you", r"do you love me", r"will you marry me", r"do you like me", r"be my (friend|girlfriend|"
                   r"boyfriend)", r"i like you"],
     ["That's kind of you! I'm just a handbook assistant, but I'm always happy to help with your questions."], False),
    ("bot_ability", [r"are you (smart|clever|intelligent)", r"can you think", r"do you have feelings",
                     r"are you (alive|conscious)"],
     ["I don't think or feel. I follow rules to find answers in the Student Handbook, which keeps me fast and "
      "accurate for handbook questions."], False),
    ("capability", [r"help", r"help me", r"i need help", r"what can you do", r"what can i ask( you)?",
                    r"what do you know", r"how do i use (this|you|this bot)", r"menu", r"options", r"commands",
                    r"what (questions|things) can i ask", r"give me (some )?examples?", r"examples?",
                    r"what do you do", r"how can you help( me)?", r"mata udaw karanna", r"udaw karanna",
                    r"mokakda karanna puluwan", r"mokada karanna puluwan", r"how does this work"],
     ["I can answer questions from the Student Handbook: modules and credits per semester, electives, grades and "
      "GPV, FGPA classes, exam rules and attendance, degree programmes and entry requirements, departments, staff "
      "emails and research areas. Try one of these:"], True),
    ("thanks", [r"thanks", r"thank you", r"thanks (for|4) (the|your) help", r"thank you for (your|the) help",
                r"many thanks", r"much appreciated", r"appreciate it", r"i appreciate it", r"isthuthi", r"sthuthi",
                r"stuti", r"bohoma isthuthi", r"nandri", r"thanks for (that|this|the info|the answer)",
                r"thank you for (that|this|the info|the answer)", r"cheers"],
     ["You're welcome! Anything else about the handbook?"], False),
    ("bye", [r"bye", r"goodbye", r"good bye", r"bye bye", r"see you( later| soon)?", r"talk to you later",
             r"gihin ennam", r"i have to go", r"i('m| am) leaving", r"thats all( for now)?", r"that is all( for now)?",
             r"nothing else", r"no more questions", r"later", r"catch you later", r"take care", r"see you later then", r"bye for now", r"gotta go",
             r"have a (nice|good|great) day", r"good day to you", r"ok bye", r"thanks bye", r"adios", r"ta ta",
             r"ba ?bye", r"signing off", r"ill be back", r"i will come back later"],
     ["Goodbye! Good luck with your studies."], False),
    ("praise", [r"good job", r"great", r"great job", r"awesome", r"nice", r"cool", r"well done", r"perfect",
                r"excellent", r"amazing", r"brilliant", r"fantastic", r"you are (amazing|great|awesome|helpful|"
                r"the best|smart|good)", r"(very |really )?helpful", r"nice bot", r"good bot", r"(this is|that is|that was|it was|that's) "
                r"(great|helpful|awesome|perfect|amazing|useful|correct|right|nice)", r"good answer", r"correct",
                r"thats? (right|correct)", r"exactly", r"spot on", r"you('re| are) (right|correct)", r"superb", r"supiri", r"niyamai",
                r"ela", r"patta", r"maru", r"wow", r"nice one", r"love it"],
     ["Glad that helped! Anything else?"], False),
    ("ack", [r"ok", r"okay", r"alright", r"all right", r"got it", r"i see", r"understood", r"hmm", r"hm", r"mm",
             r"fine", r"noted", r"sure", r"hari", r"hondai", r"ow", r"k", r"right", r"oh", r"oh ok", r"ah",
             r"ah ok", r"makes sense"],
     ["Okay! Ask me anything else about the Student Handbook."], False),
    ("apology", [r"sorry", r"my bad", r"oops", r"samawenna", r"sorry about that", r"my mistake"],
     ["No problem at all! What would you like to know?"], False),
    ("laughter", [r"haha", r"lol", r"lmao", r"rofl", r"hehe", r"xd"], ["😄 Glad to keep things light! "
                                                                       "Anything you'd like to know?"], False),
    ("insult", [r"(you are|you're|ur|this is|what a) (stupid|useless|dumb|bad|terrible|rubbish|trash|idiot|"
                r"waste)( bot)?", r"(stupid|useless|dumb|idiot) bot", r"you suck", r"your (stupid|useless|dumb|bad|terrible|rubbish|trash|"
                r"an idiot|a waste)( bot)?", r"(this|that) (sucks|is bad|is wrong|is not helpful|didnt help)",
                r"not helpful", r"(very |so )?bad (bot|answer)", r"stupid", r"useless", r"idiot",
                r"wtf( is this)?", r"what the hell", r"shut up", r"pissu( bot)?", r"booruwa"],
     ["Sorry I didn't get that right. Tell me what you were looking for and I'll do my best to find it in the "
      "handbook. You can also try one of these:"], True),
    ("emotion_negative", [r"i('m| am| feel)? ?(so |really |very )?(stressed|worried|sad|nervous|anxious|scared|"
                          r"tired|depressed|upset|confused about everything|frustrated|angry)( about [a-z ]+)?",
                          r"i (failed|have failed|flunked)( my| the| in)?( (a |an |one |two |\d )?[a-z]+){0,3}", r"sad", r"angry",
                          r"i('m| am) (going to|gonna) fail( [a-z ]+)?", r"i('m| am) (worried|afraid|scared) i('ll| will| might) fail( [a-z ]+)?",
                          r"(this|exams?|uni|university|life) (is|are) (so )?(hard|tough|difficult|stressful)",
                          r"i can'?t (do this|cope|handle this)", r"i('m| am) (lost|struggling|having a hard time)",
                          r"i feel (stressed|anxious|nervous|lost|sad|overwhelmed)", r"overwhelmed",
                          r"mata baya", r"mata amaruy", r"mata amaru", r"maha amaruy", r"i feel (bad|low|down)"],
     ["I'm sorry you're going through that. Exams can be stressful. The handbook has rules that may help: "
      "re-sitting a failed course, supplementary exams and medical/valid-reason excuses. Would you like to know "
      "about any of these?"], True),
    ("emotion_positive", [r"i (passed|got (a )?first class|got an a\+?|did it)( my| the)?( [a-z ]+)?!?",
                          r"i('m| am)? ?(so |really )?(happy|glad|excited)", r"yay", r"i got (my )?results?( and passed)?",
                          r"i (cleared|passed) (all|everything)( my)?( exams| modules)?", r"good news", r"yes!+", r"woohoo",
                          r"hurray|hooray"],
     ["Congratulations! 🎉 That's great to hear. Anything else I can help you with?"], False),
    ("nice_to_meet", [r"nice to meet you", r"pleased to meet you", r"glad to meet you"],
     ["Nice to meet you too! What would you like to know about the handbook?"], False),
    ("confused", [r"what", r"huh", r"eh", r"what do you mean", r"i (don't|do not) (understand|get it)",
                  r"i'm confused", r"confused", r"what\?+", r"meaning"],
     ["Sorry if that was unclear. You can ask me about a module (e.g. CS2082), a semester, grades, exam rules or "
      "staff. Here are some examples:"], True),
    ("test", [r"test( \d+)*", r"testing( \d+)*", r"test test", r"\d+( \d+)* testing", r"mic (check|test)", r"are you there", r"anyone there", r"is anyone there", r"hello\?",
              r"you there", r"ping"], ["Yes, I'm here and ready! Ask me anything about the Student Handbook."], True),
    ("bored", [r"i('m| am) bored", r"bored"], ["Maybe a quick handbook fact? For example, do you know the attendance "
                                                "rule for exams? Try one of these:"], True),
    ("joke", [r"tell me a joke", r"(say|tell me) something funny", r"(make|tell) me laugh", r"(a |another )?joke",
              r"tell (me )?(another|one more)( joke)?", r"do you know (any )?jokes?", r"be funny"],
     ["Why did the student bring a ladder to the exam? To reach the 3.70 FGPA for a First Class! 😄 "
      "Anything handbook-related I can help with?"], False),
    ("out_of_domain_time", [r"what time is it", r"whats the time( now)?", r"(what is|whats) (the )?(date|day)( today| now)?",
                            r"what day is (it|today)", r"(what is|whats) today'?s date", r"time now", r"date today",
                            r"what year is it"],
     ["I can't check the time or date, and I couldn't find that in the Student Handbook. I only answer questions from the handbook."], False),
    ("out_of_domain_general", [r"(what is|whats|how is) the weather( today)?", r"weather", r"whats the news",
                               r"who won [a-z ]+", r"play (some )?music", r"sing( a song)?",
                               r"(whats|what is) the (score|news|stock price|price of [a-z ]+)",
                               r"(tell me|what is) (the )?(cricket|football|match) (score|result)",
                               r"(write|do) my (homework|assignment|essay|code)", r"recommend (a )?(movie|song|book)"],
     ["That isn't in the Student Handbook, so I couldn't find an answer. I only answer questions about the Faculty "
      "of Computing Student Handbook. Try a handbook question:"], True),
]
# priority when several intents appear in one message ("ok thanks bye" -> bye)
# ---------------------------------------------------------------------------- v7 additions
def _extend(name: str, pats: list[str]) -> None:
    for i, (n, p, r, e) in enumerate(INTENTS):
        if n == name:
            INTENTS[i] = (n, p + pats, r, e)
            return
    raise KeyError(name)


_extend("greeting_time", [r"(good )?(evening|afternoon|noon)", r"good (morning|afternoon|evening) to you",
                          r"morning all", r"good mornings?", r"gud (morning|evening|afternoon)"])
_extend("greeting", [r"hi there", r"hello there", r"hey there", r"hey you", r"hi all", r"hello everyone",
                     r"hi everyone", r"heya", r"hai", r"helo", r"hlo", r"hy", r"hii+", r"good to see you",
                     r"long time no see", r"kohomada yaluwa", r"halo"])
_extend("how_are_you", [r"how are you doing today", r"how have you been", r"how do you do", r"hows everything",
                        r"how are u", r"kohomada ithin", r"oyata kohomada", r"sugada", r"saneepada"])
_extend("thanks", [r"istuti", r"bohoma istuti", r"isthuthi", r"bohoma isthuthi", r"stuti", r"nanri", r"romba nanri",
                   r"thank you so much", r"thanks a lot", r"thanks a ton", r"many thanks", r"much appreciated",
                   r"appreciate (it|that|your help)", r"thanks for (the|your) (answer|info|information|reply)",
                   r"that helped", r"that was helpful", r"you helped (me )?a lot", r"thank you very much", r"tysm"])
_extend("bye", [r"good bye for now", r"ok bye", r"ill go now", r"i('m| am) leaving", r"nice talking to you",
                r"gihin ennam", r"mama yanawa", r"poi varen"])
_extend("capability", [r"can you help( me)?", r"could you help( me)?", r"please help( me)?", r"help me please",
                       r"what do you know", r"what (questions|things) can (i|you) (ask|answer)",
                       r"what (is|are) (this|you) (for|used for)", r"how (do|can) i use (you|this)",
                       r"what (topics|subjects) do you (know|cover)", r"menu", r"options", r"start", r"get started"])
_extend("insult", [r"i hate you", r"you are (so )?(annoying|boring|slow|wrong again)", r"shut up", r"go away",
                   r"you know nothing", r"(worst|bad) bot( ever)?", r"pissu", r"moda"])
_extend("affection", [r"you are (cute|sweet|lovely|kind|nice)", r"i like you", r"you('re| are) my (best )?friend",
                      r"are you single", r"do you have a (girlfriend|boyfriend|partner)"])
_extend("ack", [r"ammo", r"aiyo", r"aney", r"ane", r"appo", r"sha", r"wow", r"oh", r"oh ok", r"ah", r"aah", r"i see now",
                r"ehema da", r"ehemada", r"ela", r"elakiri", r"hari", r"hari hari", r"ow", r"ow ow", r"noted",
                r"fine", r"k", r"kk", r"oki", r"okey", r"sure", r"makes sense", r"i understand", r"alright then"])
_extend("out_of_domain_time", [r"what time is it( now| there)?", r"(what is|whats) the time (now|today|there)",
                               r"(tell me|do you know) the (time|date)", r"(which|what) day is (it|today)( today)?"])
_extend("good_night", [r"(gn|good ?night|nighty night|nite)( all| everyone| guys| bot)?"])
_extend("praise", [r"(that|it|this) (was|is) (really |very |so )?(useful|helpful|great|good|nice|awesome|perfect|"
                   r"clear|correct)", r"(really |very )?(useful|helpful)( thanks?)?"])
_extend("test", [r"wake up", r"are you (awake|alive|working|online)", r"ping", r"you there"])
_extend("emotion_negative", [r"exam stress", r"(so )?stressed( out)?", r"i('m| am) (so )?(tired|exhausted|sleepy)",
                             r"i('m| am) (hungry|sick|not feeling well)", r"i have (an |my )?exams? (tomorrow|today|"
                             r"next week|soon)", r"i('m| am) (afraid|scared) of exams?", r"i (failed|messed up)"])
_extend("emotion_positive", [r"i got (a )?first class", r"yay( i passed)?", r"i passed( all)?( my)?( exams?)?",
                             r"i('m| am) (so )?(happy|excited)( today)?", r"best day( ever)?"])
_extend("out_of_domain_general", [r"what is \d+ ?[-+*/x] ?\d+", r"tell me a story", r"(what is|whats) the meaning "
                                  r"of life", r"(who is|whos) the (president|prime minister)( of [a-z ]+)?",
                                  r"(recommend|suggest) (a )?(movie|song|book|restaurant)",
                                  r"(what|which) (is the )?best (phone|laptop|movie)", r"play (a )?game",
                                  r"what should i eat", r"translate [a-z ]+", r"write (a )?(poem|song|essay)"])
_extend("bot_ability", [r"do you (sleep|eat|dream|get tired)", r"are you (a )?(boy|girl|male|female)( or (a )?"
                        r"(boy|girl|male|female))?", r"(what is|whats) your gender", r"do you have (a )?(family|"
                        r"friends|feelings)", r"(what is|whats) your favou?rite [a-z ]+", r"what do you like"])

INTENTS += [
    ("celebration", [r"happy birthday( to me)?", r"(it is|its|today is) my birthday", r"congrats", r"congratulations",
                     r"happy new year", r"merry christmas", r"happy (vesak|poson|deepavali|diwali|ramadan|eid|"
                     r"avurudu|new year|holidays?)", r"suba aluth avuruddak", r"suba upandinayak"],
     ["Thank you, and happy celebrations to you too! 🎉 If you need anything from the Student Handbook, "
      "just ask."], False),
    ("good_luck", [r"(good luck|all the best|best of luck)( (to|for) (you|me|us))?", r"wish me (good )?luck",
                   r"pray for me", r"suba pathum"],
     ["Good luck! You've got this. If it helps, I can check exam rules: attendance, re-sits, supplementary "
      "exams or grades."], True),
    ("user_fine", [r"i('m| am) (fine|good|great|ok|okay|doing (good|well|fine)|all good|well)( and you)?",
                   r"(fine|good|doing well|doing good) (and you|what about you)", r"not bad", r"same here", r"same", r"me too", r"pretty good",
                   r"mama hondin", r"hondin", r"hodin innawa", r"hondai"],
     ["Glad to hear it! What would you like to know from the Student Handbook?"], True),
    ("language", [r"(what|which) languages? (do|can) you (speak|understand|know)", r"can you (speak|understand) "
                  r"(sinhala|tamil|english|singlish)", r"(do you|can you) (know|speak) sinhala", r"sinhalen "
                  r"(kiyanna|kiyapan|katha karanna|puluwanda)", r"sinhala (puluwanda|danne nadda)",
                  r"tamil (theriyuma|pesuveengala)"],
     ["I understand English and Singlish (Sinhala typed in English letters, e.g. \"dean kawda?\" or "
      "\"attendance kiyada?\"). I can't read Sinhala or Tamil script yet, and my answers are in English."], True),
]
# ---------------------------------------------------------------------------- v10 additions
_extend("out_of_domain_general", [r"(what is|whats|which is) the capital (city )?of [a-z ]+", r"capital of [a-z ]+",
                                  r"who (invented|discovered|wrote|painted|founded|created) (?!.*\b(handbook|message|dean|faculty|kdu|module|"
                                  r"by-?laws?|rules?|curriculum)\b)[a-z ]+",
                                  r"(what is|whats) the population of [a-z ]+", r"how (far|tall|big|old|deep) is [a-z ]+",
                                  r"who won the [a-z ]+", r"(who is|whos) (elon musk|messi|ronaldo|taylor swift)",
                                  r"(what is|whats) (the )?(square root|factorial) of \d+"])
_extend("confused", [r"i (still )?(don't|do not|dont) (understand|get (it|that|you))", r"(that|it) (doesnt|doesn't|does not) "
                     r"make sense", r"(can you|could you) (explain|say) (it|that) (simply|more simply|again simply)",
                     r"too (complicated|confusing|long)", r"(i('m| am) )?(still )?(confused|lost)", r"what does that mean",
                     r"in simple (words|terms)", r"simpler please", r"eli5"])
INTENTS += [
    ("uncertain", [r"maybe", r"perhaps", r"i (don't|do not|dont) know", r"idk", r"dunno", r"not sure", r"i('m| am) not sure",
                   r"no idea", r"i have no idea", r"never ?mind", r"nvm", r"nothing", r"whatever", r"i guess",
                   r"possibly", r"not really sure", r"mata danne na", r"danne na", r"mama danne na"],
     ["No problem! If you're not sure what to ask, these are popular questions - or just tell me what you're "
      "worried about (a module, an exam, your grades) and I'll find the rule."], True),
    ("compare_ai", [r"are you (smarter|better|faster|more accurate|cleverer|dumber|worse) than (chat ?gpt|gpt|gemini|"
                    r"claude|siri|alexa|google|copilot|other bots?|an llm|ai)", r"(chat ?gpt|gemini|claude|google) is "
                    r"(better|smarter) (than you)?", r"why not (use )?(chat ?gpt|gemini|google)",
                    r"(is|are) (chat ?gpt|gemini) better( than you)?", r"can you replace (chat ?gpt|google)"],
     ["I'm much smaller than ChatGPT-style models and I can't chat about everything - but for the Student Handbook "
      "I answer from the actual rules and tell you the page, so I won't invent a rule that isn't there. Try me:"],
     True),
    ("wait", [r"wait( a (sec|second|moment|minute|min))?", r"hold on( a (sec|second|moment|minute|min))?", r"one (sec|second|moment|minute|min)", r"give me a (sec|second|moment|minute|min)",
              r"brb", r"be right back", r"let me think", r"hmm let me think", r"just a (sec|second|moment|minute)",
              r"poddak inna", r"inna"],
     ["Sure, take your time. I'll be here when you're ready."], False),
]
PRIORITY_EXTRA = ["celebration", "good_luck", "user_fine", "language"]

PRIORITY = ["bye", "good_night", "thanks", "emotion_negative", "emotion_positive", "insult", "capability", "identity",
            "creator", "is_bot", "how_are_you", "whats_up", "greeting_time", "greeting", "nice_to_meet", "apology",
            "confused", "test", "bored", "joke", "affection", "bot_ability", "bot_age", "bot_location",
            "out_of_domain_time", "out_of_domain_general", "laughter", "praise", "ack"]
PRIORITY = ["celebration", "good_luck"] + PRIORITY[:6] + ["language"] + PRIORITY[6:] + ["user_fine"]
PRIORITY = PRIORITY[:-1] + ["compare_ai", "wait", "uncertain"] + PRIORITY[-1:]
# intents that may open a message that continues with a real question
PREFIX_INTENTS = {"greeting_time", "greeting", "how_are_you", "thanks", "praise", "ack", "apology", "nice_to_meet",
                  "celebration", "good_luck", "user_fine"}

_COMPILED = [(name, re.compile(r"^(?:" + "|".join(pats) + r")$"), replies, ex) for name, pats, replies, ex in INTENTS]
_BY_NAME = {name: (rx, replies, ex) for name, rx, replies, ex in _COMPILED}


def _match(t: str) -> str | None:
    for name, rx, _, _ in _COMPILED:
        if rx.match(t):
            return name
    return None


def _time_word(t: str) -> str:
    if re.search(r"afternoon", t):
        return "Good afternoon"
    if re.search(r"evening|sandyawak", t):
        return "Good evening"
    if re.search(r"\bday\b|dawasak|dahawalak", t):
        return "Good day"
    return "Good morning"


def reply_for(intent: str, t: str = "") -> tuple[str, bool]:
    _, replies, ex = _BY_NAME[intent]
    return random.choice(replies).replace("{time}", _time_word(t)), ex


def classify(message: str) -> tuple[str | None, str | None, str | None]:
    """Returns (intent, remaining_question, courtesy_prefix)."""
    t = normalise(message)
    plain = strip_punct(t)
    nofill = re.sub(r"\s+", " ", re.sub(FILLER, " ", plain)).strip() or plain
    # 1) the whole message is one small-talk intent
    for cand in (plain, nofill):
        name = _match(cand)
        if name:
            return name, None, None
    # 2) a chain of small-talk intents: "ok thanks bye", "hi how are you", "cool thanks"
    words = nofill.split()
    found, i = [], 0
    while i < len(words):
        for j in range(len(words), i, -1):
            name = _match(" ".join(words[i:j]))
            if name:
                found.append(name)
                i = j
                break
        else:
            found = []
            break
    if found:
        return min(found, key=PRIORITY.index), None, None
    # 3) small talk followed by a real question: "good morning! what is the faculty email?"
    raw = message.strip()
    m = re.match(r"^\s*([^?]*?)([,.!:;]+|\s+-\s+|\s+(?=(what|who|which|when|where|how|is|are|can|could|do|does|"
                 r"tell|list|give|show|and)\b))\s*(.+)$", raw, re.I)
    if m:
        head = strip_punct(normalise(m.group(1)))
        head_nf = re.sub(r"\s+", " ", re.sub(FILLER, " ", head)).strip() or head
        rest = m.group(4).strip()
        name = _match(head) or _match(head_nf)
        if not name:  # chained prefix: "hi, thanks, ..."
            ws = head_nf.split()
            name = _match(ws[0]) if ws and all(_match(w) for w in ws) else None
        if name in PREFIX_INTENTS and len(rest.split()) >= 1 and re.search(r"[a-z0-9]", rest.lower()):
            rest = re.sub(r"^(and|also|so|then)\s+", "", rest, flags=re.I)
            prefix = {"greeting_time": _time_word(head) + "!", "greeting": "Hi!", "thanks": "You're welcome!",
                      "how_are_you": "I'm doing well, thanks!", "nice_to_meet": "Nice to meet you too!",
                      "celebration": "Thank you! 🎉", "good_luck": "Good luck!", "user_fine": "Glad to hear it!"}.get(name, "")
            return None, rest, prefix
    return None, None, None

# ---------------------------------------------------------------------------- keyword fallback
# Used only AFTER the handbook search found nothing: short messages are scored by intent keywords, so a new
# wording such as "you are so dumb" or "tell me a funny joke" still gets a small-talk reply.
KEYWORDS: list[tuple[str, str]] = [
    ("insult", r"\b(stupid|dumb|useless|idiot|trash|rubbish|hate you|worst|pathetic|shit|crap|pissu|moda|suck|sucks)\b"),
    ("joke", r"\b(joke|jokes|funny|laugh|humou?r|meme)\b"),
    ("bored", r"\b(bored|boring)\b"),
    ("out_of_domain_time", r"\b(what|whats|tell me)\b.*\b(time|date|day|year)\b.*\b(now|today|is it)\b|\bwhat time\b"),
    ("out_of_domain_general", r"\b(weather|rain|news|cricket|football|movie|song|music|recipe|stock|bitcoin|"
                              r"president|prime minister|game|story|poem|capital of|invented|discovered|population|"
                              r"planet|world cup|olympics|ipl|celebrity|actor|actress)\b"),
    ("compare_ai", r"\b(chat ?gpt|gemini|copilot|siri|alexa)\b"),
    ("uncertain", r"^(i )?(really )?(don'?t|do not) know\b|\bnot sure\b|\bno idea\b"),
    ("apology", r"\b(sorry|apolog\w*|my (fault|bad|mistake)|forgive)\b"),
    ("test", r"\b(any ?(one|body)|some ?(one|body)) (there|here|home|around)\b|\b(you|u) (there|alive|awake)\b"),
    ("nice_to_meet", r"\bnice (to )?meet\b|\bpleasure (to )?meet\b"),
    ("affection", r"\b(love|like|adore|miss) (you|u|this bot|this chatbot|ya)\b|\b(cute|sweet)\b"),
    ("good_night", r"\b(gn|good ?night)\b"),
    ("good_luck", r"\b(good luck|all the best|best of luck|wish me luck)\b"),
    ("praise", r"\b(that|it|this) (was|is) (really |very |so )?(useful|helpful|great|good|nice|perfect)\b|\b(great|awesome|amazing|excellent|brilliant|helpful|useful|best|perfect|superb|fantastic|good) "
               r"(bot|job|work|answer|answers|assistant|chatbot|one)\b|\b(you|u) (are|r) (the )?(best|great|amazing|"
               r"awesome|smart|genius)\b|\blove this (bot|chatbot|app)\b"),
    ("thanks", r"\b(thanks|thank|thx|tnx|istuti|sthuthi|isthuthi|stuti|nanri)\b"),
    ("bye", r"\b(bye|goodbye|cya|see (you|u|ya)|later|leaving|gotta go)\b"),
    ("user_fine", r"\b(i am|i'm|im|i'm doing|doing|feeling) (great|good|fine|well|ok|okay|alright|awesome)\b"),
    ("emotion_negative", r"\b(stress\w*|worried|anxious|nervous|scared|afraid|sad|depressed|upset|tired|exhausted|"
                         r"overwhelmed|fail(ed|ing)?|struggling)\b"),
    ("emotion_positive", r"\b(passed|cleared|got (a |my )?first class|happy|excited|yay|woohoo)\b"),
    ("how_are_you", r"\bhow (are|r) (you|u|ya)\b|\bkohomada\b"),
    ("greeting", r"^(hi|hello|hey|yo|hola|ayubowan|vanakkam)\b"),
    ("identity", r"\b(your name|who are (you|u)|what are (you|u))\b"),
    ("creator", r"\bwho (made|built|created|developed|designed|coded|programmed)\b"),
    ("capability", r"\b(what can (you|u)|help me|can (you|u) help|how (do|can) i use)\b"),
    ("laughter", r"\b(ha){2,}|\blol\b|\blmao\b|\bhehe\b"),
]
_KW = [(n, re.compile(p)) for n, p in KEYWORDS]


DOMAIN = re.compile(r"\b(exams?|examinations?|modules?|credits?|grades?|gpa|fgpa|sgpa|gpv|pass mark|attendance|"
                    r"semesters?|degrees?|re-?sit\w*|repeat\w*|marks?|lectur\w*|courses?|subjects?|departments?|dean|"
                    r"registrar|electives?|compulsory|class|hostels?|fees?|library|scholarships?|medical|leave|"
                    r"transfer\w*|kdu|faculty|staff|email|phone|contact|students?|programmes?|programs?|military|"
                    r"civil|height|rules?|requirements?|results?|supplementary|deadline|registration|level|year \d)\b")


def fallback(message: str) -> str | None:
    t = strip_punct(normalise(message))
    if not t or len(t.split()) > 8:
        return None
    # a feeling about studies ("I'm stressed about my results") is small talk even with a domain word in it
    if re.match(r"(i'?m|i am|im|i feel|i'?m feeling|feeling|i('m| am) so)\b", t) and \
            re.search(r"\b(stress\w*|worried|anxious|nervous|scared|afraid|sad|upset|depressed|overwhelmed|panick\w*)\b", t):
        return "emotion_negative"
    if DOMAIN.search(t):
        return None
    for name, rx in _KW:
        if rx.search(t):
            return name
    return None

