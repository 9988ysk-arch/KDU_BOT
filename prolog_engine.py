"""
prolog_engine.py - a small, dependency-free Prolog interpreter (pure Python, CPU only).

It runs the same .pl files that SWI-Prolog runs (handbook_facts.pl + handbook_rules.pl), so the logic
layer works without installing anything. If SWI-Prolog and pyswip are installed, logic_layer.py can use
them instead; the answers are the same.

Supported: facts and rules (Horn clauses), unification, backtracking, operators (:- , ; -> \\+ = \\= == \\==
is =:= =\\= < > =< >= + - * / // mod ** ^), lists [H|T], quoted atoms, strings ("..." read as atoms),
negation as failure, if-then-else, findall/3, forall/2, aggregate_all/3 (count, sum, max, min, bag, set),
between/3, length/2, msort/2, sort/2, sum_list/2, max_list/2, min_list/2, call/N, atom/number/var tests,
atom_concat/3, atom_number/2, format_atom/3 (simple ~w substitution).
Not supported (not needed by the handbook rules): cut (!), assert/retract, exceptions, modules.
"""
from __future__ import annotations

import itertools
import re
import sys
from typing import Any, Iterator

sys.setrecursionlimit(20000)


# ----------------------------------------------------------------------------- terms
class Var:
    __slots__ = ("name", "ref")
    _ids = itertools.count()

    def __init__(self, name: str = "_"):
        self.name = name
        self.ref: Any = None

    def __repr__(self) -> str:
        return f"_{self.name}"


class Struct:
    __slots__ = ("functor", "args")

    def __init__(self, functor: str, args: tuple):
        self.functor = functor
        self.args = args

    def __repr__(self) -> str:
        return term_to_str(self)


NIL = "[]"


def mklist(items: list, tail: Any = NIL) -> Any:
    out = tail
    for x in reversed(items):
        out = Struct(".", (x, out))
    return out


def deref(t: Any) -> Any:
    while isinstance(t, Var) and t.ref is not None:
        t = t.ref
    return t


def resolve(t: Any) -> Any:
    """Fully dereferenced copy of a term (variables that are still free stay as Var)."""
    t = deref(t)
    if isinstance(t, Struct):
        return Struct(t.functor, tuple(resolve(a) for a in t.args))
    return t


def to_python(t: Any) -> Any:
    """Prolog term -> Python value: lists -> list, atoms -> str, numbers -> number, other -> str."""
    t = deref(t)
    if isinstance(t, Struct) and t.functor == "." and len(t.args) == 2:
        out = []
        while isinstance(t, Struct) and t.functor == "." and len(t.args) == 2:
            out.append(to_python(t.args[0]))
            t = deref(t.args[1])
        return out
    if t == NIL:
        return []
    if isinstance(t, Struct):
        if t.functor == "-" and len(t.args) == 2:
            return (to_python(t.args[0]), to_python(t.args[1]))
        return term_to_str(t)
    if isinstance(t, Var):
        return None
    return t


def from_python(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return mklist([from_python(x) for x in v])
    return v


def term_to_str(t: Any) -> str:
    t = deref(t)
    if isinstance(t, Var):
        return f"_{t.name}"
    if isinstance(t, Struct):
        if t.functor == "." and len(t.args) == 2:
            return "[" + ", ".join(term_to_str(x) for x in to_python_list(t)) + "]"
        if len(t.args) == 2 and t.functor in BINARY_OPS:
            return f"{term_to_str(t.args[0])}{t.functor}{term_to_str(t.args[1])}"
        return f"{t.functor}(" + ", ".join(term_to_str(a) for a in t.args) + ")"
    if isinstance(t, float):
        return f"{t:.2f}".rstrip("0").rstrip(".") if t != int(t) else f"{t:.1f}"
    return str(t)


def to_python_list(t: Any) -> list:
    out = []
    t = deref(t)
    while isinstance(t, Struct) and t.functor == "." and len(t.args) == 2:
        out.append(t.args[0])
        t = deref(t.args[1])
    return out


# ----------------------------------------------------------------------------- reader
OPS: dict[str, tuple[int, str]] = {  # infix operators: name -> (priority, type)
    ":-": (1200, "xfx"), "-->": (1200, "xfx"), ";": (1100, "xfy"), "|": (1100, "xfy"), "->": (1050, "xfy"),
    ",": (1000, "xfy"), "=": (700, "xfx"), "\\=": (700, "xfx"), "==": (700, "xfx"), "\\==": (700, "xfx"),
    "is": (700, "xfx"), "=:=": (700, "xfx"), "=\\=": (700, "xfx"), "<": (700, "xfx"), ">": (700, "xfx"),
    "=<": (700, "xfx"), ">=": (700, "xfx"), "@<": (700, "xfx"), "@>": (700, "xfx"), "=..": (700, "xfx"),
    "+": (500, "yfx"), "-": (500, "yfx"), "*": (400, "yfx"), "/": (400, "yfx"), "//": (400, "yfx"),
    "mod": (400, "yfx"), "**": (200, "xfx"), "^": (200, "xfy"),
}
PREFIX: dict[str, tuple[int, str]] = {":-": (1200, "fx"), "dynamic": (1150, "fx"), "discontiguous": (1150, "fx"), "\\+": (900, "fy"), "-": (200, "fy"), "+": (200, "fy")}
BINARY_OPS = set(OPS)

TOKEN_RE = re.compile(r"""
    (?P<ws>\s+|%[^\n]*)
  | (?P<float>\d+\.\d+(?:[eE][+-]?\d+)?)
  | (?P<int>\d+)
  | (?P<var>[A-Z_][A-Za-z0-9_]*)
  | (?P<atom>[a-z][A-Za-z0-9_]*)
  | (?P<qatom>'(?:[^'\\]|\\.|'')*')
  | (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<punct>[()\[\],|])
  | (?P<end>\.(?=\s|%|$))
  | (?P<sym>[+\-*/\\^<>=~:.?@#&$]+|!|;)
""", re.X)


def tokenize(text: str) -> list[tuple[str, str, bool]]:
    """-> list of (kind, value, followed_by_open_paren)"""
    toks: list[tuple[str, str, bool]] = []
    pos = 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise SyntaxError(f"Prolog syntax error near: {text[pos:pos + 30]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        val = m.group(kind)
        if kind == "qatom":
            val, kind = val[1:-1].replace("''", "'").replace("\\'", "'").replace("\\n", "\n"), "atom"
        elif kind == "str":
            val, kind = val[1:-1].replace('\\"', '"').replace("\\n", "\n"), "atom"
        elif kind == "sym":
            kind = "atom"
        toks.append((kind, val, pos < len(text) and text[pos] == "("))
    return toks


class Parser:
    def __init__(self, tokens: list):
        self.t = tokens
        self.i = 0
        self.vars: dict[str, Var] = {}

    def peek(self) -> tuple[str, str, bool] | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self) -> tuple[str, str, bool]:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, val: str) -> None:
        tok = self.take()
        if tok[1] != val:
            raise SyntaxError(f"expected {val!r}, got {tok[1]!r}")

    def clauses(self) -> Iterator[Any]:
        while self.peek() is not None:
            self.vars = {}
            term = self.parse(1200)
            tok = self.take()
            if tok[0] != "end":
                raise SyntaxError(f"expected '.', got {tok[1]!r}")
            yield term

    def parse(self, maxp: int) -> Any:
        left, lp = self.primary(maxp)
        while True:
            tok = self.peek()
            if tok is None or tok[0] == "end":
                break
            name = tok[1]
            if tok[0] not in ("atom", "punct") or name not in OPS or name in ("(", ")", "[", "]"):
                break
            if name == "|" and maxp < 1100:
                break
            p, typ = OPS[name]
            if p > maxp:
                break
            left_max = p - 1 if typ[0] == "x" else p
            if lp > left_max:
                break
            self.take()
            right_max = p - 1 if typ[2] == "x" else p
            right = self.parse(right_max)
            left, lp = Struct(";" if name == "|" else name, (left, right)), p
        return left

    def primary(self, maxp: int) -> tuple[Any, int]:
        kind, val, paren = self.take()
        if kind == "int":
            return int(val), 0
        if kind == "float":
            return float(val), 0
        if kind == "var":
            if val == "_":
                return Var("_"), 0
            if val not in self.vars:
                self.vars[val] = Var(val)
            return self.vars[val], 0
        if kind == "punct":
            if val == "(":
                t = self.parse(1200)
                self.expect(")")
                return t, 0
            if val == "[":
                if self.peek() and self.peek()[1] == "]":
                    self.take()
                    return NIL, 0
                items = [self.parse(999)]
                while self.peek()[1] == ",":
                    self.take()
                    items.append(self.parse(999))
                tail = NIL
                if self.peek()[1] == "|":
                    self.take()
                    tail = self.parse(999)
                self.expect("]")
                return mklist(items, tail), 0
            raise SyntaxError(f"unexpected {val!r}")
        # atom
        if paren:
            self.take()  # (
            args = [self.parse(999)]
            while self.peek()[1] == ",":
                self.take()
                args.append(self.parse(999))
            self.expect(")")
            return Struct(val, tuple(args)), 0
        if val == "-" and self.peek() and self.peek()[0] in ("int", "float"):
            k, v, _ = self.take()
            return (-int(v) if k == "int" else -float(v)), 0
        if val in PREFIX and self.peek() and self.peek()[0] != "end" and \
                not (self.peek()[0] in ("atom", "punct") and self.peek()[1] in OPS and self.peek()[1] not in PREFIX
                     and self.peek()[1] != "("):
            p, typ = PREFIX[val]
            if p > maxp:
                p = 999
            arg = self.parse(p - 1 if typ == "fx" else p)
            return Struct(val, (arg,)), p
        return val, (OPS[val][0] if val in OPS else 0)


def read_terms(text: str) -> list[Any]:
    return list(Parser(tokenize(text)).clauses())


# ----------------------------------------------------------------------------- engine
class PrologError(Exception):
    pass


class Prolog:
    PRELUDE = r"""
    member(X, [X|_]).
    member(X, [_|T]) :- member(X, T).
    append([], L, L).
    append([H|T], L, [H|R]) :- append(T, L, R).
    nth1(N, L, X) :- var(N), nth1_(L, X, 1, N).
    nth1(N, L, X) :- integer(N), N >= 1, nth1_i(N, L, X).
    nth1_([X|_], X, I, I).
    nth1_([_|T], X, I0, I) :- I1 is I0 + 1, nth1_(T, X, I1, I).
    nth1_i(1, [X|_], X).
    nth1_i(N, [_|T], X) :- N > 1, N1 is N - 1, nth1_i(N1, T, X).
    last([X], X).
    last([_|T], X) :- last(T, X).
    reverse(L, R) :- rev_(L, [], R).
    rev_([], A, A).
    rev_([H|T], A, R) :- rev_(T, [H|A], R).
    include(_, [], []).
    include(P, [H|T], R) :- ( call(P, H) -> R = [H|R1] ; R = R1 ), include(P, T, R1).
    exclude(_, [], []).
    exclude(P, [H|T], R) :- ( call(P, H) -> R = R1 ; R = [H|R1] ), exclude(P, T, R1).
    maplist(_, []).
    maplist(P, [A|As]) :- call(P, A), maplist(P, As).
    maplist(_, [], []).
    maplist(P, [A|As], [B|Bs]) :- call(P, A, B), maplist(P, As, Bs).
    count_in(_, [], 0).
    count_in(X, [Y|T], N) :- count_in(X, T, N0), ( X == Y -> N is N0 + 1 ; N = N0 ).
    """

    def __init__(self) -> None:
        self.db: dict[tuple[str, int], list[tuple[Any, Any]]] = {}
        self.trail: list[Var] = []
        self.consult_text(self.PRELUDE)

    # -- loading
    def consult_text(self, text: str) -> int:
        n = 0
        for term in read_terms(text):
            if isinstance(term, Struct) and term.functor == ":-" and len(term.args) == 1:
                continue  # directives (:- dynamic ...) are ignored
            if isinstance(term, Struct) and term.functor == ":-" and len(term.args) == 2:
                head, body = term.args
            else:
                head, body = term, "true"
            key = (head.functor, len(head.args)) if isinstance(head, Struct) else (head, 0)
            self.db.setdefault(key, []).append((head, body))
            n += 1
        return n

    def consult(self, path: str) -> int:
        with open(path, encoding="utf-8") as f:
            return self.consult_text(f.read())

    # -- unification
    def bind(self, v: Var, t: Any) -> None:
        v.ref = t
        self.trail.append(v)

    def undo(self, mark: int) -> None:
        while len(self.trail) > mark:
            self.trail.pop().ref = None

    def unify(self, a: Any, b: Any) -> bool:
        a, b = deref(a), deref(b)
        if a is b:
            return True
        if isinstance(a, Var):
            self.bind(a, b)
            return True
        if isinstance(b, Var):
            self.bind(b, a)
            return True
        if isinstance(a, Struct) and isinstance(b, Struct):
            if a.functor != b.functor or len(a.args) != len(b.args):
                return False
            return all(self.unify(x, y) for x, y in zip(a.args, b.args))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return type(a) is type(b) and a == b
        return a == b

    @staticmethod
    def rename(t: Any, mapping: dict) -> Any:
        if isinstance(t, Var):
            if t.ref is not None:
                return Prolog.rename(deref(t), mapping)
            if t not in mapping:
                mapping[t] = Var(t.name)
            return mapping[t]
        if isinstance(t, Struct):
            return Struct(t.functor, tuple(Prolog.rename(a, mapping) for a in t.args))
        return t

    # -- arithmetic
    def eval(self, t: Any) -> int | float:
        t = deref(t)
        if isinstance(t, (int, float)):
            return t
        if isinstance(t, Var):
            raise PrologError("arguments are not sufficiently instantiated")
        if isinstance(t, Struct):
            a = [self.eval(x) for x in t.args]
            f = t.functor
            if len(a) == 2:
                x, y = a
                if f == "+": return x + y
                if f == "-": return x - y
                if f == "*": return x * y
                if f == "/":
                    r = x / y
                    return int(r) if isinstance(x, int) and isinstance(y, int) and r == int(r) else r
                if f == "//": return int(x // y)
                if f == "mod": return x % y
                if f in ("**", "^"): return x ** y
                if f == "min": return min(x, y)
                if f == "max": return max(x, y)
            if len(a) == 1:
                x = a[0]
                if f == "-": return -x
                if f == "+": return x
                if f == "abs": return abs(x)
                if f == "round": return int(round(x))
                if f == "float": return float(x)
                if f == "integer": return int(round(x))
                if f == "truncate": return int(x)
                if f == "floor": return int(x // 1)
        if t == "pi":
            return 3.141592653589793
        raise PrologError(f"not a number: {term_to_str(t)}")

    # -- solving
    def solve(self, goal: Any, depth: int = 0) -> Iterator[None]:
        if depth > 6000:
            raise PrologError("depth limit exceeded")
        goal = deref(goal)
        if isinstance(goal, Var):
            raise PrologError("unbound goal")
        name, args = (goal.functor, goal.args) if isinstance(goal, Struct) else (goal, ())
        n = len(args)

        # control constructs
        if name == "true" and n == 0:
            yield
            return
        if name in ("fail", "false") and n == 0:
            return
        if name == "," and n == 2:
            for _ in self.solve(args[0], depth + 1):
                yield from self.solve(args[1], depth + 1)
            return
        if name == ";" and n == 2:
            left = deref(args[0])
            if isinstance(left, Struct) and left.functor == "->" and len(left.args) == 2:
                mark = len(self.trail)
                gen = self.solve(left.args[0], depth + 1)
                try:
                    next(gen)
                    found = True
                except StopIteration:
                    found = False
                if found:
                    yield from self.solve(left.args[1], depth + 1)
                    gen.close()
                    return
                self.undo(mark)
                yield from self.solve(args[1], depth + 1)
                return
            mark = len(self.trail)
            yield from self.solve(left, depth + 1)
            self.undo(mark)
            yield from self.solve(args[1], depth + 1)
            return
        if name == "->" and n == 2:
            gen = self.solve(args[0], depth + 1)
            try:
                next(gen)
            except StopIteration:
                return
            yield from self.solve(args[1], depth + 1)
            gen.close()
            return
        if name == "\\+" and n == 1:
            mark = len(self.trail)
            ok = any(True for _ in self.solve(args[0], depth + 1))
            self.undo(mark)
            if not ok:
                yield
            return
        if name == "call" and n >= 1:
            g = deref(args[0])
            if n > 1:
                extra = args[1:]
                g = Struct(g.functor, g.args + extra) if isinstance(g, Struct) else Struct(g, extra)
            yield from self.solve(g, depth + 1)
            return
        if name == "forall" and n == 2:
            yield from self.solve(Struct("\\+", (Struct(",", (args[0], Struct("\\+", (args[1],)))),)), depth + 1)
            return

        builtin = getattr(self, f"b_{BUILTIN_NAMES.get((name, n), '')}", None) if (name, n) in BUILTIN_NAMES \
            else None
        if builtin is not None:
            mark = len(self.trail)
            for ok in builtin(*args):
                if ok:
                    yield
                self.undo(mark)
            return

        clauses = self.db.get((name, n))
        if clauses is None:
            raise PrologError(f"unknown procedure {name}/{n}")
        for head, body in clauses:
            mark = len(self.trail)
            mapping: dict = {}
            h = self.rename(head, mapping)
            if self.unify(h, goal):
                b = self.rename(body, mapping)
                yield from self.solve(b, depth + 1)
            self.undo(mark)

    # -- builtins (each yields True per solution; the caller undoes bindings)
    def b_unify(self, a, b):
        yield self.unify(a, b)

    def b_not_unify(self, a, b):
        mark = len(self.trail)
        ok = self.unify(a, b)
        self.undo(mark)
        yield not ok

    def b_eq(self, a, b):
        yield term_to_str(resolve(a)) == term_to_str(resolve(b)) and _same_vars(a, b)

    def b_neq(self, a, b):
        yield not (term_to_str(resolve(a)) == term_to_str(resolve(b)) and _same_vars(a, b))

    def b_is(self, a, b):
        yield self.unify(a, self.eval(b))

    def b_lt(self, a, b): yield self.eval(a) < self.eval(b)
    def b_gt(self, a, b): yield self.eval(a) > self.eval(b)
    def b_le(self, a, b): yield self.eval(a) <= self.eval(b)
    def b_ge(self, a, b): yield self.eval(a) >= self.eval(b)
    def b_aeq(self, a, b): yield self.eval(a) == self.eval(b)
    def b_aneq(self, a, b): yield self.eval(a) != self.eval(b)

    def b_findall(self, template, goal, out):
        results = []
        mark = len(self.trail)
        for _ in self.solve(goal):
            results.append(resolve(template))
        self.undo(mark)
        yield self.unify(out, mklist(results))

    def b_aggregate_all(self, spec, goal, out):
        spec = deref(spec)
        mark = len(self.trail)
        if spec == "count":
            n = sum(1 for _ in self.solve(goal))
            self.undo(mark)
            yield self.unify(out, n)
            return
        if not isinstance(spec, Struct) or len(spec.args) != 1:
            raise PrologError("aggregate_all: unsupported spec")
        vals = []
        for _ in self.solve(goal):
            vals.append(resolve(spec.args[0]))
        self.undo(mark)
        kind = spec.functor
        if kind == "sum":
            yield self.unify(out, sum(self.eval(v) for v in vals))
        elif kind == "max":
            yield bool(vals) and self.unify(out, max(self.eval(v) for v in vals))
        elif kind == "min":
            yield bool(vals) and self.unify(out, min(self.eval(v) for v in vals))
        elif kind == "bag":
            yield self.unify(out, mklist(vals))
        elif kind == "set":
            uniq = {term_to_str(v): v for v in vals}
            yield self.unify(out, mklist([uniq[k] for k in sorted(uniq)]))
        else:
            raise PrologError(f"aggregate_all: unsupported {kind}")

    def b_between(self, lo, hi, x):
        lo, hi, xv = self.eval(lo), self.eval(hi), deref(x)
        if isinstance(xv, (int, float)):
            yield lo <= xv <= hi
            return
        for i in range(lo, hi + 1):
            yield self.unify(x, i)

    def b_length(self, lst, n):
        items = to_python_list(lst)
        tail = deref(lst)
        while isinstance(tail, Struct) and tail.functor == ".":
            tail = deref(tail.args[1])
        if tail == NIL:
            yield self.unify(n, len(items))
        else:
            k = deref(n)
            if not isinstance(k, int):
                raise PrologError("length/2: open list with unbound length")
            yield self.unify(lst, mklist([Var() for _ in range(k)]))

    def b_msort(self, lst, out):
        items = [resolve(x) for x in to_python_list(lst)]
        yield self.unify(out, mklist(sorted(items, key=_order_key)))

    def b_sort(self, lst, out):
        items = {term_to_str(resolve(x)): resolve(x) for x in to_python_list(lst)}
        yield self.unify(out, mklist(sorted(items.values(), key=_order_key)))

    def b_sum_list(self, lst, out):
        yield self.unify(out, sum(self.eval(x) for x in to_python_list(lst)))

    def b_max_list(self, lst, out):
        vals = [self.eval(x) for x in to_python_list(lst)]
        yield bool(vals) and self.unify(out, max(vals))

    def b_min_list(self, lst, out):
        vals = [self.eval(x) for x in to_python_list(lst)]
        yield bool(vals) and self.unify(out, min(vals))

    def b_atom(self, a): yield isinstance(deref(a), str)
    def b_number(self, a): yield isinstance(deref(a), (int, float)) and not isinstance(deref(a), bool)
    def b_integer(self, a): yield isinstance(deref(a), int)
    def b_var(self, a): yield isinstance(deref(a), Var)
    def b_nonvar(self, a): yield not isinstance(deref(a), Var)
    def b_is_list(self, a):
        t = deref(a)
        while isinstance(t, Struct) and t.functor == ".":
            t = deref(t.args[1])
        yield t == NIL

    def b_atom_concat(self, a, b, c):
        yield self.unify(c, f"{term_to_str(a)}{term_to_str(b)}")

    def b_atom_number(self, a, n):
        try:
            v = float(str(deref(a)))
            yield self.unify(n, int(v) if v == int(v) else v)
        except ValueError:
            yield False

    def b_format_atom(self, fmt, args, out):
        text, vals = str(deref(fmt)), [term_to_str(resolve(x)) for x in to_python_list(args)]
        for v in vals:
            text = text.replace("~w", v, 1)
        yield self.unify(out, text)

    # -- public API
    def query(self, text: str, limit: int | None = None) -> list[dict[str, Any]]:
        """query('grade_of(72, G, P)') -> [{'G': 'A-', 'P': 3.7}]"""
        p = Parser(tokenize(text.rstrip(" .") + " ."))
        goal = p.parse(1200)
        names = {k: v for k, v in p.vars.items() if not k.startswith("_")}
        out = []
        mark = len(self.trail)
        for _ in self.solve(goal):
            out.append({k: to_python(v) for k, v in names.items()})
            if limit and len(out) >= limit:
                break
        self.undo(mark)
        return out

    def once(self, text: str) -> dict[str, Any] | None:
        r = self.query(text, limit=1)
        return r[0] if r else None

    def holds(self, text: str) -> bool:
        return self.once(text) is not None


def _same_vars(a: Any, b: Any) -> bool:
    a, b = resolve(a), resolve(b)
    if isinstance(a, Var) or isinstance(b, Var):
        return a is b
    if isinstance(a, Struct) and isinstance(b, Struct):
        return all(_same_vars(x, y) for x, y in zip(a.args, b.args))
    return True


def _order_key(t: Any) -> tuple:
    t = deref(t)
    if isinstance(t, Var):
        return (0, "")
    if isinstance(t, (int, float)):
        return (1, t)
    if isinstance(t, str):
        return (3, t)
    return (4, len(t.args), t.functor, tuple(_order_key(a) for a in t.args))


BUILTIN_NAMES = {
    ("=", 2): "unify", ("\\=", 2): "not_unify", ("==", 2): "eq", ("\\==", 2): "neq", ("is", 2): "is",
    ("<", 2): "lt", (">", 2): "gt", ("=<", 2): "le", (">=", 2): "ge", ("=:=", 2): "aeq", ("=\\=", 2): "aneq",
    ("findall", 3): "findall", ("aggregate_all", 3): "aggregate_all", ("between", 3): "between",
    ("length", 2): "length", ("msort", 2): "msort", ("sort", 2): "sort", ("sum_list", 2): "sum_list",
    ("sumlist", 2): "sum_list", ("max_list", 2): "max_list", ("min_list", 2): "min_list", ("atom", 1): "atom",
    ("number", 1): "number", ("integer", 1): "integer", ("var", 1): "var", ("nonvar", 1): "nonvar",
    ("is_list", 1): "is_list", ("atom_concat", 3): "atom_concat", ("atom_number", 2): "atom_number",
    ("format_atom", 3): "format_atom",
}
