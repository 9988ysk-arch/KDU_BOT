% handbook_rules.pl - the rules of the KDU Faculty of Computing Student Handbook 2022 as Prolog clauses.
% Facts (unit/9, grade_band/4, ...) come from handbook_facts.pl, generated from the PDF by logic_layer.py.
% Every rule cites its handbook page. Runs in SWI-Prolog and in prolog_engine.py (no cut, no assert).
%
%   swipl handbook_facts.pl handbook_rules.pl
%   ?- grade_of(72, G, P).                      G = 'A-', P = 3.7
%   ?- best_class(3.8, 1, 0, 0, 0, C).          C = 'Second Class (Upper Division)'

:- dynamic student/2.

% ---------------------------------------------------------------- grades (Table 12, p.54)
grade_of(Marks, Grade, GPV) :-
    number(Marks), Marks >= 0, Marks =< 100,
    grade_band(Grade, Low, High, GPV), Marks >= Low, Marks =< High.

% Each component (CA and ES) needs at least 35%, and the total at least 45% for a C (p.54).
grade_of_components(_, CA, ES, 'Ib', 0) :- CA < 35, ES < 35.
grade_of_components(_, CA, ES, 'Ie', 0) :- CA >= 35, ES < 35.
grade_of_components(_, CA, ES, 'Ia', 0) :- CA < 35, ES >= 35.
grade_of_components(Total, CA, ES, G, P) :- CA >= 35, ES >= 35, grade_of(Total, G, P).

gpv(Grade, GPV) :- grade_band(Grade, _, _, GPV).
gpv(Grade, 0) :- failure_grade(Grade).

grade_rank(G, R) :- grade_order(Order), nth1(R, Order, G).
better_or_equal(G1, G2) :- grade_rank(G1, R1), grade_rank(G2, R2), R1 =< R2.

pass_grade(G)  :- better_or_equal(G, 'C').                     % C or above (p.54)
weak_pass(G)   :- member(G, ['C-', 'D+']).                     % permitted weak passes (p.55)
fail_grade(G)  :- failure_grade(G).                            % Ie, Ia, Ib, Ne, Ab (p.54)

% ---------------------------------------------------------------- completing a semester (p.55, 4.5.1)
% a. C or above for all units (except permitted weak passes); b. not more than one D+ or C-;
% c. no failure grades or Ex.
semester_problem(Grades, failure(G)) :- member(G, Grades), fail_grade(G).
semester_problem(Grades, excused)     :- member('Ex', Grades).
semester_problem(Grades, too_many_weak(N)) :-
    aggregate_all(count, (member(G, Grades), weak_pass(G)), N), N > 1.
semester_complete(Grades) :- \+ semester_problem(Grades, _).

% ---------------------------------------------------------------- GPA (p.54, formula 4.4)
% Items are g(Grade, Credits); only GPA credits count; Ex is left out.
gpa_items(Items, SGPA) :-
    findall(P-C, (member(g(G, C), Items), G \== 'Ex', gpv(G, P)), PCs),
    findall(X, member(_-X, PCs), Cs), sum_list(Cs, Total), Total > 0,
    findall(Y, (member(P2-C2, PCs), Y is P2 * C2), Ws), sum_list(Ws, W),
    SGPA is W / Total.

% Module codes with grades: m(Code, Grade); credits are the module's GPA credits (NGPA units excluded).
% one value per module, even when the module is listed in both the CS and SE tables
unit_gpa_credits(Code, C) :- aggregate_all(max(X), unit(Code, _, _, _, _, _, X, _, _), C), C > 0.
gpa_of_modules(Pairs, SGPA, Used, Skipped) :-
    findall(g(G, C), (member(m(Code, G), Pairs), unit_gpa_credits(Code, C)), Items),
    findall(Code, (member(m(Code, _), Pairs), unit_gpa_credits(Code, _)), Used0), sort(Used0, Used),
    findall(Code, (member(m(Code, _), Pairs), \+ unit_gpa_credits(Code, _)), Skipped),
    gpa_items(Items, SGPA).

% ---------------------------------------------------------------- classes (Table 13 and p.57-58)
% eligible(Class, FGPA, Failures, FailuresInSem7or8, CarriedWeak, CarriedFail)
eligible('First Class', F, Fails, _, Weak, CFail) :-
    F >= 3.70, Fails =:= 0, Weak =:= 0, CFail =:= 0.
eligible('Second Class (Upper Division)', F, Fails, Fails78, Weak, CFail) :-
    F >= 3.30, Fails =< 1, Fails78 =:= 0, Weak =:= 0, CFail =:= 0.
eligible('Second Class (Lower Division)', F, Fails, Fails78, _, CFail) :-
    F >= 3.00, Fails =< 2, Fails78 =:= 0, CFail =:= 0.
eligible('Pass', F, _, _, _, CFail) :- F >= 2.00, CFail =:= 0.

% the highest class the student is eligible for ("The highest eligible Class shall be awarded", p.57)
best_class(F, Fails, Fails78, Weak, CFail, Class) :-
    class_order(Order), member(Class, Order),
    eligible(Class, F, Fails, Fails78, Weak, CFail),
    \+ ( member(Higher, Order), higher_class(Higher, Class),
         eligible(Higher, F, Fails, Fails78, Weak, CFail) ).

higher_class(A, B) :- class_order(O), nth1(IA, O, A), nth1(IB, O, B), IA < IB.

% why a student is NOT eligible for a class (used to explain the answer)
blocker('First Class', F, _, _, _, _, fgpa_below(3.70)) :- F < 3.70.
blocker('First Class', _, Fails, _, _, _, failures(Fails, 0)) :- Fails > 0.
blocker('First Class', _, _, _, Weak, _, carried_weak(Weak)) :- Weak > 0.
blocker('Second Class (Upper Division)', F, _, _, _, _, fgpa_below(3.30)) :- F < 3.30.
blocker('Second Class (Upper Division)', _, Fails, _, _, _, failures(Fails, 1)) :- Fails > 1.
blocker('Second Class (Upper Division)', _, _, F78, _, _, failure_in_sem_7_8(F78)) :- F78 > 0.
blocker('Second Class (Upper Division)', _, _, _, Weak, _, carried_weak(Weak)) :- Weak > 0.
blocker('Second Class (Lower Division)', F, _, _, _, _, fgpa_below(3.00)) :- F < 3.00.
blocker('Second Class (Lower Division)', _, Fails, _, _, _, failures(Fails, 2)) :- Fails > 2.
blocker('Second Class (Lower Division)', _, _, F78, _, _, failure_in_sem_7_8(F78)) :- F78 > 0.
blocker(_, _, _, _, _, CFail, carried_fail(CFail)) :- CFail > 0.

% ---------------------------------------------------------------- graduation (p.57, criteria a-f)
graduation_problem(GPA, _, gpa_below(2.00)) :- GPA < 2.00.
graduation_problem(_, MaxWeakPerSem, too_many_weak_per_semester(MaxWeakPerSem)) :- MaxWeakPerSem > 1.
can_graduate(GPA, MaxWeakPerSem) :- \+ graduation_problem(GPA, MaxWeakPerSem, _).

% ---------------------------------------------------------------- repeating units (p.55, 4.5.2 h-j)
repeat_rule(G, must_repeat) :- fail_grade(G).
repeat_rule(G, may_repeat_up_to_c) :- weak_pass(G).
repeat_rule(G, gpa_pass_no_repeat) :- pass_grade(G).
ngpa_repeat_rule(G, may_repeat_to_upgrade) :- pass_grade(G).          % rule i, p.55
ngpa_repeat_rule(G, R) :- \+ pass_grade(G), repeat_rule(G, R).
resit_component('Ie', 'ES').
resit_component('Ia', 'CA/PBCA').
resit_component('Ib', 'ES and CA/PBCA').

% ---------------------------------------------------------------- admission (p.10)
stream_ok(Prog, maths) :- member(Prog, ['CS', 'SE', 'CE', 'IT', 'IS']).
stream_ok(Prog, maths_physics_plus) :- member(Prog, ['CS', 'SE', 'CE']).    % Maths, Physics + Chem/HMaths/ICT
stream_ok(Prog, S) :- member(Prog, ['IT', 'IS']), member(S, [biology, commerce, arts]).
stream_not_ok(Prog, technology) :- member(Prog, ['CS', 'SE', 'CE', 'IT', 'IS']).

% military students: one fact per requirement, checked against what the student said
military_problem(age(A))      :- student(age, A), military_min_age(Min), A < Min.
military_problem(age(A))      :- student(age, A), military_max_age(Max), A > Max.
military_problem(married)     :- student(married, yes).
military_problem(weight(W))   :- student(weight, W), military_min_weight(Min), W < Min.
military_problem(chest(C))    :- student(chest, C), military_min_chest(Min), C < Min.
military_problem(citizen)     :- student(citizen, no).
military_problem(height(H, Min)) :-
    student(height, H), student(gender, G), student(service, S), height_min(G, S, Min), H < Min.
military_unknown(Req) :- military_requirement(Req), \+ student(Req, _).
military_eligible :- \+ military_problem(_).

% ---------------------------------------------------------------- curriculum reasoning
modules_in(Prog, Sem, Codes) :- findall(C, unit(C, _, Prog, _, Sem, _, _, _, _), L), sort(L, Codes).
same_modules(Sem) :- modules_in('CS', Sem, L), modules_in('SE', Sem, L).            % slot is not needed
slot_needed(degree, Sem) :- between(1, 8, Sem), \+ same_modules(Sem).
elective_semester(Prog, Sem) :- unit(_, _, Prog, _, Sem, elective, _, _, _).

compulsory_credits(Prog, Sem, Sum) :-
    aggregate_all(sum(C), unit(_, _, Prog, _, Sem, compulsory, C, _, _), Sum).
% elective GPA credits a student must take = semester total (summary table) - compulsory GPA credits
elective_credits_needed(Prog, Sem, Need) :-
    semester_total(Prog, Sem, Total, _, _), compulsory_credits(Prog, Sem, Comp),
    Need is Total - Comp, Need > 0.

% consistency check: semesters without electives must add up to the summary table
credit_mismatch(Prog, Sem, Table, Sum) :-
    semester_total(Prog, Sem, Table, _, _), \+ elective_semester(Prog, Sem),
    aggregate_all(sum(C), unit(_, _, Prog, _, Sem, _, C, _, _), Sum), Sum =\= Table.

% required slots of a question type (used by the dialog manager, eq. 23)
required_slots(curriculum, [degree, period]).
required_slots(electives, [degree, semester]).
required_slots(head_of_department, [department]).
required_slots(entry_requirements, [student_type]).
required_slots(staff_list, [department]).
missing_slot(Type, Known, Slot) :- required_slots(Type, Slots), member(Slot, Slots), \+ member(Slot, Known).
