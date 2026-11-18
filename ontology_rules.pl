% ontology_rules.pl - general rules over the automatically generated ontology (data/handbook_ontology.pl).
% Loads in SWI-Prolog unchanged:  swipl data/handbook_ontology.pl ontology_rules.pl
% Each rule replaces a family of hand-coded special cases with one inference over the relations.

% ---- departments, programmes, heads ---------------------------------------------------------------
department_of_programme(P, D)  :- offers(D, P).
head_of_programme(P, H)        :- offers(D, P), head_of(H, D).
programmes_of_department(D, P) :- offers(D, P).
department_without_degree(D)   :- type(D, 'Department'), \+ offers(D, _).
degrees_of_head(H, P)          :- head_of(H, D), offers(D, P).
staff_department(S, D)         :- member_of(S, D).

% ---- modules ------------------------------------------------------------------------------------
department_of_module(M, D)     :- in_programme(M, P), offers(D, P).
head_for_module(M, H)          :- department_of_module(M, D), head_of(H, D).
programmes_of_module(M, P)     :- in_programme(M, P).
in_both(M, P1, P2)             :- in_programme(M, P1), in_programme(M, P2), P1 \== P2.
only_in(M, P, Other, Sem)      :- offered(M, P, Sem), \+ offered(M, Other, Sem).
shared_in_semester(M, P1, P2, Sem) :- offered(M, P1, Sem), offered(M, P2, Sem).
modules_of_department(D, M)    :- offers(D, P), in_programme(M, P).
credits_of(M, P, C, gpa)       :- gpa_credits(M, P, C).
credits_of(M, P, C, ngpa)      :- ngpa_credits(M, P, C).
credits_of(M, P, C, mgpa)      :- mgpa_credits(M, P, C).

% ---- staff and research -------------------------------------------------------------------------
researcher(T, S)               :- research_topic(S, T).
researcher_in_department(T, D, S) :- research_topic(S, T), member_of(S, D).
researcher_for_programme(T, P, S) :- offers(D, P), researcher_in_department(T, D, S).

% ---- grades and classes -------------------------------------------------------------------------
passing_grade(G)               :- grade_status(G, pass).
weak_pass_grade(G)             :- grade_status(G, weak_pass).
failing_grade(G)               :- grade_status(G, fail).

% ---- consistency checks (run at start-up) ------------------------------------------------------
programme_without_department(P) :- type(P, 'Programme'), \+ offers(_, P).
module_without_programme(M)     :- type(M, 'Module'), \+ in_programme(M, _).
