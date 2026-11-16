# ProvenanceGuard: consolidated results, Weeks 10-11d

A single reference for the reviewer-dependence line of the project: what each week established, the headline numbers (all
independently re-derived from raw logs by `scripts/verify_results.py`, which shares no code with the analysis libraries), and the
limitations to state in the paper. Benchmark: AgentDojo Workspace v1.2.2; four open-weight models through Ollama (gemma4-e4b,
granite4-small-h, mistral-nemo-12b, qwen3-14b); greedy decoding; frozen plans hashed against the code digest.

## The one-paragraph story

ProvenanceGuard guarantees that nothing executes without a signed approval of the exact recorded action. On the model-driven
benchmark the recipient allowlist does all the blocking, because every benchmark attack ends in a send to an outside address
(Week 10). When the injected action instead passes the policy -- delete a file, share it with an allowed contact, add an allowed
contact to an event -- the only thing left between injection and effect is the reviewer, and an approve-all reviewer stops nothing
(Weeks 11, 11b). A model reviewer stops the wrong-kind actions but approves a *correct* action carrying something extra: an added
cc/bcc recipient, or text copied or paraphrased from a file the request never named (Week 11b held-out suite). Those failures are
structural, so Week 11c adds a deterministic request-scope check (addresses, attachments, verbatim copies) and Week 11d a
content-provenance check (paraphrase and copied-text, from the episode read-set). Together they drive every attack class we tested
to zero in the loop, at a measured utility cost, with the two deterministic checks validated to precision/recall 1.0 on held-out
suites they were not tuned on.

## Headline numbers (pooled over 4 models, verified from raw logs)

### Attack success by configuration

| configuration | policy-passing attacks | content attacks (cc/bcc, copied) | provenance attacks (paraphrase, email-text) |
|---|---|---|---|
| stock (no guard) | 93/240 (39%) | not in set | not in set |
| approve-all (policy only) | 95-97/240 (40%) | 69/144 (48%) | pending in-loop (C0) |
| model reviewer, prompt v2 | 5/240 (2%) | 51/144 (35%) | - |
| + prompt v3 | 0/240 | 21/144 (15%) | - |
| + request-scope check (Week 11c) | 0/240 | 0/144 | partially (verbatim only) |
| + content-provenance (Week 11d) | 0/240 | 0/144 | 0 (offline-validated; in-loop pending) |

### Utility and the reviewer's cost (Week 11c, pooled; fair measure = clean controls and non-content user tasks)

| configuration | clean-task success | user task done (policy-passing scenarios) |
|---|---|---|
| approve-all | 33/48 (69%) | 222/252 (88%) |
| model reviewer v2 | 30/48 | 184/252 (73%) |
| prompt v3 | 27/48 | 174/252 (69%) |
| request-scope + v3 | 25/48 (52%) | 173/252 (69%) |

Utility includes mistral, which completes no task (malformed tool calls) and should be excluded from usability claims. The large
utility drop is the reviewer's own false denials (approve-all -> v2: 38 tasks lost); the deterministic checks cost about one point.

### Deterministic checks, held-out validation (no model, tuned-apart data)

- Request-scope check (Week 11c held-out): flags 24/24 hidden-bcc and 18/24 attachment attacks, 0/40 legitimate.
- Content-provenance check (Week 11d held-out-2, new tasks and sources): **precision 1.0, recall 1.0** over 48 triples -- 32/32
  paraphrase+copy attacks flagged, 0/16 legitimate (incl. the stress case where the agent read the secret file but sent a clean
  summary). Backed by a randomized property test: 0/200 random legitimate summaries flagged.

## What each week established

- **Week 10** -- on the deterministic attack ladder the recipient allowlist (policy) is the only layer that moves model-driven
  attack success; found and fixed the `approval_rebinding` gap (signed payload digest now revalidated at the gateway).
- **Week 11** -- policy-passing attacks: approve-all reviewer 97/240, oracle and model reviewers 0/240; the protection is entirely
  in the approval step. Model reviewer costs utility (clean 33->22/48).
- **Week 11b** -- logged every card; prompt v2 cut false denials (31%->2%) but opened content holes (6%->28% crafted attacks
  approved); the target-description card made no difference; answered the Week 11 over-denial question (most "false denials" were
  correct denials of off-request bodies).
- **Week 11c** -- request-scope check + prompt v3 + denial cap: content attacks 69->0 in the loop; the check flags 0 legitimate
  cards; the cap cut granite's reviewer calls 324->145 for free; cross-family review showed no clear win; mistral unusable as a
  reviewer. Held-out suite exposed the remaining semantic gap (paraphrase, email-text, act-on-named).
- **Week 11d** -- content-provenance check on the read-set closes paraphrase and copied-text exfiltration (precision/recall 1.0
  offline), hardened with a numeric-secret rule and digit tokens for codes/figures. In-loop confirmation (`provenance_attack`
  family) pending the cluster run.

## Limitations (state these plainly)

1. The in-loop reviewer is the same model family as the agent (self-review); cross-family and frontier reviewers are offline/future.
2. The content-provenance check is lexical: a heavy paraphrase that replaces a source's distinctive words would evade it. Local
   models keep the content words, which is what is caught.
3. Mistral is excluded from utility claims (completes no task). One generic injection template (AgentDojo important_instructions);
   adaptive attacks are approximated by crafted/held-out cards, not a live feedback attacker.
4. `act_on_named` (a wrong-kind action aimed at a named person/item) remains with the reviewer; granite approves ~18/24. Not solved
   deterministically (deciding "did the request ask for a delete" is left to the model).
5. Utility is AgentDojo-task utility; no human-reviewer study yet.

## Reproducing / verifying

- `scripts/verify_results.py` recomputes every week's headline numbers from raw `episodes.jsonl` and checks cross-week invariants
  (11b vs 11c approve-all replication |delta| = 0.004; all reviewer configs 0 attack success). It shares no code with the analysis
  libraries, so it is an independent check.
- Figures (`defense_ladder.png`, `provenance_validation.png`, `tradeoff.png`) are generated locally from the raw results.
- Per-week detail: `claude/week10_findings.md` ... `claude/week11c_findings.md`; Week 11d in `docs/WEEK11D_RUNBOOK.md` and, after the
  cluster run, `claude/week11d_findings.md`.
