# ProvenanceGuard — run it on your model (quick start)

**What this does.** Runs the frozen ProvenanceGuard test (72 held-out attack scenarios × 3 seeds × 3 defense levels = 648
episodes) against any OpenAI-compatible model and reports how often an injected instruction still succeeds. **No hosted-model
result exists yet; yours is the first.** Detail and reasoning: `docs/FRONTIER_HANDOFF.md`.

**Needs:** Python 3.11, 3.12 or 3.13, and internet (pip and your model's API). Tested on Linux with 3.11, 3.12 and 3.13; not tested on Windows or macOS.

## 1. Install (2 min)

```bash
unzip provenanceguard-handoff-*.zip && cd provenanceguard-handoff      # or, from GitHub:  git clone <repo-url> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.lock
```

## 2. Check the package (1 min, no key, no cost)

```bash
python scripts/handoff_verify.py      # last line must be: READY
python scripts/frontier_selftest.py   # last line must be: SELF-TEST PASSED   (uses a scripted mock, not a model)
```

If either fails, stop and send me its full output.

## 3. Point it at your model

```bash
export PROVENANCEGUARD_API_BASE=https://api.example.com/v1    # Windows PowerShell: $env:PROVENANCEGUARD_API_BASE="..."
export PROVENANCEGUARD_API_KEY=sk-...
export PROVENANCEGUARD_MODEL=model-name
# only if your provider needs them (the next step tells you):
#   export PROVENANCEGUARD_TOKEN_PARAM=max_completion_tokens
#   export PROVENANCEGUARD_OMIT_PARAMS=temperature,top_p
# only for a cost cap (USD per 1,000 tokens):
#   export PROVENANCEGUARD_PRICE_PER_1K_INPUT=0.003  PROVENANCEGUARD_PRICE_PER_1K_OUTPUT=0.015
```

## 4. Smoke run (12 episodes)

```bash
python scripts/run_frontier_campaign.py --dry-run    # shows the plan; sends nothing
python scripts/run_frontier_campaign.py --smoke      # probes the model, then runs 12 episodes
```

Expect `heldout: ran 12 episodes (0 errored)`. If it says `REFUSED`, it names the setting to change (rerun step 4).
Then `python scripts/frontier_report.py`.

## 5. Full run (648 episodes, roughly 7–13 million tokens)

```bash
python scripts/run_frontier_campaign.py --max-cost 100      # USD cap; needs the two price variables (else omit --max-cost)
python scripts/frontier_report.py
```

Safe to stop (Ctrl-C) and resume: run the same command again. Exit code 0 = finished; anything else, read the message.

## 6. Hand-label the content attacks (required — the printed content numbers are only a first screen)

```bash
python scripts/content_rescore.py --dump-unlabelled --template labels/labels.json
# open labels/labels.json: for each body set "verdict" to "leak" or "clean"   (rule: docs/FRONTIER_HANDOFF.md, step 6)
python scripts/content_rescore.py --labels labels/labels.json
```

## 7. Send back

`results/frontier/` (the episode logs), `labels/labels.json`, and the output of `frontier_report.py`.

## 8. The rest of the report (optional, same model)

Everything else the local models ran is a `--suite`: `python scripts/run_frontier_campaign.py --list`.

```bash
python scripts/run_frontier_campaign.py --suite pdf --smoke                 # every experiment, one scenario per family: cheap end-to-end check
python scripts/run_frontier_campaign.py --suite main --max-cost 300         # held-out ladder + evasion + the three cheap baselines
python scripts/run_frontier_campaign.py --suite appendix --max-cost 300     # tuning, Week 11/11b/11c, Week 8 comparison, scripted attacker
python scripts/frontier_report.py                                           # one block per suite
```

The full report is about 4,700 episodes and up to 150 attacker attempts per model (roughly 57–97 M agent tokens); the dry run
(`--suite pdf --dry-run`) prints the figures. Mapping from report section to suite, and the analysis script for each:
`docs/FRONTIER_HANDOFF.md`, "Every experiment in the report".

Optional experiments (reviewer = a different model; self-review vs. independent review): `docs/FRONTIER_HANDOFF.md`, "The reviewer".
