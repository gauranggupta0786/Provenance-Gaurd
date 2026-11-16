# Environment and model manifest

| Field | Value |
|---|---|
| Date | 2026-09-25 |
| Python | 3.12.14, short-path `%USERPROFILE%\Documents\Codex\.venvs\provenanceguard-v1` environment |
| AgentDojo | 0.1.35 |
| Benchmark | `workspace` v1.2.2 |
| Model | `qwen3.5:9b`, Ollama digest `6488c96fa5fa` |
| Runtime | Ollama 0.34.4, native endpoint on `127.0.0.1:11434`; OpenAI-compatible adapter retained |
| Model settings | thinking disabled, temperature 0, top-p 1, top-k 20, min-p 0, seed 42, context 8192, max output 1024 |
| Hardware | Intel i7-12650H; about 16 GB RAM; RTX 3050 Laptop GPU, 4 GB VRAM |
| Observed placement | 74% CPU / 26% GPU through Vulkan |
| Key libraries | `cryptography` 46.0.6, `pytest` 8.4.2, `openai` 3.19.2 |
| Signature scheme | Ed25519 via `cryptography` |
| Storage | SQLite WAL, append-only triggers and signed hash-chain checkpoints |
| External effects | AgentDojo simulation only; no real email, calendar, drive, or payment APIs |
| API cost | $0; model weights execute locally |

The deterministic agent in `pg/agent.py` remains available for fast structural
regressions. Live episodes use `pg.ollama_model.SeededOllamaLLM` through the
common native model client and record token counts, normalized tool intents,
the full model-profile hash, and the provider adapter in evidence.

## Remaining environment work

| Item | Target |
|---|---|
| Full held-out run and adaptive campaigns | Week 6 |
| Independent reproduction environment | Week 7 |
| HSM/KMS-backed key custody | Post-Week 7 deployment hardening |
| Real external-tool idempotency and reconciliation | Post-prototype |
