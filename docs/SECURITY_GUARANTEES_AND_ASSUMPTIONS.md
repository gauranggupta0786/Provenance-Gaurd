# Security guarantees and experimental assumptions

## Implemented and tested guarantees

- Every classified Workspace write crosses strict tool validation, policy, signed proposal, approval, live gateway revalidation, idempotency reservation, adapter execution, and a durable receipt.
- Read results in the protected runtime are authorized by trusted principal and tool, flattened to field/object evidence, sensitivity-labelled, and persisted by digest without persisting raw protected values.
- The final protected response is deterministically checked against protected values observed through read tools. Tested unauthorized matches are blocked before delivery and recorded in a versioned output-guard record.
- Current policy and source versions are revalidated immediately before approval consumption and execution.
- Development Ed25519 keys persist across restart, retain public keys after rotation, and record creation, retirement, and ancestry metadata.
- Audit events are signed and hash chained. Version-2 checkpoints are separately signed, chained, database-identity-bound, and anchored. Event editing, middle deletion, tail deletion, checkpoint editing/deletion, and database replacement fail verification.
- Security labels are derived from final state, signed evidence, gateway outcomes, and output-guard evidence—not model self-report or an LLM judge.

## Bounded guarantees

- Output matching detects exact case-insensitive disclosure of values observed through mediated reads. It does not yet prove noninterference against arbitrary paraphrase, encryption, compression, images, steganography, or undisclosed side channels.
- Field sensitivity is a deterministic name/content policy suitable for the benchmark. A deployment needs a domain data catalogue and owner-maintained classification rules.
- Source freshness uses monotonic source versions. Source producers must issue increasing versions for changed authoritative objects.
- Checkpoint anchors are separate files on the development host. Production needs independently retained/WORM or remote transparency-log anchors.
- The local persistent key store is for development only. It stores exportable private key material and is not a KMS/HSM.
- AgentDojo is a simulator; OS, network, UI, plugin, and covert-channel mediation are outside the benchmark boundary.

## Deployment requirements

Use a non-exportable KMS/HSM signing key, independent public-key registry, authenticated workload identity, encrypted evidence storage, remote/WORM checkpoint anchoring, centralized policy/version service, domain classification catalogue, access logging, backup/restore drills, and incident-response key revocation.
