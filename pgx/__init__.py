"""ProvenanceGuard v2 extensions (``pgx``): everything v2 adds, kept OUT of ``pg/``.

``pg.week8_protocol.code_digest`` hashes every ``pg/*.py``; editing ``pg/`` would invalidate every frozen plan and with it the
Week 13 held-out ladder the mentor must also run.  So ``pg/`` stays byte-identical (v1, digest ``dca1a798...``) and v2 lives
here, with its own digest (``pgx.version.pgx_digest``) that covers ``pg/`` AND ``pgx/`` and the v2 data files.

Two mechanisms connect v2 to v1 code without editing it:

* ``pgx.registry.install()`` registers the new suites' tool contracts in v1's tool tables at runtime (it only ADDS entries for
  new tool names and delegates everything else, which ``tests/test_pgx_registry.py`` checks byte-for-byte on v1 tools);
* ``pgx.runtime.ProtectedRuntime`` is a generalised fork of ``pg.agentdojo_adapter.ProtectedWorkspaceRuntime`` parameterised by
  a ``SuiteSpec``; the v1 runtime itself is still what the v1 configurations (C0, SC-v3, SCP-v3) run.
"""
