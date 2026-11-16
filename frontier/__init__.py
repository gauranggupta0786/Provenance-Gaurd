"""Handoff tooling that runs ProvenanceGuard against a hosted model.

This package is deliberately OUTSIDE ``pg/``.  ``pg.week8_protocol.code_digest`` hashes every ``pg/*.py`` file, so a module
added there changes the Week 8 defense version and the Week 11 code digest, which would invalidate the frozen Week 11e plan
and the four model results already measured under it.  The adapter is scaffolding around the defense, not part of it, so it
lives in its own package and the frozen digests stay exactly as they were.
"""
