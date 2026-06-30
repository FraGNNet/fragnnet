# Tests

This suite exercises featurizers, datasets, and training utilities. When adding tests:

- Assert specific expected values/shapes/edge cases instead of only checking non-empty results.
- Include at least one known example and one edge case for new features.
- Prefer pytest fixtures for reusable molecules, graphs, and temp files.
- Keep imports clean (stdlib / third-party / local) and avoid print; use assertions.
- Run `pytest tests/ -v` (editable install `pip install -e .` recommended for imports).
