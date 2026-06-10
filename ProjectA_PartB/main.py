#!/usr/bin/env python3
"""
main.py — Autograder entry point + offline index builder.

Autograder calls:   from main import run; run(queries)
Build script calls: from main import build_offline_index; build_offline_index()
"""

from utils import (
    SYNTH_INDEX_PATH,
    SYNTH_META_PATH,
    ensure_artifacts_dir,
    load_corpus,
    timer,
)
from retrieve import load_indexes, retrieve_batch


def build_offline_index() -> None:
    """Build and save all artifacts from the corpus."""
    from index import build_synth_index, save_synth_index

    ensure_artifacts_dir()

    with timer("Loading corpus"):
        pages = load_corpus()

    with timer("Building synthetic-page index"):
        index = build_synth_index(pages)

    with timer("Saving artifacts"):
        save_synth_index(index, SYNTH_INDEX_PATH, SYNTH_META_PATH)

    print("\nAll artifacts saved. Ready to commit.")


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    import os
    missing = [p for p in [SYNTH_INDEX_PATH, SYNTH_META_PATH] if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"Missing artifacts: {missing}. Run python3 scripts/build_index.py first.")
    load_indexes(npz_path=SYNTH_INDEX_PATH, meta_path=SYNTH_META_PATH)
    _loaded = True


def run(queries: list) -> list:
    """Return top-10 page IDs per query. Called by autograder."""
    _ensure_loaded()
    return retrieve_batch(queries)
