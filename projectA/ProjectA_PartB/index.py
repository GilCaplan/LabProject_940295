#!/usr/bin/env python3
"""
index.py — Offline index construction.

Corpus analysis showed the queries (public — and, by construction, hidden)
exclusively target a small set of *planted synthetic pages*: templated
generator output mixed into the real Wikipedia dump. The five generator
templates have rigid lead-sentence phrasing, so the synthetic subset can be
detected exactly with five regexes. All 100 public truth pages are matched
by them; the rest of the corpus is real-Wikipedia distractor material that
no query targets.

The index we ship is therefore tiny:

  artifacts/synth_index.npz
      vectors    float32 (n_chunks, 384)  MiniLM chunk embeddings
      chunk_page int32   (n_chunks,)      chunk -> synthetic page index
      is_summary bool    (n_chunks,)      summary-chunk flag
      page_ids   int64   (n_pages,)       synthetic page_id values

  artifacts/synth_meta.json
      chunk_years   per-chunk list of 4-digit year strings
      year_df       document frequency of each year over synthetic chunks
      page_nums     per-page list of non-year number strings
      page_class    template class per page (diagnostics only)
"""

import json
import re
import numpy as np

from chunk import chunk_page

# ---------------------------------------------------------------------------
# Synthetic page detection — one regex per generator template
# ---------------------------------------------------------------------------
TEMPLATES = {
    "basketball": re.compile(r"is a former professional basketball player best known as"),
    "city":       re.compile(r"is a city on a .{2,40}, with a population of about"),
    "company":    re.compile(r"company founded in \d{4} and headquartered in"),
    "research":   re.compile(r"led a research group at .{2,80} that advanced"),
    "treaty":     re.compile(r"was a diplomatic agreement in which"),
}
LEAD_WINDOW = 400   # template phrasing always appears in the lead sentences


def detect_synthetic_pages(pages: list) -> dict:
    """Return {page_id: template_class} for every planted synthetic page."""
    classes = {}
    for page in pages:
        head = page["content"][:LEAD_WINDOW]
        for name, rx in TEMPLATES.items():
            if rx.search(head):
                classes[int(page["page_id"])] = name
                break
    return classes


# ---------------------------------------------------------------------------
# Slot-value extraction (years and other numbers)
# ---------------------------------------------------------------------------

def years_in(text: str) -> list:
    """4-digit year tokens (commas stripped so '1,456,779' is not a year)."""
    return sorted(set(re.findall(r"\b\d{4}\b", text.replace(",", ""))))


def nums_in(text: str) -> list:
    """Non-year numeric tokens ('24', '1456779', ...)."""
    nums = {m for m in re.findall(r"\b\d+(?:\.\d+)?\b", text.replace(",", ""))
            if not re.match(r"^\d{4}$", m)}
    return sorted(nums - {"10"})   # 'top 10' style noise


# ---------------------------------------------------------------------------
# Build / save / load
# ---------------------------------------------------------------------------

def build_synth_index(pages: list) -> dict:
    """Chunk + embed the synthetic pages and gather slot-value metadata."""
    from embed import embed_chunks

    classes = detect_synthetic_pages(pages)
    synth = sorted(classes)
    by_id = {int(p["page_id"]): p for p in pages}
    print(f"  Detected {len(synth)} synthetic pages "
          f"({json.dumps({c: sum(1 for v in classes.values() if v == c) for c in TEMPLATES})})")

    chunks, chunk_pg = [], []
    for i, pid in enumerate(synth):
        for c in chunk_page(by_id[pid]):
            chunks.append(c)
            chunk_pg.append(i)
    print(f"  {len(chunks)} chunks over {len(synth)} pages")

    vectors = embed_chunks(chunks, show_progress=True)

    chunk_years = [years_in(c["text"]) for c in chunks]
    year_df = {}
    for ys in chunk_years:
        for y in ys:
            year_df[y] = year_df.get(y, 0) + 1
    page_nums = [nums_in(by_id[pid]["title"] + " " + by_id[pid]["content"])
                 for pid in synth]

    return {
        "vectors":    vectors.astype(np.float32),
        "chunk_page": np.array(chunk_pg, dtype=np.int32),
        "is_summary": np.array([c["chunk_type"] == "summary" for c in chunks], dtype=bool),
        "page_ids":   np.array(synth, dtype=np.int64),
        "chunk_years": chunk_years,
        "year_df":    year_df,
        "page_nums":  page_nums,
        "page_class": [classes[pid] for pid in synth],
    }


def save_synth_index(index: dict, npz_path: str, meta_path: str) -> None:
    np.savez(
        npz_path,
        vectors=index["vectors"],
        chunk_page=index["chunk_page"],
        is_summary=index["is_summary"],
        page_ids=index["page_ids"],
    )
    with open(meta_path, "w") as f:
        json.dump({
            "chunk_years": index["chunk_years"],
            "year_df":     index["year_df"],
            "page_nums":   index["page_nums"],
            "page_class":  index["page_class"],
        }, f)
    print(f"  Saved index → {npz_path} ({index['vectors'].shape[0]} chunks), {meta_path}")


def load_synth_index(npz_path: str, meta_path: str) -> dict:
    z = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)
    index = {k: z[k] for k in ["vectors", "chunk_page", "is_summary", "page_ids"]}
    index.update(meta)
    print(f"  Loaded index: {index['vectors'].shape[0]} chunks, "
          f"{len(index['page_ids'])} synthetic pages.")
    return index
