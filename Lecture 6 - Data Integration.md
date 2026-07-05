# Data Integration

## Lecture Outline
1. Data Integration
2. Entity Matching
3. Pipeline
4. DITTO
5. Challenges & Opportunities

---

## 1. Data Integration

### What is Data Integration?
**Bring them together.** Data integration is the process of combining data from multiple data sources into a unified, consistent and coherent dataset. Multiple raw data sources feed into one unified store, which downstream users query, analyze, and make decisions from.

### Why Data Integration?
Key motivations:
- Fragmentation of data
- Inconsistent format and schemas
- Reduce computational effort
- Support downstream applications

### The Data Integration Realm
Main Data Integration tasks:
- Data Discovery
- Data Extraction
- Data Cleaning
- Schema Matching
- **Entity Matching**

### DI and ML: A Natural Synergy
- **DI for ML**: ML applications rely on data integration to use accurate, clean, and relevant data.
- **ML for DI**: Most contemporary DI solutions rely on ML-based approaches.

---

## 2. Entity Matching

### What is an Entity?
Examples: Person, Product, Place, Organization, Academic paper.

### Definition
**Entity Matching (EM)** is a core DI task that focuses on identifying and connecting data records that refer to the same real-world entity.

Example (two product tables, dashed lines = same real-world product):
| title | manf./modelno | price |
|---|---|---|
| instant immersion spanish deluxe 2.0 | topics entertainment | 49.99 |
| adventure workshop 4th-6th grade 7th edition | encore software | 19.99 |
| sharp printing calculator | sharp el1192bl | 37.63 |

| title | manf./modelno | price |
|---|---|---|
| instant immers spanish dlux 2 | NULL | 36.11 |
| encore inc adventure workshop 4th-6th grade 8th edition | NULL | 17.1 |
| new-sharp shr-el1192bl two-color printing calculator 12-digit lcd black red | NULL | 56.0 |

(row 1↔row 1 match, row 2↔row 2 is a near-miss/non-match in the example — edition differs, row 3↔row 3 match)

### Why is EM Necessary?
- Typographical errors and missing values
- Inconsistent formats and terminology
- Multiple representations of the same real-world entity
- Multi-dimensional data aspects: temporal, spatial, context, etc.

### Terminology: EM and Its Variants
EM has numerous variants (entity resolution, record linkage, record deduplication, etc.). These terms are often used interchangeably, though some distinctions may exist in specific domains.

### Why EM is Hard?
Challenges:
- Heterogeneous schemas
- Dirty data
- Language understanding
- Scalability
- Recognizing the crucial matching factors
- Imbalanced supervision (far more non-matches than matches)

---

## 3. Pipeline

### EM Steps
1. **Blocking**
2. **Matching**
3. **Merging**

```
D1 ─┐
    ├─▶ Blocker ─▶ Candidate Pairs ─▶ Matcher ─▶ Resolution ─▶ Merger ─▶ Clean Dataset
D2 ─┘
```

### Blocking
**Goal:** Reduce the number of record comparisons by grouping potentially matching records into "blocks" before matching. Without blocking, comparing all pairs across two datasets is O(n·m), infeasible at scale.

Example table:
| Record ID | FName | LName | Dept | Phone |
|---|---|---|---|---|
| r1 | Judith | Jones | Sales | 0523332225 |
| r2 | Dave | Cohen | R&D | 0524242166 |
| r3 | Mike | Cohen | R&D | 0523347181 |
| r4 | David | Cohen | Research and Development | 0524242167 |
| r5 | Amalia | Jones | Farms | 0522443861 |
| r6 | Yehudit | Jones | Sales | 0523332225 |
| r7 | Jim | Jones | Freedom Fighters | – |
| r8 | Jim | Goldberg | Research and Development | 0524242367 |
| r9 | David | Kagan | Sales | – |
| r10 | Amalia | Goldberg | Farms | 0523334652 |
| r11 | Judith | Jons | sales | – |

**Approaches:**
- Token-based (e.g., blocking key = shared token/attribute value, such as LName)
- Q-grams (character n-gram overlap)
- Semantic (embedding-based similarity)

**Blocking (Traditional) example** — group records sharing a blocking key into blocks, then only compare within-block pairs:

| Block ID | Records |
|---|---|
| 1 | r1, r5, r6, r7 |
| 2 | r2, r3, r4 |
| 3 | r1, r6, r11 |
| 4 | r1, r9, r11 |

Expanded into candidate pairs:
| Block ID | Candidate Pairs |
|---|---|
| 1 | (r1,r5), (r1,r6), (r1,r7), (r5,r6), (r5,r7), (r6,r7) |
| 2 | (r2,r3), (r2,r4), (r3,r4) |
| 3 | (r1,r6), (r1,r11), (r6,r11) |
| 4 | (r1,r9), (r1,r11), (r9,r11) |

Note: a pair can appear in multiple blocks (e.g., (r1,r6) and (r1,r11) both appear twice) — de-duplicate candidate pairs before matching.

**Semantic Blocking**: uses embeddings/semantic similarity instead of exact token match to group records — catches synonyms and paraphrases that token-based blocking misses (at higher compute cost).

### Matching
**Goal:** Determining whether two records (a candidate pair) refer to the same real-world entity.

**Analogy to reranking:** Blocking + Matching resembles the two-stage IR process of first-stage retrieval + reranking — blocking is a cheap, high-recall filter (like retrieval), and matching is an expensive, high-precision classifier applied only to the surviving candidates (like reranking).

**Different Matching Approaches:**
- **Heuristic-based**
  - String similarity (edit distance, Jaccard, cosine over tokens, etc.)
  - Rule-based
- **Machine Learning-based**
  - Traditional ML models (feature vectors → classifier)
- **Foundation Models**
  - Pretrained LMs (e.g., BERT)
  - LLMs (e.g., GPT-4)

### Merging
**Goal:** Merging is the final step in the pipeline, focusing on producing a unified representation of matching objects (a.k.a. resolution/fusion).

**Methods:**
- Longest name string
- Most frequent value
- Most recent
- Domain knowledge

---

## 4. DITTO — Deep Entity Matching with Pre-trained Language Models

### Key Idea
Use pretrained language models (e.g., BERT) with structured input formatting to learn matching directly from raw text.

### How?
- The PTLM was already trained on large corpora.
- Finetune the PTLM on the specific matching task using labeled data.

### DITTO Pipeline
```
Table A ─┐                    ┌─▶ Serialize ─▶ Inject DK ─▶ Summarize ─▶ Matcher ─▶ Matched Pairs
         ├─▶ Blocker ─▶ Candidate Pairs
Table B ─┘                    └─▶ Sample & Label ──▶ Augment ──▶ (used to Train Matcher)
                                          │
                                          └── (feeds back to) Train Advanced Blocking
```
Three DITTO-specific components sit between candidate-pair generation and the matcher: **① Inject DK, ② Summarize, ③ Augment** (numbered on the diagram), wrapped around **Serialize**.

### ① Serialize
Turn a structured record into a flat text sequence the LM can consume, tagging column names and values with special tokens:

```
[CLS] [COL] title [VAL] instant immersion spanish deluxe 2.0
      [COL] manf./modelno [VAL] topics entertainment
      [COL] price [VAL] 49.99
[SEP] [COL] title [VAL] instant immers spanish dlux 2
      [COL] manf./modelno [VAL]
      [COL] price [VAL] 36.11
```

### ② Domain Knowledge (DK) Injection
Manually point at information that is more relevant (**span type**) and normalize it to a unified format (**span normalization**).

| Entity Type | Types of Important Spans |
|---|---|
| Publications, Movies, Music, Organizations, Employers | Persons (e.g., Authors), Year, Publisher |
| Products | Last 4-digit of phone, Street number, Product ID, Brand, Configurations (num.) |

- **Span type:** use NER to identify known types and insert special tokens to highlight them.
- **Span normalization:** use a specified set of rewriting rules to replace the spans of interest (e.g., normalize "topics entertainment" / brand names, canonicalize numbers).

### Summarize
BERT-based models are limited to 512 tokens. Summarization retains only non-stopword tokens with the highest TF-IDF scores, so long records still fit in the context window without losing the most discriminative content.

### ③ Augmentation
- A widely-used technique for generating additional training data from existing samples.
- Done by applying simple operations such as delete, shuffle, and swap.
- Helps the model learn from "harder" samples.
- **Main idea:** interpolate the original sample with the modified one (based on their representations) — i.e., **MixUp**.

```
Original ──────────────┐
                        ├─▶ BERT ─▶ [seq rep 1] ─┐
Augmented (DA-Op) ──────┘           [seq rep 2] ─┤─▶ Interpolate (MixUp, λ / 1-λ) ─▶ Linear → Softmax → Loss ─▶ Backprop
```

---

## 5. Challenges & Opportunities

### Challenges (Traditional and Modern)
- Generalization across domains
- Multiple perspectives for EM
- Positive labels scarcity (few true matches vs. huge non-match space)
- Interpretability
- Scale

### Opportunities — Do LLMs Completely Solve the EM Task?
- Few-shot and zero-shot EM with LLMs
- Prompt engineering and instruction tuning
- Human/LLMs in the EM loop
- Multimodal EM (e.g., matching 3D building models / CityJSON geometry, not just text)
