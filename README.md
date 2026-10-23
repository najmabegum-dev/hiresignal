# HireSignal

**Explainable multi-signal candidate ranking: 100,000 profiles → a ranked, justified top 100.**
Built solo for India Runs Track 1 (Redrob AI Hackathon).

HireSignal ranks a candidate pool against a job description using semantic retrieval,
four independent scoring signals, honeypot (trap-profile) detection, and a cross-encoder
reranker. Every ranked candidate gets a short reason built from fields in their own record.

---

## Results and honest runtime

| Scenario | Hardware | Time |
|---|---|---|
| Ranking with a **warm embedding cache** (what `rank.py` does) | Colab CPU, 2 vCPUs, 12 GB RAM | **~2 min** (119–146 s across runs) |
| Same, on GPU | Colab T4 | ~82 s |
| **One-time embedding** of 100K candidates (BGE-Large), cold | CPU, 2 vCPUs | **~2 h 20 min** (projected from 2.8 s/batch × 3,125 batches; the run was interrupted) |
| One-time embedding, cold | Colab T4 | ~97 min |

**What this means:** the 5-minute ranking step is only reachable with pre-computed embeddings.
Embedding is a separate, documented pre-computation step (`precompute.py`); the cache built
from the full pool is published as a [release asset](https://github.com/najmabegum-dev/hiresignal/releases/tag/v1.0-precomputed).
A fresh clone **without** the cache cannot rank 100K candidates within 5 minutes on CPU.

Other measured facts:
- Ranking runs with **zero network calls** when models are cached and `HF_HUB_OFFLINE=1` is set (verified: 0 HTTP requests in the run log).
- The published cache was verified end to end from a fresh clone: downloaded from the release, loaded with a spot-check cosine of 1.0000, and produced a submission that passes `validate_submission.py`.
- Honeypot stage on the final run: 0 candidates excluded, 20 score-capped among the 2,000 scored. None of the final top 100 carries any honeypot flag from my detector. The organizers' hidden honeypot list may differ from my detector's flags.
- Recency features (days since last active) are measured against a fixed reference date: the latest `last_active_date` in the data (2026-05-27), not the system clock. Earlier versions used `date.today()`, so scores drifted depending on the day the code ran.
- Bit-for-bit reproducibility across different library versions has not been verified. Dependencies are pinned in `requirements.txt` (tested on Python 3.13 in Google Colab).

---

## Pipeline

```
Job description + 100,000 candidates
        │
Stage 1  Semantic retrieval   BGE-Large embeddings + FAISS: 100K → 2,000
Stage 2  Multi-signal scoring skills 45% · career 25% · behavioral 20% · education 10%
                              + honeypot exclusion / score capping
Stage 3  Cross-encoder rerank MiniLM cross-encoder on the top 300
Stage 4  Reasoning            template-based, no LLM calls, built from record fields
        │
Top 100: candidate_id, rank, score, reasoning
```

**Design choices**
- **Semantic search before scoring.** Keyword matching misses people who describe the same skill differently ("vector databases" vs "FAISS"); the JD itself warns against keyword filtering.
- **Four separate signals,** weighted in `config.yaml`, so each ranking is auditable and retunable without code changes.
- **Honeypot checks run before scoring** so a single strong field can't carry a fake profile upward.
- **Template-based reasoning, no generative model,** so no invented facts. Each skill is mapped to a named JD requirement area (vector search / retrieval, embeddings, ranking / recommendation, LLM / RAG, MLOps, NLP), and the text states which areas a candidate covers and which have no listed skills. Phrasing variants are chosen deterministically from the `candidate_id`, so identical input always gives identical output.

---

## Run it

```bash
git clone https://github.com/najmabegum-dev/hiresignal.git && cd hiresignal
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the hackathon's `candidates.jsonl` (or `.jsonl.gz`) in `data/`, then:

**1. (Optional, saves about 2 hours) Fetch the pre-built embedding cache.**
Download `emb_215fd55033507577.npy` from the
[release](https://github.com/najmabegum-dev/hiresignal/releases/tag/v1.0-precomputed)
into `.cache/`, keeping the filename. Skip this step to build the cache yourself in step 2.

**2. Pre-compute (needs network, run once).**
```bash
python precompute.py --candidates data/candidates.jsonl
```
This downloads the model weights, and embeds all candidates **only if no valid cache exists**
(about 2+ hours on a 2-vCPU CPU, about 97 minutes on a T4 GPU). If you fetched the cache in
step 1, this step just verifies it and caches the models.

**3. Rank (works fully offline).**
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python rank.py --candidates data/candidates.jsonl --out output/submission.csv
python validate_submission.py output/submission.csv
```
On Windows PowerShell, set the variables first:
`$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1`.

The cache is keyed by the model name and the candidate texts, and verified on load (shape plus a
spot-check of 16 vectors). A mismatched cache is ignored rather than used silently, which would
trigger a full re-embed.

Quick smoke test on the 50-candidate sample (not a valid 100-row submission):
```bash
python rank.py --candidates data/sample_candidates.json --out output/test.csv --sample
```

---

## Output

`output/submission.csv`: exactly 100 rows, columns `candidate_id,rank,score,reasoning`.
Scores are non-increasing; ties break by ascending `candidate_id`.
A top-10 excerpt from a validated run is in [`output/sample_output_top10.csv`](output/sample_output_top10.csv).

---

## Configuration (`config.yaml`)

| Setting | Default | Purpose |
|---|---|---|
| `weights.skill_match` | 0.45 | Skill overlap with JD requirements |
| `weights.career` | 0.25 | Experience fit, progression, industry relevance |
| `weights.behavioral` | 0.20 | Activity signals (recency, responsiveness, etc.) |
| `weights.education` | 0.10 | Institution tier |
| `honeypot.exclude_threshold` | 2 | Flags needed to exclude a candidate |
| `honeypot.single_flag_cap` | 0.40 | Score ceiling for a candidate with one flag |
| `pipeline.top_k_semantic` | 2000 | Kept after semantic retrieval |
| `pipeline.top_k_rerank` | 300 | Sent to the cross-encoder |

---

## Engineering log (what actually happened)

1. **Environment:** NumPy/CPU-instruction mismatch on my laptop; `sentence-transformers` kept
   upgrading NumPy. Fixed by pinning and installing together. The laptop had 4 GB of RAM, so
   embedding looked like a hang (memory thrashing); I moved to Google Colab.
2. **Speed:** cold BGE-Large embedding on Colab's 2-vCPU CPU was projected at ~2h20m; a T4 GPU
   did it in ~97 min.
3. **Validator failure:** 6–7 tie-break errors. Root cause: scores were sorted at full precision
   but written to 4 decimals, so equal-looking scores weren't in `candidate_id` order. Fixed in
   `rank.py` (round first, then sort by score and `candidate_id`). Earlier I had patched the CSV
   by hand in a notebook; that is now in code.
4. **Cache bug:** the cache key hashed the first 1 MB of the input file, so the same candidates
   as `.jsonl` vs `.jsonl.gz` produced different keys and a silent cold start. The key is now
   derived from the texts being embedded, and the cache is verified on load.
5. **Network:** ranking originally contacted HuggingFace at runtime. Models are now fetched in
   `precompute.py`, and ranking runs offline (0 HTTP requests in the verified run).
6. **Non-deterministic dates:** three modules used `date.today()` for recency, so scores and the
   "last active N days ago" text changed with the run date (runs on different days gave 20 vs 21
   score-capped candidates). Fixed with `src/refdate.py`, which pins the reference date to the
   data. Note: the CSV I submitted to the portal was generated before this fix.
7. **Reasoning quality:** a review against the spec's checks found that low-rank rows shared one
   skeleton and never named a JD requirement. Rewrote `src/reasoning.py` to map skills to JD
   requirement areas, state gaps explicitly, and vary phrasing deterministically. Also fixed a
   substring-matching bug (the term "rag" matched "storage").

## Known limitations

- `src/fast_filter.py` (a BGE-small pre-filter) is a **prototype and is not used by `rank.py`**.
  It was meant to cut cold-start time; I did not finish wiring it in or validate it.
- Cold-start CPU ranking without the cache is far outside 5 minutes.
- Skills carry 45% of the weight, but career, behavioral, and education signals can lift a
  candidate with little skill overlap into the top 100 (for example, a QA engineer with a skill
  score of 0.05 and a pre-rerank composite of 0.51 ranked 64th). I did not retune the weights
  after seeing this.
- No labeled ground truth was available to me during development, so ranking quality (e.g. NDCG)
  was not measured. Checks I did run: format validation, honeypot flags in the top 100, and a
  manual review of sampled reasoning text.
- Reasoning is generated from templates with a few deterministic phrasing variants, so it reads
  more uniformly than hand-written text.
- No hosted sandbox/demo yet.

---
## Troubleshooting

- **`ModuleNotFoundError`:** activate the virtual environment and run `pip install -r requirements.txt`.
- **An error about files missing from the local cache when running offline:** the models were never downloaded. Run `precompute.py` once with network access.
- **Log line "Cache spot-check failed — ignoring cache":** the cache doesn't match your data or model, so `rank.py` will start a multi-hour re-embed. Stop it and check that the cache file and the candidate file are the ones the cache was built from.
- **Validator reports tie-break or ordering errors:** make sure you're on the current `rank.py`, which rounds scores and sorts by score then `candidate_id`.
- 
## Project structure

```
hiresignal/
├── rank.py                 # ranking entrypoint
├── precompute.py           # one-time: cache models + embed candidates
├── config.yaml             # weights, thresholds, models
├── requirements.txt        # pinned dependencies
├── submission_metadata.yaml
├── src/
│   ├── loader.py           # load + flatten candidates
│   ├── embeddings.py       # BGE-Large + FAISS, verified cache
│   ├── fast_filter.py      # prototype pre-filter (not wired in)
│   ├── scorer.py           # 4-signal scoring
│   ├── honeypot.py         # trap-profile detection
│   ├── reasoning.py        # template-based explanations
│   └── refdate.py          # fixed reference date for recency features
├── tests/test_all.py
├── data/                   # JD + 50-candidate sample (full pool not included)
├── output/                 # generated results land here
├── .github/workflows/      # CI
└── validate_submission.py  # official format validator
```

## AI assistance

I used Claude for architecture discussion, debugging, and code review, and Cursor/Antigravity
for editing. Scoring rules, honeypot logic, and design decisions were made and verified by me.
No candidate data was sent to any external LLM, and the ranking step makes no API calls.

## License
MIT
