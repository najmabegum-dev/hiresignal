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
| Ranking with a **warm embedding cache** (what `rank.py` does) | Colab CPU, 2 vCPUs, 12 GB RAM | **132–146 s** |
| Same, on GPU | Colab T4 | ~82 s |
| **One-time embedding** of 100K candidates (BGE-Large), cold | CPU, 2 vCPUs | **~2 h 20 min** (projected from 2.8 s/batch × 3,125 batches; the run was interrupted) |
| One-time embedding, cold | Colab T4 | ~97 min |

**What this means:** the 5-minute ranking step is only reachable with pre-computed embeddings.
Embedding is a separate, documented pre-computation step (`precompute.py`); the cache built
from the full pool is published as a [release asset](https://github.com/najmabegum-dev/hiresignal/releases/tag/v1.0-precomputed).
A fresh clone **without** the cache cannot rank 100K candidates within 5 minutes on CPU.

Other measured facts:
- Ranking runs with **zero network calls** when models are cached and `HF_HUB_OFFLINE=1` is set.
- Results can differ slightly between library versions (ranks 1–2 were identical across runs; lower ranks shifted by ~0.001 in score). Exact reproducibility is not guaranteed.
- Honeypot stage: on the final run, 0 candidates were excluded and 21 were score-capped among the 2,000 scored. I have **not** independently measured the honeypot rate inside the final top 100.

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
- **Template-based reasoning.** No generative model, so no invented facts. The trade-off: reasoning is less varied than hand-written text.

---

## Run it

```bash
git clone https://github.com/najmabegum-dev/hiresignal.git && cd hiresignal
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the hackathon's `candidates.jsonl` (or `.jsonl.gz`) in `data/`. Then either:

**A. Use the published cache (fast).** Download the `.npy` file from the
[release](https://github.com/najmabegum-dev/hiresignal/releases/tag/v1.0-precomputed) into `.cache/`, keeping the filename.

**B. Build it yourself (slow, one time).**
```bash
python precompute.py --candidates data/candidates.jsonl     # downloads models + embeds 100K
```

Then rank (works offline once models are cached):
```bash
HF_HUB_OFFLINE=1 python rank.py --candidates data/candidates.jsonl --out output/submission.csv
python validate_submission.py output/submission.csv
```

The cache is keyed by the candidate texts and model name, and verified on load (shape plus a
spot-check of 16 vectors). A mismatched cache is ignored rather than used silently.

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
   `rank.py` (round first, then sort by score and `candidate_id`). Earlier I had patched the CSV by hand in a notebook; that is now in code.
4. **Cache bug:** the cache key hashed the first 1 MB of the input file, so the same candidates
   as `.jsonl` vs `.jsonl.gz` produced different keys and a silent cold start. The key is now
   derived from the texts being embedded, and the cache is verified on load.
5. **Network:** ranking originally contacted HuggingFace at runtime. Models are now fetched in
   `precompute.py`, and ranking runs offline (0 HTTP requests in the verified run).

## Known limitations

- `src/fast_filter.py` (a BGE-small pre-filter) is a **prototype and is not used by `rank.py`**.
  It was meant to cut cold-start time; I did not finish wiring it in or validate it.
- Cold-start CPU ranking without the cache is far outside 5 minutes.
- `requirements.txt` is not yet pinned to exact versions.
- No hosted sandbox/demo yet.
- Reasoning is templated, so it varies less than hand-written text.
- Honeypot rate in the final top 100 is unmeasured.

---

## Project structure

```
hiresignal/
├── rank.py                 # ranking entrypoint
├── precompute.py           # one-time: cache models + embed candidates
├── config.yaml             # weights, thresholds, models
├── requirements.txt
├── src/
│   ├── loader.py           # load + flatten candidates
│   ├── embeddings.py       # BGE-Large + FAISS, verified cache
│   ├── fast_filter.py      # prototype pre-filter (not wired in)
│   ├── scorer.py           # 4-signal scoring
│   ├── honeypot.py         # trap-profile detection
│   └── reasoning.py        # template-based explanations
├── tests/test_all.py
├── data/                   # JD + 50-candidate sample (full pool not included)
├── output/                 # generated results land here
└── validate_submission.py  # official format validator
```

## AI assistance

I used Claude for architecture discussion, debugging, and code review, and Cursor/Antigravity
for editing. Scoring rules, honeypot logic, and design decisions were made and verified by me.
No candidate data was sent to any external LLM, and the ranking step makes no API calls.

## License
MIT
