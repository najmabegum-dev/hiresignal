# HireSignal

**Multi-signal AI candidate ranking engine — built for India Runs Track 1 (Redrob AI Hackathon)**

HireSignal takes a job description and a pool of 100,000 candidate profiles and produces an explainable, ranked shortlist of the top 100 — without relying on simple keyword matching. It combines semantic search, multi-signal scoring, honeypot detection, and cross-encoder reranking into a single pipeline.

> **Compute disclosure:** this submission's ranking was produced using a GPU (Google Colab T4) and requires network access on first run to download model weights from HuggingFace (cached locally afterward). This does not fully meet the challenge's stated CPU-only / no-network / 5-minute constraints. I attempted a CPU-compliant two-stage architecture (see **Engineering Journey** below) and ran out of time to validate it before the deadline. I chose to submit a working, transparently-documented result rather than an untested one. See `submission_metadata.yaml` for full compute details.

---

## What it does

Given a JD and a raw candidate dataset, HireSignal:

1. Retrieves the most semantically relevant candidates using embedding-based search
2. Scores them across four independent signals — skills, career trajectory, behavioral activity, and education
3. Detects and penalizes fake/trap ("honeypot") profiles before they can pollute the ranking
4. Reranks the strongest contenders with a cross-encoder for higher precision
5. Generates a short, factual explanation for every ranked candidate — no hallucinated justifications, every reason is derived directly from the candidate's actual record

The result is a ranked list where every score is traceable to real signals in the data, not a black box.

---

## Architecture

```
                 ┌─────────────────┐
                 │  Job Description │
                 └────────┬─────────┘
                           │
   100,000 candidates      ▼
   ┌──────────┐    ┌──────────────────┐
   │ Load &   │───▶│  Stage 1         │
   │ Flatten  │    │  Semantic Search  │  BGE-Large embeddings + FAISS
   └──────────┘    │  100K → 2,000     │  narrows the pool by meaning,
                    └────────┬─────────┘  not keywords
                             ▼
                    ┌──────────────────┐
                    │  Stage 2          │  4-signal weighted scoring:
                    │  Multi-Signal     │  skill match (45%) · career (25%)
                    │  Scoring + Trap    │  behavioral (20%) · education (10%)
                    │  Detection         │  + honeypot exclusion/capping
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │  Stage 3          │  Cross-encoder reranks the
                    │  Reranking         │  strongest 300 candidates for
                    │  2,000 → 300 →100  │  higher-precision ordering
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │  Stage 4          │  Offline, template-driven —
                    │  Explainable       │  no LLM calls, no hallucination
                    │  Reasoning         │  risk, fully attributable to data
                    └────────┬─────────┘
                             ▼
                  Top 100 ranked candidates
                  (candidate_id, rank, score, reasoning)
```

---

## Why these design choices

**Semantic search before scoring, not after.**
Keyword matching misses candidates who describe the same skill differently (e.g. "vector databases" vs "FAISS" vs "embedding retrieval"). Running BGE-Large + FAISS first means the scoring stage works on a semantically relevant pool, not a keyword-filtered one.

**Four independent signals instead of one blended score.**
Skill match, career trajectory, behavioral activity, and education are weighted separately (`config.yaml`) rather than folded into a single fuzzy "fit score." This keeps the ranking explainable and auditable — anyone can see *why* a candidate ranked where they did, and the weights can be retuned for a different JD without touching code.

**Honeypot detection runs before scoring, not as a filter after.**
Trap/fake profiles get flagged on multiple signals (impossible career timelines, contradictory data, suspicious patterns) and either excluded outright or have their score capped — so they can't accidentally rank highly due to one strong matching field.

**Offline, template-driven reasoning — not an LLM call.**
Every reasoning string is built from the candidate's actual extracted fields. There's no generative model writing justifications, so there's no hallucination risk: the explanation can always be traced back to a real value in the candidate record.

---

## Engineering journey — what actually happened

I'm including this section deliberately. The pipeline didn't work on the first machine I tried it on, and the hackathon's own compute constraints (CPU-only, 5-minute runtime, no network during ranking) turned out to be a genuinely hard target to hit with a transformer-based embedding model at 100K-candidate scale. I think *how* a system gets to "working," including the dead ends, is part of the engineering story — not something to hide.

### Phase 1 — Getting the pipeline running at all

| Problem hit | Root cause | Fix |
|---|---|---|
| `RuntimeError: NumPy was built with baseline optimizations (X86_V2) but your machine doesn't support it` | Default `numpy` wheel used CPU instructions not supported by the development laptop's processor | Pinned `numpy==1.26.4`, which uses a more conservative baseline |
| `ModuleNotFoundError: No module named 'sentence_transformers'` immediately after the numpy fix | Reinstalling `sentence-transformers` pulled in its own numpy dependency and silently upgraded numpy back to a version incompatible with the CPU — the two fixes kept undoing each other | Installed both packages together in a single resolved command instead of sequentially |
| Pipeline appeared to hang indefinitely at the embedding stage, no error, no crash | The development laptop had only **4 GB of RAM** — far below what's needed to hold BGE-Large, PyTorch, FAISS, and 100,000 candidate records in memory simultaneously. Windows was paging to disk, which looks like a hang but is actually severe memory thrashing | Migrated execution to a cloud notebook (Google Colab) with adequate RAM |
| Embedding 100,000 candidates projected to take **~83 hours** on Colab's free CPU runtime | `sentence-transformers` defaults to CPU unless a device is explicitly passed; the model was never told a GPU was available | Patched `src/embeddings.py` to detect and use CUDA when available, then switched the Colab runtime to a T4 GPU — full embedding stage dropped to well under two hours |
| `validate_submission.py` failed with 7 tie-break errors (equal-score rows not in ascending `candidate_id` order) | The scoring stage sorts by score, but doesn't apply a secondary sort key for exact ties | Re-sorted output by `(score descending, candidate_id ascending)` and reassigned ranks before writing the final file |

### Phase 2 — Trying to meet the actual compute constraints (CPU-only, 5 minutes, no network)

Once I read the full submission spec closely, I realized my working GPU pipeline didn't actually satisfy three hard constraints: CPU-only execution, a 5-minute runtime budget, and zero network access during ranking. The GPU run above took ~97 minutes and required live calls to HuggingFace to fetch model weights. I treated this as a real problem to solve, not a footnote.

**My first instinct — and why I rejected it:** the obvious shortcut is to pre-filter the 100,000 candidates down to a small pool using keyword matching before running any embedding model, so the expensive model only has to process a few thousand candidates. I didn't do this. The job description itself explicitly warns against exactly this approach: *"The 'right answer' to this JD is not 'find candidates whose skills section contains the most AI keywords.' That's a trap we've explicitly built into the dataset."* A candidate who describes building "a recommendation system" without using the word "embeddings" would be unfairly excluded by a keyword filter before ever being evaluated properly. I considered this a real constraint on my solution space, not just a nice-to-have.

**What I built instead — a two-stage semantic retrieval architecture:**
1. **Stage 0 (new):** a much smaller, faster embedding model (BGE-small) does a cheap semantic pass over all 100,000 candidates using a short text summary (headline, current title, top skills, most recent role), narrowing the pool to a generous shortlist of 5,000. This is still meaning-based matching, not keyword matching — it preserves the "substance over wording" reasoning the JD asks for, just with a faster, smaller model for the first pass.
2. **Stage 1 (existing):** BGE-large then runs its full, detailed embedding + FAISS retrieval only on that 5,000-candidate shortlist instead of the full 100,000 — a 20x reduction in workload for the expensive model.

This is the standard two-stage retrieval pattern used in real production search and recommendation systems: a cheap, high-recall first pass followed by an expensive, high-precision second pass. I'm confident this is the right architecture for the constraint, and I'd build it this way even outside a hackathon deadline.

**Where I got stuck, honestly:** in the time remaining before submission, I could not get this architecture to run fast enough on Google Colab's free CPU runtime to validate it actually hits the 5-minute target. BGE-small — a model that benchmarks elsewhere at hundreds of embeddings per second — ran at roughly 5-6 seconds *per batch* of 128-256 short text snippets in my Colab CPU sessions, 10-50x slower than published benchmarks suggested it should. I tried:
- Reducing batch size (256 → 16) — no meaningful improvement, ruling out a batch-size/memory issue
- Explicitly forcing PyTorch to use all available CPU threads (`torch.set_num_threads`) — no measurable change, ruling out a thread-starvation issue
- Confirming the correct model architecture actually loaded (verified `BertModel`, 384-dim output — BGE-small's correct signature, not an accidental fallback to a larger model)

I was not able to root-cause this within my remaining time — my working theory is that Colab's free-tier CPU allocation is more throttled or shared than its advertised specs suggest, but I could not confirm this conclusively before the deadline.

**My final decision:** rather than submit an untested architecture that might not even run, or miss the deadline entirely while continuing to debug, I made the deliberate call to submit using my validated, working GPU-based pipeline — disclosed honestly in `submission_metadata.yaml` as using GPU inference and requiring network access during the first run (to fetch model weights, which are then cached locally). I believe a working, transparently-documented result that doesn't fully meet one constraint is more valuable than an unvalidated one that might meet it, or no submission at all. The two-stage architecture above remains in this repository (`src/fast_filter.py`) as evidence of the approach I attempted, even though it isn't what produced the final submitted ranking.



---

## Project structure

```
hiresignal/
├── rank.py                      # Main entrypoint
├── config.yaml                  # All tunable weights, thresholds, JD settings
├── requirements.txt
├── src/
│   ├── loader.py                 # Loads & flattens candidate JSONL/JSON
│   ├── embeddings.py              # BGE-Large + FAISS semantic retrieval
│   ├── fast_filter.py             # Two-stage retrieval pre-filter (BGE-small) — see Engineering Journey; not in the active pipeline path for this submission
│   ├── scorer.py                  # 4-signal weighted scoring
│   ├── honeypot.py                # Trap-profile detection
│   └── reasoning.py               # Offline, template-based explanations
├── tests/
│   └── test_all.py                # Unit tests
├── data/
│   ├── job_description.txt        # JD used for this submission
│   └── sample_candidates.json     # 50-candidate sample for quick testing
├── validate_submission.py        # Official format validator
└── output/                        # Generated ranked output lands here
```

> **Note on the dataset:** the full `candidates.jsonl` (100,000 records, ~465 MB) is not included in this repository due to GitHub's file size limits. Place the hackathon-provided `candidates.jsonl` in `data/` before running the pipeline — see Setup below.

---

## Setup & running it yourself

```bash
# 1. Clone and enter the repo
git clone https://github.com/najmabegum-dev/hiresignal.git
cd hiresignal

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
#    Copy the hackathon-provided candidates.jsonl into data/

# 5. Run on the full dataset
python rank.py --candidates data/candidates.jsonl --out output/submission.csv

# 6. Validate the output
python validate_submission.py output/submission.csv
```

**Quick test on the 50-candidate sample** (won't produce a valid 100-row submission, but confirms the pipeline runs):
```bash
python rank.py --candidates data/sample_candidates.json --out output/test.csv --sample
```

**Hardware note:** the embedding stage is the heaviest part of the pipeline. I recommend at least 8 GB of RAM, or a GPU-enabled environment (e.g. Google Colab's free T4 runtime) for a reasonable runtime on the full 100K dataset — see the Engineering Journey section above for why.

---

## Configuration

All scoring behavior is tunable in `config.yaml` without touching code:

| Setting | Default | Purpose |
|---|---|---|
| `weights.skill_match` | 0.45 | Skill overlap with JD requirements |
| `weights.career` | 0.25 | Years of experience fit, progression, industry relevance |
| `weights.behavioral` | 0.20 | Platform activity signals (recency, responsiveness, GitHub activity, etc.) |
| `weights.education` | 0.10 | Institution tier bonus |
| `honeypot.exclude_threshold` | 2 | Flags required to exclude a candidate outright |
| `honeypot.single_flag_cap` | 0.40 | Score ceiling for a candidate with exactly one flag |
| `pipeline.top_k_semantic` | 2000 | Candidates retained after semantic retrieval |
| `pipeline.top_k_rerank` | 300 | Candidates passed into the cross-encoder reranker |

---

## Output format

`output/submission.csv` contains exactly 100 rows and 4 columns:

```
candidate_id,rank,score,reasoning
CAND_0002025,1,0.7232,"Senior AI Engineer with 5.9 yrs at Apple..."
CAND_0018499,2,0.6796,"Senior Machine Learning Engineer with 7.2 yrs at Zomato..."
...
```

Scores are non-increasing; ties are broken by ascending `candidate_id`.

---

## Built with

Python · sentence-transformers (BGE-Large) · FAISS · cross-encoder (MiniLM) · pandas · scikit-learn

AI assistance (Claude) was used for architectural discussion, debugging guidance, and code review throughout development. All scoring logic, honeypot detection rules, and final implementation decisions were made and verified by me. No candidate data was ever sent to an external LLM — the ranking pipeline itself runs without any API calls.