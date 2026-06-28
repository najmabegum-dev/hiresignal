# HireSignal
### Multi-Signal AI Candidate Ranking — India Runs Track 1

**Redrob AI Hackathon | Senior AI Engineer JD | 100K candidate pool**

---

## Reproduce the submission (one command)

```bash
pip install -r requirements.txt
python rank.py --candidates ./data/candidates.jsonl.gz --out ./submission.csv
```

Runs in < 5 minutes on CPU, 16 GB RAM, no internet connection required.

---

## Architecture

```
candidates.jsonl.gz (100K)
         │
         ▼
┌─────────────────────────┐
│  Stage 1: Semantic      │  BGE-Large embeddings + FAISS
│  Retrieval              │  100K → 2,000 candidates
└──────────┬──────────────┘  (cached after first run)
           │
           ▼
┌─────────────────────────┐
│  Stage 2: Honeypot      │  10 flag checks:
│  Detection              │  all-expert skills, impossible
└──────────┬──────────────┘  timelines, triple-perfect signals
           │  excluded candidates removed
           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Multi-Signal Scoring                          │
│                                                         │
│  skill_match (0.45)   ← proficiency × duration × JD    │
│  career      (0.25)   ← YoE fit, progression, industry │
│  behavioral  (0.20)   ← 8 Redrob signals               │
│  education   (0.10)   ← institution tier                │
│                                                         │
│  composite = Σ(weight × component_score)               │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  Stage 3: Cross-encoder │  MiniLM reranker on top-300
│  Reranking              │  60% reranker + 40% composite
└──────────┬──────────────┘
           │  top 100
           ▼
┌─────────────────────────┐
│  Stage 4: Reasoning     │  Offline template engine
│  Generation             │  Specific facts, honest concerns
└──────────┬──────────────┘  Varies by rank tier
           ▼
    submission.csv
    (100 rows, 4 columns)
```

---

## Design decisions

**Why BGE-Large for embeddings?**
BGE-Large-en-v1.5 leads MTEB for asymmetric retrieval (short query → long doc), exactly the JD-to-profile task. Handles nuanced semantic overlap that TF-IDF and BM25 miss.

**Why multi-signal scoring instead of pure embeddings?**
Embedding similarity ranks by skill text overlap. It can't tell if the candidate is actively looking, has a 150-day notice period, or hasn't logged in for 8 months. The Redrob signals directly address hirability — not just fit on paper.

**Why honeypot detection before scoring?**
The spec says >10% honeypots in top 100 = disqualification at Stage 3. Detecting them beforehand costs nothing and prevents systematic errors from pulling implausible profiles up the ranking.

**Why skill proficiency × duration weighting?**
The JD explicitly says: "we care that you've handled embedding drift, index refresh, retrieval-quality regression in production." Someone with Pinecone marked expert and 88 months duration is categorically different from someone with Pinecone marked beginner and 2 months. Proficiency × duration captures this.

**Why the 5-9 YoE band scoring?**
The JD says this explicitly. Under-experienced gets partial credit. Over-experienced (>10 yrs) gets a slight penalty — the JD explicitly mentions they've had bad experiences with over-senior people who haven't written production code recently.

**What I would improve with more time:**
- Learn-to-rank (LambdaMART) trained on recruiter click data
- Skills ontology graph (PyTorch → Deep Learning → Machine Learning hierarchy)
- Candidate-JD salary compatibility filter (high salary expectations vs role band)
- Fine-tune the cross-encoder on recruiting preference pairs

---

## File structure

```
hiresignal/
├── rank.py                  ← single entrypoint
├── config.yaml              ← all tunable parameters
├── requirements.txt
├── submission_metadata.yaml ← portal metadata
├── README.md
├── HOW_TO_RUN.md            ← step-by-step setup guide
├── validate_submission.py   ← from hackathon bundle
├── src/
│   ├── loader.py            ← handles .jsonl.gz / .json
│   ├── scorer.py            ← multi-signal scoring
│   ├── honeypot.py          ← trap detection (10 checks)
│   ├── embeddings.py        ← BGE + FAISS
│   └── reasoning.py        ← offline reasoning generation
├── tests/
│   └── test_all.py          ← unit tests
├── data/
│   ├── candidates.jsonl.gz  ← dataset (not in repo)
│   └── job_description.txt  ← JD text
└── output/
    └── submission.csv       ← final ranked output
```

---

## Run tests

```bash
pytest tests/ -v
```

---

## Tech stack

| Component | Technology |
|---|---|
| Embeddings | BAAI/bge-large-en-v1.5 |
| Vector search | FAISS (faiss-cpu) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Data loading | gzip + json (stdlib only) |
| Scoring | Pure Python + numpy |
| Validation | validate_submission.py (from bundle) |
