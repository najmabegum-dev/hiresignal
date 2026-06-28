# HireSignal — Complete Step-by-Step Guide
## India Runs Track 1 | Antigravity Edition

---

## BEFORE YOU START — Read this

The real dataset has strict rules. Your submission MUST:
- Rank exactly **100 candidates** (not 20, not 50 — exactly 100)
- Have exactly **4 columns**: `candidate_id, rank, score, reasoning`
- Have scores **non-increasing** (rank 1 = highest score)
- Use candidate IDs exactly as they appear: `CAND_0000001` format
- Run in **under 5 minutes** on CPU with **NO internet** during ranking
- Pass `validate_submission.py` before uploading

---

## STEP 1 — Open Antigravity and create the project folder

In Antigravity, open a terminal and run:

```
mkdir hiresignal
cd hiresignal
```

Then open this folder as your project in Antigravity.

---

## STEP 2 — Copy all files into the project

Take every file from the zip I gave you and put them inside the `hiresignal` folder.
The structure should look like this:

```
hiresignal/
├── rank.py                        ← main file you run
├── src/
│   ├── loader.py                  ← loads the dataset
│   ├── scorer.py                  ← scores each candidate
│   ├── signals.py                 ← behavioral signals
│   ├── honeypot.py                ← trap detection
│   └── reasoning.py               ← generates reasoning text
├── data/
│   ├── candidates.jsonl.gz        ← PUT THE REAL DATASET HERE
│   └── job_description.txt        ← PUT THE JD TEXT HERE
├── output/                        ← submission CSV will appear here
├── tests/
│   └── test_all.py
├── config.yaml
├── requirements.txt
├── submission_metadata.yaml       ← fill this in before submitting
├── README.md
└── validate_submission.py         ← copy from the hackathon bundle
```

---

## STEP 3 — Put the real dataset files in the data/ folder

From the hackathon zip you downloaded, copy:
- `candidates.jsonl.gz` → paste into `hiresignal/data/`
- The job description text → save as `hiresignal/data/job_description.txt`

To save the JD as a text file:
- Open `job_description.docx` from the bundle
- Select all text (Ctrl+A), copy it
- Open Notepad, paste, save as `job_description.txt`
- Move that file into `hiresignal/data/`

---

## STEP 4 — Set up Python environment

In your Antigravity terminal:

```bash
python -m venv venv
```

Then activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

You will see `(venv)` appear in your terminal. That means it worked.

Then install all packages:

```bash
pip install -r requirements.txt
```

This takes 5-10 minutes the first time. It will download the BGE embedding model.

---

## STEP 5 — Run a quick test first (50 sample candidates)

Before running on 100,000 candidates, test on the 50 sample candidates:

```bash
python rank.py --candidates data/sample_candidates.json --out output/test_submission.csv --sample
```

You should see:
- Each scoring stage running
- "50 candidates scored"
- "Output saved to output/test_submission.csv"

Open `output/test_submission.csv` and check it looks right — 50 rows, 4 columns, scores decreasing.

---

## STEP 6 — Run on the full 100,000 candidate dataset

```bash
python rank.py --candidates data/candidates.jsonl.gz --out output/submission.csv
```

This takes 3-5 minutes. You will see a progress bar.

When it finishes you will see:
```
✓ Done. 100 candidates ranked.
  Honeypots excluded: X
  Output: output/submission.csv
```

---

## STEP 7 — Validate your submission

Copy `validate_submission.py` from the hackathon bundle into your `hiresignal/` folder, then run:

```bash
python validate_submission.py output/submission.csv
```

It must print:
```
Submission is valid.
```

If it shows errors, see the Troubleshooting section at the bottom.

---

## STEP 8 — Rename the CSV to your team ID

The submission spec says the filename must be your registered participant ID:

```bash
# Replace team_xxx with your actual Hack2Skill participant ID
copy output\submission.csv output\team_xxx.csv
```

---

## STEP 9 — Fill in submission_metadata.yaml

Open `submission_metadata.yaml` in Antigravity and fill in every field:

```yaml
team_name: "your-team-name"
primary_contact:
  name: "Najma Begum"
  email: "your@email.com"
  phone: "+91-XXXXXXXXXX"
github_repo: "https://github.com/YOURUSERNAME/hiresignal"
sandbox_link: "https://huggingface.co/spaces/YOURUSERNAME/hiresignal"
reproduce_command: "python rank.py --candidates ./data/candidates.jsonl.gz --out ./submission.csv"
```

---

## STEP 10 — Push to GitHub

```bash
git init
git add .
git commit -m "HireSignal: Multi-signal candidate ranking for India Runs Track 1"
git remote add origin https://github.com/YOURUSERNAME/hiresignal.git
git push -u origin main
```

Make sure the repo is PUBLIC and `output/submission.csv` is committed.

---

## STEP 11 — Set up a sandbox on HuggingFace Spaces (Required)

The submission form requires a live sandbox link. Easiest option: HuggingFace Spaces.

1. Go to huggingface.co → Sign in → New Space
2. Choose "Streamlit" as the SDK
3. Name it: `hiresignal-ranker`
4. Upload your project files
5. The URL will be: `https://huggingface.co/spaces/YOURUSERNAME/hiresignal-ranker`
6. Use this URL as your sandbox_link

---

## STEP 12 — Submit on Hack2Skill

1. Go to hack2skill.com/event/india_runs/
2. Log in with your registered account
3. Find the Track 1 submission form
4. Upload your `team_xxx.csv` file
5. Fill in all the metadata fields
6. Submit before **28 June 2026**

---

## Troubleshooting

**"Not exactly 100 rows" error from validator**
The dataset may have fewer good candidates than expected. The ranker always outputs exactly 100 — if it outputs less, check that your `data/candidates.jsonl.gz` file is complete (should be ~52 MB).

**"score not non-increasing" error**
This means a lower-ranked candidate has a higher score. Re-run `rank.py` — this is a bug that was fixed. If it persists, tell Antigravity: "In rank.py save_output(), sort by score descending before assigning ranks."

**"candidate_id does not match CAND_ pattern" error**
You may have an extra space or character. Re-run the pipeline — the IDs come directly from the dataset.

**Pipeline takes more than 5 minutes**
Open `config.yaml` and reduce `top_k_rerank` from 200 to 100. This makes it faster with minimal quality loss.

**"Module not found" errors**
Make sure your venv is activated (you see `(venv)` in the terminal) and you ran `pip install -r requirements.txt`.

**Embedding model download fails**
You need internet during setup (not during ranking). Make sure your internet connection is stable when first running `pip install -r requirements.txt`.
