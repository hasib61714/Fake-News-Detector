<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:6E40C9,50:A855F7,100:06B6D4&height=200&section=header&text=Fake%20News%20Detector&fontSize=48&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=NLP-Based%20Machine%20Learning%20News%20Classifier&descAlignY=58&descSize=18" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/hasib61714/Fake-News-Detector/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT" />
  </a>
  <a href="https://github.com/hasib61714/Fake-News-Detector/stargazers">
    <img src="https://img.shields.io/github/stars/hasib61714/Fake-News-Detector?style=for-the-badge&color=A855F7" alt="Stars" />
  </a>
  <a href="https://github.com/hasib61714/Fake-News-Detector/network/members">
    <img src="https://img.shields.io/github/forks/hasib61714/Fake-News-Detector?style=for-the-badge&color=06B6D4" alt="Forks" />
  </a>
</p>

---

## About

**Fake News Detection System** is an NLP-driven machine learning classifier that determines whether a news article is **Real** or **Fake**. The system applies a full text preprocessing pipeline — tokenization, stop-word removal, and stemming/lemmatization — followed by TF-IDF feature extraction. Four classification algorithms were trained and rigorously compared to identify the most reliable model for real-world misinformation detection.

---

## Status

This repository is a **monorepo foundation with a working end-to-end MVP**. The target feature set is listed under [Features](#features); this table shows what exists today.

| Area | State |
|---|---|
| Web UI (article input, loading/error states, results panel) | Implemented |
| FastAPI backend (`/api/v1/health`, `/api/v1/analyze`) | Implemented |
| Dataset validation for `Fake.csv` / `True.csv`, TF-IDF + Logistic Regression vs Linear SVM comparison (training-only cross-validation), isolated test evaluation, validation-derived "uncertain" rule, versioned artifact (`ml/`) | Implemented — **no dataset or trained model is included, and no metrics have been produced yet** |
| "Model not trained yet" handling | Implemented |
| Demo mode (fixed sample output, separate from inference) | Implemented, off by default |
| NLTK preprocessing, Naive Bayes / Random Forest, hyperparameter tuning, ROC curves, notebooks | Not yet |
| MySQL + SQLAlchemy + Alembic persistence | Not yet (planned) |
| Authentication, external fact-checking, admin features | Out of scope for now |

Until you train a model (see [Train a model](#4-train-a-model)), the analyze endpoint answers **503 "Model not trained yet"** and the UI says so. It never guesses. The real-dataset training run and evaluation are still to be done by you; see [`ml/README.md`](ml/README.md).

## Features

> Target scope. Not every item below is implemented yet; see [Status](#status).

- **Binary Classification** — Instantly labels news articles as Real or Fake with confidence scores
- **NLP Preprocessing Pipeline** — Tokenization, stop-word removal, stemming, and lemmatization
- **TF-IDF Feature Extraction** — Converts raw text into meaningful numerical representations
- **Multi-Algorithm Comparison** — Naive Bayes, Logistic Regression, SVM, and Random Forest benchmarked
- **Text Cleaning** — Removes HTML tags, punctuation, URLs, and irrelevant noise from raw input
- **Comprehensive Evaluation** — Confusion matrix, F1 score, precision, recall, and ROC curve analysis
- **Reproducible Experiments** — Fixed random seeds and structured notebooks for full reproducibility

---

## Tech Stack

<p align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=python,fastapi,sklearn,nextjs,ts,tailwind&theme=dark" />
  </a>
</p>

| Layer | Technology |
|---|---|
| Frontend | Next.js (App Router), React, TypeScript, Tailwind CSS |
| Backend | Python 3.10+, FastAPI, Pydantic |
| ML | scikit-learn (TF-IDF with Logistic Regression and Linear SVM), joblib |
| Tests | Pytest (backend + ML), Vitest + Testing Library (frontend) |
| Quality | Ruff, mypy (strict), ESLint, Prettier, TypeScript `tsc` |
| Persistence (later) | MySQL, SQLAlchemy, Alembic |

---

## Getting Started

### Prerequisites

- Python >= 3.10 (with `venv`)
- Node.js >= 20.9 (22 recommended, see `.nvmrc`) and npm
- `make` (Linux/macOS; on Windows use WSL)

### 1. Install

```bash
git clone https://github.com/hasib61714/Fake-News-Detector.git
cd Fake-News-Detector
make setup        # creates .venv, installs API + ML packages, runs `npm ci` in apps/web
```

Python dependencies are pinned by `constraints.txt`; JavaScript by `apps/web/package-lock.json`.

### 2. Configure environment

```bash
cp services/api/.env.example services/api/.env
cp apps/web/.env.example apps/web/.env.local
```

Both files contain only non-secret defaults that work for local development.

| Variable | Where | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `apps/web/.env.local` | Backend URL the **browser** calls (default `http://localhost:8000`). Inlined at build time. |
| `CORS_ORIGINS` | `services/api/.env` | Comma-separated origins allowed to call the API (default `http://localhost:3000`). |
| `MODEL_PATH` | `services/api/.env` | Location of the active model artifact (default `ml/artifacts/model.joblib`; its `model.metadata.json` must sit next to it). |
| `ENABLE_DEMO_MODE` | `services/api/.env` | Exposes the clearly labeled demo endpoint. Default `false`. |
| `APP_ENV` | `services/api/.env` | `development` / `production`. |

### 3. Run

```bash
make dev          # API on http://localhost:8000 and web on http://localhost:3000
# or in two terminals:
make api
make web
```

Open <http://localhost:3000>. Interactive API docs are at <http://localhost:8000/docs>.

At this point the status pill reads **"model not trained yet"**, and pressing Analyze shows the "Model not trained yet" message. That is expected.

### 4. Train a model

No dataset is bundled. Download `Fake.csv` and `True.csv` yourself (for example the Kaggle "Fake and Real News Dataset"; check its license) and place them in `ml/data/`. Full instructions, the validation rules and the training procedure are in [`ml/README.md`](ml/README.md).

```bash
make validate-data     # check ml/data/Fake.csv and ml/data/True.csv, print the validation report
make train             # validate, compare models, derive the uncertainty rule, evaluate, save
make report            # re-print the recorded evaluation of the active model
```

Training splits the data into train / validation / test. Logistic Regression and Linear SVM are compared with cross-validation on the training split only, the "uncertain" confidence threshold is derived on the validation split, and the test split is scored once at the end. The printed report and `ml/artifacts/model.metadata.json` contain precision, recall, F1, the confusion matrix and the classification report. Both are git-ignored.

The artifact is versioned under `ml/artifacts/releases/<version>/` and copied to `ml/artifacts/model.joblib` (the file the API loads). It contains the preprocessing and the model together; the API verifies its checksum and preprocessing version when loading. A running API reloads the model automatically when the files change, so no restart is needed.

Other input forms: `make train FAKE=path/Fake.csv REAL=path/True.csv`, `make train DATA=path/articles.csv` (columns `text`, `label`).

### Demo mode (optional)

Set `ENABLE_DEMO_MODE=true` in `services/api/.env` and restart the API. The UI then shows a **Preview demo result** button that calls `POST /api/v1/demo/analyze`. It returns a fixed, clearly labeled sample payload for previewing the interface. It lives in its own module (`services/api/app/demo/`), never loads or calls the model, and does not affect `/api/v1/analyze`.

### Quality checks

```bash
make check        # lint + type checks + all tests + production web build
```

Individual targets: `make lint`, `make typecheck`, `make test`, `make test-api`, `make test-ml`, `make test-web`, `make format`. Run `make help` for the full list. The same checks run in CI (`.github/workflows/ci.yml`).

---

## API

Base path: `/api/v1`. Errors share one shape: `{"error": {"code": "...", "message": "...", "fields": [...]}}`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Always `200` while the service is up. `model.status` is `ready`, `not_trained` or `error`. |
| `POST` | `/analyze` | Body `{"text": "<article>"}` (50 to 20,000 characters after trimming). Returns `label` (`fake`, `real`, or `uncertain` when confidence is below the model's validation-derived threshold), `top_label` (the higher-scoring class even when uncertain), `confidence`, `probabilities`, `uncertainty_threshold`, model name/version and a disclaimer. |
| `POST` | `/demo/analyze` | Only mounted when `ENABLE_DEMO_MODE=true`. Fixed sample response with `mode: "demo"`. |

`POST /analyze` status codes: `200` success, `422` `invalid_request`, `503` `model_not_trained` ("Model not trained yet…") or `model_load_failed` (artifact missing its metadata, checksum or preprocessing-version mismatch, unreadable file; the API refuses it rather than guessing).

```bash
curl -s -X POST http://localhost:8000/api/v1/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "Paste an article of at least fifty characters here to analyze it."}'
```

---

## Project Structure

```
Fake-News-Detector/
├── apps/web/                  # Next.js (App Router) + TypeScript + Tailwind
│   ├── src/app/               #   layout, page, global styles
│   ├── src/components/        #   Analyzer (form + state), ResultPanel, ApiStatus
│   └── src/lib/               #   API client, env config, input validation (+ tests)
├── services/api/              # FastAPI backend
│   ├── app/api/routes/        #   health, analyze, demo (demo is separate from inference)
│   ├── app/inference/         #   ModelService: verified load + predict, uncertain handling
│   ├── app/schemas.py         #   typed request/response models
│   └── tests/                 #   pytest
├── ml/                        # Training code (scikit-learn); datasets/artifacts are git-ignored
│   ├── fakenews_ml/           #   data validation, preprocessing, models, selection, uncertainty,
│   │                          #   evaluation, artifacts, training CLI
│   ├── data/                  #   put Fake.csv and True.csv here (git-ignored)
│   └── tests/
├── .github/workflows/ci.yml   # lint, type check, test, build
├── constraints.txt            # pinned Python dependency versions
├── Makefile                   # setup / run / train / check workflow
└── README.md
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:06B6D4,50:A855F7,100:6E40C9&height=120&section=footer" width="100%" />
</p>

<p align="center">
  Made with dedication by <a href="https://github.com/hasib61714">Md. Hasibul Hasan</a>
</p>
