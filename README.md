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

## Features

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
    <img src="https://skillicons.dev/icons?i=python,sklearn&theme=dark" />
  </a>
</p>

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | Scikit-learn |
| NLP | NLTK (tokenization, stemming, lemmatization) |
| Feature Extraction | TF-IDF Vectorizer |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Algorithms | Naive Bayes, Logistic Regression, SVM, Random Forest |

---

## Getting Started

### Prerequisites

- Python >= 3.10 & pip
- Jupyter Notebook or JupyterLab (optional, for notebooks)

### Installation

```bash
# Clone the repository
git clone https://github.com/hasib61714/Fake-News-Detector.git
cd Fake-News-Detector

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Usage

```bash
# Run training and evaluation
python src/train.py

# Predict on a custom article
python src/predict.py --text "Your news article text here..."

# Or explore via Jupyter Notebook
jupyter notebook notebooks/
```

---

## Project Structure

```
Fake-News-Detector/
├── data/
│   ├── True.csv
│   └── Fake.csv
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
└── requirements.txt
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
