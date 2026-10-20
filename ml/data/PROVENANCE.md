# Dataset provenance, license and attribution

This file documents the dataset used with this project. It is the only file in `ml/data/` that is
committed. The CSV files are **never** committed (`.gitignore` excludes them), and neither are
trained models derived from them.

> This is a plain-language record for project hygiene, **not legal advice**. If you plan to
> publish a model, share derived data, or use any of this commercially, read the license text and
> consult someone qualified.

## Source

| | |
|---|---|
| Dataset | "Fake and Real News Dataset" (`Fake.csv`, `True.csv`) |
| Published on | Kaggle, by Clément Bisaillon |
| URL | <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset> |
| Downloaded by | Md. Hasibul Hasan (project owner), 2026-10-20, as `Fake.csv.zip` and `True.csv.zip` |
| Underlying data | Appears to be the ISOT Fake News Dataset (University of Victoria): real articles from Reuters.com, fake articles from sites flagged as unreliable, 2015 to 2018 |
| Columns | `title`, `text`, `subject`, `date` (only `text` is used for training) |

## License

**CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International).

* Status: **stated by the project owner as shown on the Kaggle dataset page on 2026-10-20.** It
  has **not** been independently verified by the tooling used to prepare this repository (the
  Kaggle and Creative Commons sites were not reachable from the development environment, and the
  downloaded archives contain no license file).
* Deed: <https://creativecommons.org/licenses/by-nc-sa/4.0/>
  Legal code: <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>
  (linked here; not fetched or quoted by the tooling).

### Verifying the license for the exact downloaded version

Do this once, on your own computer, and fill in the blanks below:

1. Open the dataset page above while signed in, and note the **License** field and the dataset
   **Version** shown on the page.
2. Open the page's **Data** tab and confirm the files are `Fake.csv` and `True.csv`.
3. Check that the SHA-256 hashes of your files match the table below (they identify the exact
   bytes used here): `sha256sum ml/data/*.csv`, or on Windows PowerShell
   `Get-FileHash ml\data\Fake.csv, ml\data\True.csv -Algorithm SHA256`.
4. Record the result:

   * License field on the page: `CC BY-NC-SA 4.0` (confirm) &nbsp; Verified on: `____-__-__`
   * Dataset version shown: `__` &nbsp; Verified by: `__________`

### What the conditions mean here

Summary of the license's three conditions and how this project handles them:

| Condition | Meaning (plain language) | In this repository |
|---|---|---|
| **BY** attribution | Credit the creator, link to the license, and say if you changed the material | Credit and citation below; the changes made are listed below |
| **NC** non-commercial | No use "primarily intended for or directed toward commercial advantage or monetary compensation" | **No commercial use.** Do not use the dataset, or models trained on it, in a paid product, service or commercial deployment |
| **SA** share-alike | If you remix or adapt the material, share your contribution under the same (or a compatible) license | Do not republish the data or derived datasets under other terms (see the model caveat below) |

Do **not** assume this permits commercial use. It does not.

### Caveats to be aware of

* **Models trained on the data.** Whether a trained model counts as "adapted material" under
  share-alike is not settled. The conservative policy for this project: treat any trained model,
  its metadata and any derived dataset as subject to the same non-commercial, share-alike terms;
  keep them out of Git (already enforced by `.gitignore`); do not publish or distribute them or
  use them commercially without getting advice.
* **The repository's MIT license covers the code only.** It does not apply to the dataset, to
  models trained on it, or to evaluation outputs derived from it.
* **Upstream rights.** The articles are third-party news content (for example Reuters articles).
  A license label on a Kaggle page may not grant rights the uploader does not hold. If you need
  certainty about reuse beyond private research and evaluation, check the original publishers'
  terms and the ISOT project's own terms (<https://www.uvic.ca/engineering/ece/isot/>).

## Attribution and citation

Please keep this attribution wherever results or derived material from this dataset are shown:

> Dataset: "Fake and Real News Dataset" by Clément Bisaillon, Kaggle,
> <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>, licensed under
> CC BY-NC-SA 4.0 (as stated on the dataset page; verify there). Changes made: see below.

Original research on the underlying ISOT data (verify the preferred citation on the ISOT project
page before citing):

> H. Ahmed, I. Traore, S. Saad. "Detection of Online Fake News Using N-Gram Analysis and Machine
> Learning Techniques." In: *Intelligent, Secure, and Dependable Systems in Distributed and Cloud
> Environments*, Lecture Notes in Computer Science, 2017, pp. 127-138.
> DOI: 10.1007/978-3-319-69155-8_9

## Exact files used

| File | SHA-256 | Size (bytes) | Rows read |
|---|---|---|---|
| `Fake.csv` | `bebf8bcfe95678bf2c732bf413a2ce5f621af0102c82bf08083b2e5d3c693d0c` | 62,789,876 | 23,481 |
| `True.csv` | `ba0844414a65dc6ae7402b8eee5306da24b6b56488d6767135af466c7dcb2775` | 53,582,940 | 21,417 |

Both are UTF-8 CSV files with the header `title,text,subject,date`. The archives they were
extracted from were `Fake.csv.zip` (23,982,555 bytes) and `True.csv.zip` (18,993,378 bytes).

## Changes made to the data (for the BY "indicate changes" condition)

The CSV files themselves are never modified. At training time, `fakenews_ml` (see `ml/README.md`):

1. Drops empty rows and rows under 50 characters after cleaning.
2. Drops duplicate articles (compared case-insensitively after cleaning) and any article that
   appears in both files.
3. Removes publisher and image-embed boilerplate from the text as part of preprocessing
   **version 2** (leading wire datelines, Reuters credit lines and sponsored-content footers,
   trailing image credits, tweet-embed links and attributions, 21st Century Wire promotion).
4. Splits the result into train, validation and test sets with a fixed seed.

Outcome of validation (`make validate-data`) with preprocessing version 2 on the files above:
17,292 fake and 21,189 real articles kept (38,481 total); dataset fingerprint
`509386809ca6b554…`. These are counts only; no article text is stored in this repository.

## Known data issues

The two classes come from different sources, which creates shortcut signals a model can exploit.
Details, measurements and what is and is not removed are in the "Remaining leakage" section of
`ml/README.md`. Treat any metric on this dataset as optimistic.
