# Star, Galaxy or Quasar? 🔭

**Teaching a computer to read the night sky.** A machine learning project that classifies objects seen by a telescope as a **star**, a **galaxy** or a **quasar**.

*Lervin De La Rosa · Ironhack Machine Learning project*

## The question

Can a model tell what kind of object a telescope is looking at, using only its measurements?

- **Input:** brightness in 5 color filters, redshift (distance) and sky position
- **Output:** `STAR`, `GALAXY` or `QSO` (quasar)
- **Type:** supervised classification

## Data

[Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) (Kaggle), from the Sloan Digital Sky Survey.

- 100,000 objects, 17 columns, no missing values
- Classes: 59% galaxies, 22% stars, 19% quasars

The dataset is **not stored in this repo**. The notebook downloads it automatically with `kagglehub` the first time it runs (about 7 MB).

## Method

1. **Cleaning:** dropped 9 ID / bookkeeping columns (they describe the telescope's records, not the object) and 1 row with a placeholder brightness of -9999.
2. **Feature engineering:** added 4 colors, the differences between neighbouring filters: `u−g`, `g−r`, `r−i`, `i−z`.
3. **Split:** 80% train / 20% test, stratified so both keep the same class mix.
4. **Models:** a baseline (always guess "galaxy"), Logistic Regression, KNN, Random Forest and Gradient Boosting. Logistic Regression and KNN get a `StandardScaler` inside a pipeline, because they compare feature values directly; without scaling, KNN drops from 94.4% to 83.7%.
5. **Evaluation:** accuracy, plus macro **precision**, **recall** and **F1**. "Macro" averages each score over the 3 classes, so every class counts equally even though the classes are uneven.
   - **Precision:** when the model says a type, how often is it right? TP / (TP + FP)
   - **Recall:** of all real objects of a type, how many did the model find? TP / (TP + FN)
   - **F1:** the balance of the two, 2 × (precision × recall) / (precision + recall)
6. **Tuning:** Grid Search with 5-fold stratified cross-validation on Gradient Boosting.
7. **Experiment:** the same models using **colors only**, without the expensive redshift measurement.

## Results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Baseline (always "galaxy") | 59.4% | 0.198 | 0.333 | 0.249 |
| Logistic Regression | 95.8% | 0.955 | 0.951 | 0.952 |
| KNN (5 neighbours, scaled) | 94.4% | 0.944 | 0.933 | 0.938 |
| **Random Forest** | **98.0%** | **0.980** | **0.974** | **0.976** |
| Gradient Boosting | 97.7% | 0.977 | 0.971 | 0.974 |
| Gradient Boosting, tuned | 97.9% | 0.978 | 0.973 | 0.976 |
| Best model, colors only | 88.6% | 0.862 | 0.846 | 0.853 |

Per class, for the winning Random Forest:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Star | 0.993 | 1.000 | 0.996 |
| Galaxy | 0.979 | 0.987 | 0.983 |
| Quasar | 0.967 | 0.934 | 0.950 |

Quasars have the lowest recall: the model misses about 6% of them, mostly calling them galaxies.

Confusion matrix for the Random Forest (rows = real class, columns = predicted class, 20,000 test objects):

| Real \ Predicted | Galaxy | Quasar | Star |
|---|---|---|---|
| **Galaxy** | **11,737** | 120 | 32 |
| **Quasar** | 252 | **3,540** | 0 |
| **Star** | 1 | 0 | **4,318** |

The diagonal (bold) holds the correct predictions. The biggest error is 252 quasars predicted as galaxies.

## Key findings

- **Distance is the #1 clue.** Redshift accounts for about 49% of the Random Forest's feature importance. The next two are engineered colors (`r−i`, `g−r`).
- **Stars are easy** (4,318 of 4,319 found). The main mistake is **quasars labelled as galaxies** (252 of 3,792), never as stars. The missed quasars are the closer ones (median redshift 0.60 vs 1.64 for all quasars), where quasars and galaxies overlap, and they are redder than typical quasars. A quick test with `class_weight="balanced"` did not help: for the Random Forest it changed almost nothing, and for Gradient Boosting it caught 45 more quasars but mislabelled 72 more galaxies.
- **Tuning Gradient Boosting helped only a little** (macro F1 0.974 → 0.976): the models were already near their limit.
- **Colors only still reaches 88.6%.** That's useful in the real world: a survey could sort millions of cheap photos first, then spend expensive spectroscopy only on the interesting objects.
- **Fun fact:** 63% of stars have a slightly negative redshift, meaning they're moving towards us.

## Limitations and next steps

- Tested on one survey (SDSS); data from other telescopes may behave differently.
- Close the gap for the colors-only model with more features or other models.
- Go deeper: classify types of stars and galaxies.

## Project structure

```
stellar-classification/
├── README.md
├── requirements.txt
├── notebooks/
│   └── stellar_classification.ipynb   # full analysis, charts and results
├── src/
│   ├── data.py       # download and clean the data
│   ├── features.py   # color features, feature/target split
│   └── models.py     # models, evaluation, tuning
├── images/           # space photos used in the slides
└── slides/           # presentation (PDF)
```

## Image credits

The `images/` folder holds the pictures used in the slides. Most come from Wikimedia Commons; the three `sdss-*.jpg` photos are SDSS DR17 image cutouts of objects in this dataset:

| File | Shows | Credit and licence |
|---|---|---|
| `sun.jpg` | The Sun (a star) | NASA/SDO (AIA), public domain |
| `whirlpool-galaxy.jpg` | Whirlpool Galaxy (M51) | NASA & ESA (Hubble), public domain |
| `quasar-3c273.jpg` | Quasar 3C 273 | ESA/Hubble & NASA, CC BY 2.0 |
| `quasar-artist-impression.jpg` | Artist's drawing of quasar ULAS J1120+0641 (not used in the current slides) | ESO/M. Kornmesser, CC BY 4.0 |
| `sombrero-galaxy.jpg` | Sombrero Galaxy (M104) | NASA/ESA and the Hubble Heritage Team (STScI/AURA), public domain |
| `webb-deep-field.jpg` | Webb's First Deep Field | NASA, ESA, CSA and STScI, public domain |
| `sdss-star.jpg` | A star from the dataset (RA 36.915, Dec 24.086) | Sloan Digital Sky Survey (SDSS DR17) |
| `sdss-galaxy.jpg` | A galaxy from the dataset (RA 242.733, Dec 26.437) | Sloan Digital Sky Survey (SDSS DR17) |
| `sdss-quasar.jpg` | A quasar from the dataset (RA 224.957, Dec 53.319) | Sloan Digital Sky Survey (SDSS DR17) |
| `color-map.png` | Chart: stars, galaxies and quasars by color (u−g vs g−r) | Made from this project's data |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/stellar_classification.ipynb
```

Run all cells. The first run downloads the dataset. On Windows with antivirus HTTPS scanning (e.g. Avast), `truststore` fixes the SSL certificate error during the download.
