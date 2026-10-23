
# Handwritten Answer Grading Under Recognition Noise

## Overview

This project investigates how character-level recognition errors affect automated grading of short descriptive answers.

Instead of building a handwriting recognition system, the project focuses on the grading stage: comparing student answers with instructor reference answers and predicting human-assigned marks.

## Objectives

- Develop classical machine learning baselines for answer grading.
- Extract text similarity and answer-length features.
- Simulate character substitutions, insertions, and deletions.
- Measure grading performance under different character error rates.
- Compare grading models using RMSE and quadratic weighted kappa.

## Methodology

1. Prepare and validate the short-answer grading dataset.
2. Split data by question to evaluate generalization to unseen questions.
3. Extract TF-IDF and lexical similarity features.
4. Train baseline regression models.
5. Inject controlled character-level recognition noise.
6. Evaluate grading performance at increasing noise levels.

## Models

- Dummy Regressor
- Support Vector Regression (SVR)
- Random Forest Regressor

## Evaluation Metrics

- Root Mean Squared Error (RMSE)
- Quadratic Weighted Kappa (QWK)
- Character Error Rate (CER)

## Project Structure

```text
src/
  preprocess.py
  split_data.py
  noise.py
  features.py
  train.py
  evaluate.py
  run_experiments.py
data/
models/
results/
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the requirements:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

```bash
python src/preprocess.py
python src/split_data.py
python src/train.py
python src/evaluate.py
python src/run_experiments.py
```

The raw dataset must be downloaded separately and placed at:

`data/raw/mohler_open_ended.csv`

## Research Question

How does increasing character error rate affect automated grading accuracy, and which grading models are most robust to recognition noise?

## Status

Initial baseline implementation: dataset preprocessing, noise injection, feature extraction, classical regression models, and evaluation.

## Data and Attribution

The project uses the Mohler short-answer grading dataset. Follow the original dataset's terms of use and provide appropriate attribution when redistributing or publishing results.
