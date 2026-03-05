# PFO-RS Smart Agriculture Recommendation System

This repository provides a **reproducible implementation** of the research framework described in the manuscript:

**Permutation Flamingo Optimization–Driven Smart Agriculture Recommendation System (PFO-RS)**

The framework integrates **permutation-based Flamingo Optimization for feature selection**, **RNN-LSTM learning**, and **explainable AI techniques (SHAP and LIME)** to generate crop recommendation insights from agro-environmental data.

The repository is designed to be **minimal, clear, and fully reproducible**, allowing reviewers and researchers to replicate training, evaluation, and explainability experiments using a single configuration file.

---

# Repository Structure

```
pfo-rs-smart-agri
│
├── README.md
├── requirements.txt
├── config.yaml
│
├── run_train.py
├── run_eval.py
├── run_explain.py
│
└── src
    ├── data.py
    ├── preprocess.py
    ├── features.py
    ├── foa.py
    ├── model.py
    ├── train.py
    ├── eval.py
    └── explain.py
```

### Description of Components

| File                | Purpose                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| `config.yaml`       | Central configuration for paths, hyperparameters, and training settings |
| `run_train.py`      | Entry point for model training                                          |
| `run_eval.py`       | Generates evaluation metrics and performance reports                    |
| `run_explain.py`    | Produces explainability outputs using SHAP and LIME                     |
| `src/data.py`       | Dataset loading and splitting                                           |
| `src/preprocess.py` | Data cleaning, encoding, and scaling                                    |
| `src/features.py`   | Feature preparation and dataset tensor creation                         |
| `src/foa.py`        | Permutation Flamingo Optimization algorithm for feature selection       |
| `src/model.py`      | RNN-LSTM architecture                                                   |
| `src/train.py`      | Training pipeline and optimization loop                                 |
| `src/eval.py`       | Model performance evaluation                                            |
| `src/explain.py`    | Model interpretation using SHAP and LIME                                |

---

# Installation

Clone the repository and install dependencies.

```bash
git clone <YOUR_GITHUB_LINK>
cd pfo-rs-smart-agri
pip install -r requirements.txt
```

The implementation requires **Python 3.9 or later**.

---

# Dataset

The system expects a tabular dataset containing **agronomic and environmental attributes** used for crop recommendation.

Typical input attributes include:

* Soil nutrients (N, P, K)
* Soil pH
* Temperature
* Humidity
* Rainfall
* Additional agro-climatic indicators

Example dataset format:

| N  | P  | K  | temperature | humidity | ph  | rainfall | label |
| -- | -- | -- | ----------- | -------- | --- | -------- | ----- |
| 90 | 42 | 43 | 20.8        | 82       | 6.5 | 202      | rice  |

Place the dataset file inside a directory named:

```
data/
```

The path can also be configured in **config.yaml**.

---

# Configuration

All parameters for the experiment are controlled through:

```
config.yaml
```

Example parameters include:

* dataset path
* random seed
* FOA population size
* number of optimization iterations
* LSTM hidden size
* batch size
* learning rate
* training epochs

This design ensures that **all experiments can be reproduced by modifying a single configuration file**.

---

# Training the Model

To train the PFO-RS system:

```bash
python run_train.py
```

The training process performs:

1. Data loading and preprocessing
2. Feature scaling and encoding
3. Feature selection using **Permutation Flamingo Optimization**
4. RNN-LSTM training on the selected features
5. Saving the trained model

The trained model will be stored in:

```
outputs/models/
```

---

# Model Evaluation

To reproduce the evaluation metrics reported in the manuscript:

```bash
python run_eval.py
```

This step computes:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Regression metrics (MAE, RMSE, R²) if yield prediction is enabled

Evaluation results are saved to:

```
outputs/results/
```

---

# Model Explainability

To generate explanations for model predictions:

```bash
python run_explain.py
```

Two explanation techniques are supported:

### SHAP

Used to compute **global and local feature importance**.

### LIME

Provides **instance-level explanations for individual predictions**.

Generated outputs include:

* feature importance plots
* explanation summaries
* instance explanation reports

All explanation outputs are saved in:

```
outputs/explanations/
```

---

# Reproducibility

The repository is designed to ensure deterministic results.

Reproducibility measures include:

* fixed random seeds
* centralized configuration
* consistent dataset preprocessing
* modular experiment pipeline

All scripts are intended to run with **a single command**, enabling straightforward replication of experiments.

---

# Expected Outputs

Running the full pipeline will generate:

```
outputs/
│
├── models/
│   └── pfo_rs_model.pt
│
├── results/
│   ├── metrics.json
│   └── confusion_matrix.png
│
└── explanations/
    ├── shap_summary.png
    └── lime_examples/
```

---

# Citation

If you use this implementation in academic research, please cite the associated manuscript describing the **Permutation Flamingo Optimization-Driven Smart Agriculture Recommendation System**.

---

# Contact

For technical issues or questions related to the implementation, please open an issue in the repository.

---
