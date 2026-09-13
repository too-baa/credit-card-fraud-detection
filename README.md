# credit-card-fraud-detection
Case study 2 : Credit Card Fraud Detection — Apply XGBoost on a heavily imbalanced transactions dataset (e.g. Kaggle IEEE-CIS). Use SMOTE for oversampling, tune decision thresholds, and interpret results with feature importance scores.


# Credit Card Fraud Detection using XGBoost and SMOTE

A complete end-to-end Machine Learning pipeline designed to detect fraudulent credit card transactions using an extreme class imbalance dataset. This project implements advanced data handling via **SMOTE (Synthetic Minority Over-sampling Technique)**, trains an **XGBoost Classifier**, and optimizes performance using **custom decision threshold tuning**.

## Project Overview

In real-world fraud detection, transaction datasets are overwhelmingly skewed—normal transactions vastly outnumber fraudulent ones. Blindly training models on this data leads to silent failure where the model simply guesses the majority class. This project demonstrates how to effectively identify, correct, and model highly imbalanced tabular data.

### Key Highlights

* **Dataset:** Official [Kaggle ULB Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) containing 284,807 transactions with 30 PCA-transformed numerical features and a binary target (`Class`).
* **Exploratory Data Analysis (EDA):** Diagnosed severe class imbalance (`value_counts()` showing ~99.8% normal vs ~0.17% fraud).
* **Handling Imbalance:** Applied `SMOTE` strictly on the training split to synthesize minority fraud samples, balancing the classes from 227,451/394 to a 1:1 ratio (227,451/227,451).
* **Modeling:** Trained an `XGBClassifier` capable of capturing complex non-linear patterns.
* **Threshold Tuning:** Shifted the default classification threshold from `0.50` to an optimized `0.30` to maximize the F1-score and strike the ideal balance between catching fraudulent charges and minimizing false alarms.

---

## Performance Metrics

At the tuned threshold of **0.30**, the evaluation on the held-out test set (`56,962` transactions) yielded exceptional results:

* **Confusion Matrix:**

$$\begin{bmatrix} 56843 & 21 \\ 14 & 84 \end{bmatrix}$$



*(True Negatives: 56,843 | False Positives: 21 | False Negatives: 14 | True Positives: 84)*
* **Classification Report:**
* **Precision (Fraud - Class 1):** `0.80` (80% of flagged transactions are actual fraud)
* **Recall (Fraud - Class 1):** `0.86` (Catches 86% of all hidden fraud cases)
* **F1-Score:** `0.83`
* **Overall Accuracy:** `1.00`



---

## Project Structure

```text
├── CaseStudy2.ipynb         # Complete end-to-end Jupyter Notebook pipeline
└── README.md                # Project documentation

```

---

## Getting Started & Replication

You can instantly launch and run this notebook in Google Colab using your secure environment keys to fetch the dataset directly from Kaggle.
