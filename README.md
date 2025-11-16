# Fraud Detection in New Bank Account Transactions

## Overview

This project implements a robust framework utilizing **supervised machine learning techniques** to detect fraudulent activities within **new bank account transactions**. The goal is to enhance the accuracy and efficiency of fraud detection systems, ultimately safeguarding customer trust and reducing the impact of sophisticated fraudulent schemes on banking operations.

The research addresses the critical challenge of **highly imbalanced data**—a common issue in financial fraud detection where fraudulent transactions constitute only a small fraction of the total records.

| Detail | Information |
| :--- | :--- |
| **Topic** | Fraud Detection in New Bank Account |
| **Dataset Source** | Base.csv file from the Bank Account Fraud (BAF) dataset (NeurIPS 2022). |
| **Model Focus** | Supervised classification using resampling techniques (e.g., Random Forest, Logistic Regression). |

## Objectives

The primary objectives of this study were to:

1.  **Develop Predictive Models:** Implement and evaluate supervised machine learning techniques specifically tailored for detecting fraud in bank account transactions.
2.  **Handle Imbalance:** Address the severe class imbalance problem using advanced data balancing strategies like **SMOTE-ENN**.
3.  **Identify Optimal Algorithms:** Determine which machine learning models (e.g., Random Forest, Logistic Regression) are most accurate in identifying fraudulent transactions.
4.  **Provide Actionable Insights:** Analyze the underlying patterns and characteristics of fraudulent behavior through empirical analysis.

## Data and Preprocessing

### Dataset Description

The study uses the Base.csv file from the BAF dataset, which simulates transactional data and includes detailed customer and transaction-level information. The data includes features such as customer demographics, transaction history, and behavioral attributes.

Key features used include velocity metrics (`velocity_6h`, `velocity_24h`, `velocity_4w`), `income`, `customer_age`, and `credit_risk_score`. The target variable is the binary column `fraud_bool`.

### Data Preprocessing Pipeline

1.  **Feature Engineering:** Irrelevant or redundant features were dropped (e.g., `name_email_similarity`, certain identifier features) to reduce model complexity.
2.  **Scaling:** Numerical features were scaled using **RobustScaler** because it is less sensitive to outliers compared to other methods (like StandardScaler) as it relies on the interquartile range (IQR).
3.  **Encoding:** Categorical variables were converted to numerical representations using **One-Hot Encoding** with the `drop='first'` parameter to avoid multicollinearity.
4.  **Time-Based Splitting:** The data was split based on the `month` column, using earlier months (0-5) for the training set and later months (6-7) for the test set, ensuring a realistic evaluation for time-sensitive fraud detection.

## Methodology

### Addressing Class Imbalance

Due to the scarcity of fraudulent transactions, resampling techniques were applied *exclusively* to the training data.

*   **SMOTE-ENN:** This hybrid method, combining **SMOTE** (Synthetic Minority Oversampling Technique) and **ENN** (Edited Nearest Neighbors), was integrated to generate synthetic samples for the minority class (fraudulent transactions) while simultaneously removing noisy or ambiguous samples from both classes, enhancing dataset quality and detection accuracy.
*   Other methods explored included SMOTE, ADASYN (Adaptive Synthetic Sampling), SMOTETomek, and RandomUnderSampler.

### Machine Learning Models

The following supervised machine learning models were trained and evaluated:

*   **Random Forest** (Top performer)
*   **Logistic Regression**
*   **K-Nearest Neighbors (KNN)**
*   **Naive Bayes** (Gaussian and Bernoulli variants)
*   **Decision Tree**

### Evaluation Metrics

Due to the highly imbalanced nature of fraud data, evaluation focused on metrics that assess performance on the minority class:

*   **F1 Score:** The harmonic mean of Precision and Recall, providing a balanced assessment.
*   **Precision-Recall AUC (PR-AUC):** A crucial metric for imbalanced datasets, emphasizing precision and recall.
*   **Recall (True Positive Rate):** Measures the proportion of actual fraudulent transactions correctly identified, which is critical for minimizing missed detections (false negatives).
*   **ROC-AUC:** Measures the overall discriminatory ability across all thresholds.

## Results and Performance

### Key Findings

The implementation of resampling techniques, particularly hybrid methods like SMOTEENN, significantly improved model performance by balancing the dataset and reducing noise.

The models were ranked based on their overall effectiveness in detecting fraudulent transactions:

1.  **Random Forest**
2.  Logistic Regression
3.  Bernoulli Naive Bayes
4.  Decision Tree
5.  Gaussian Naive Bayes
6.  KNN

*Analogy:* Implementing fraud detection in a bank is like setting up a high-security checkpoint at a major financial hub. Traditional systems might only check for basic suspicious behaviors, but a robust Random Forest model combined with SMOTE-ENN resampling acts like a specialized counter-terrorism unit. It not only uses complex rules (the ensemble of trees) but also intentionally studies rare patterns (synthetic minority samples) to ensure it identifies high-risk threats that are subtle and infrequent, balancing the need to catch every threat (high Recall) against the need to avoid unnecessarily delaying genuine customers (high Precision).
