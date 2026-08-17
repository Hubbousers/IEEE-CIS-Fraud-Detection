Absolutely. I’ve reorganized the material into a **clean, professional project-document structure** while preserving the technical journey, results, explanations, code, and portfolio narrative. I also separated the **engineering phases**, **technical concepts**, **results**, **architecture**, and **interview questions** so the project is easier to understand and present.

# IEEE-CIS Fraud Detection

## End-to-End Data Engineering, Machine Learning & MLOps Project

---

## 1. Project Overview

### 1.1 Project Name

**IEEE-CIS Fraud Detection**

### 1.2 Objective

Build an efficient, production-oriented machine learning pipeline capable of identifying fraudulent e-commerce transactions while addressing:

* Severe class imbalance
* Missing identity information
* Large-scale tabular data
* Multi-table relational data
* Memory constraints
* Temporal data leakage
* Low-latency inference requirements
* False-positive management

### 1.3 Dataset

The project uses the **IEEE-CIS Fraud Detection** dataset from Kaggle.

The dataset contains real-world e-commerce transaction information provided by **Vesta Corporation**, including:

* Transaction information
* Card information
* Product information
* Device information
* Browser information
* Identity information
* Network-related information

The primary training files are:

```text
train_transaction.csv
train_identity.csv
```

Supporting files:

```text
test_transaction.csv
test_identity.csv
sample_submission.csv
```

### 1.4 Business Problem

Fraud detection is a highly imbalanced classification problem.

Only a small percentage of transactions are fraudulent, while the overwhelming majority are legitimate.

The engineering challenge is therefore:

> Detect as many fraudulent transactions as possible without unnecessarily disrupting legitimate customers.

---

# 2. Business Problem in Simple Terms

Imagine a supermarket processing thousands of transactions.

Most customers are legitimate, but a small number may be using stolen cards.

A system that blocks everyone would prevent fraud but create terrible customer experiences.

A system that approves everyone would provide a great customer experience but allow fraudsters to operate freely.

Therefore, we need a system that produces a **risk score** for every transaction.

For example:

```text
Transaction A → Fraud probability = 0.02
Transaction B → Fraud probability = 0.17
Transaction C → Fraud probability = 0.91
```

The bank can then use these scores to determine the appropriate action:

```text
Low Risk
   ↓
Approve

Medium Risk
   ↓
Additional verification / OTP

High Risk
   ↓
Manual review / decline
```

The machine learning model is therefore not necessarily a simple:

```text
FRAUD / NOT FRAUD
```

system.

It is better understood as a:

```text
TRANSACTION → RISK SCORE → BUSINESS DECISION
```

system.

---

# 3. Six-Phase Engineering Framework

The project follows a six-phase engineering framework:

```text
┌──────────────────────┐
│ Phase 1              │
│ Data Ingestion       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Phase 2              │
│ Memory Optimization  │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Phase 3              │
│ Relational Join      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Phase 4              │
│ Time Validation      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Phase 5              │
│ Baseline LightGBM    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Phase 6              │
│ MLOps + Documentation│
└──────────────────────┘
```

### Current Project Status

| Phase                     | Status     | Result                     |
| ------------------------- | ---------- | -------------------------- |
| Phase 1 — Ingestion       | ✅ Complete | Dataset verified           |
| Phase 2 — Compression     | ✅ Complete | 69.4% memory reduction     |
| Phase 3 — Relational Join | ✅ Complete | 590,540 × 434 master table |
| Phase 4 — Time Validation | ✅ Complete | 80/20 chronological split  |
| Phase 5 — Baseline ML     | ✅ Complete | LightGBM ROC-AUC = 0.8979  |
| Phase 6 — MLOps/README    | 🔄 Next    | Portfolio packaging        |

---

# 4. Phase 1 — Environment Setup & Data Ingestion

## Objective

Verify that all competition files are available and establish the dataset pipeline.

### Dataset Files

```text
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
sample_submission.csv
```

### Primary Training Tables

```text
train_transaction
        +
train_identity
        ↓
Unified Training Dataset
```

---

# 5. Phase 2 — Memory Optimization

## 5.1 The Memory Problem

Pandas can automatically assign large numerical data types such as:

```text
int64
float64
```

These types may use significantly more memory than necessary.

For example, if a column contains only:

```text
0, 1, 2, 3, 4
```

it does not necessarily need a 64-bit integer representation.

This becomes important when working with hundreds of thousands of rows and hundreds of columns.

---

## 5.2 Optimization Strategy

The pipeline examines each numerical column and determines the smallest safe data type.

Conceptually:

```text
Original Data Type
       ↓
Inspect min/max
       ↓
Can int8 represent it?
       ↓
Yes → int8
No
       ↓
Can int16 represent it?
       ↓
Yes → int16
No
       ↓
Try int32 / int64
```

The same concept is applied to floating-point columns.

---

## 5.3 Memory Optimization Function

```python
import pandas as pd
import numpy as np

def reduce_mem_usage(df, verbose=True):
    """
    Reduce dataframe memory usage by safely downcasting
    numerical columns.
    """

    start_mem = df.memory_usage().sum() / 1024**2

    numerics = [
        'int16',
        'int32',
        'int64',
        'float16',
        'float32',
        'float64'
    ]

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and \
                   c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and \
                     c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and \
                     c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

            else:

                if c_min > np.finfo(np.float16).min and \
                   c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and \
                     c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem

        print(
            f"Memory reduced to {end_mem:.2f} MB "
            f"({reduction:.1f}% reduction)"
        )

    return df
```

---

## 5.4 Phase 2 Result

The raw data was compressed by approximately:

```text
69.4%
```

This substantially reduced the memory footprint before performing the relational join.

### Engineering Benefit

```text
Raw Data
   ↓
Memory Optimization
   ↓
Smaller DataFrames
   ↓
Safer Join
   ↓
Lower OOM Risk
   ↓
More Efficient ML Training
```

---

# 6. Phase 3 — Relational Table Joining

## 6.1 Dataset Structure

The project contains two important training tables.

### Transaction Table

```text
train_transaction
```

Contains information such as:

* Transaction amount
* Card information
* Product information
* Transaction timing
* Address-related information
* Transaction metadata

### Identity Table

```text
train_identity
```

Contains additional information such as:

* Device information
* Browser information
* Operating system
* Network/browser fingerprints
* Identity-related attributes

---

## 6.2 Why a LEFT JOIN?

The transaction table contains:

```text
590,540 rows
```

The identity table contains:

```text
144,233 rows
```

Therefore, many transactions do not have corresponding identity information.

A `LEFT JOIN` allows us to preserve every transaction.

Conceptually:

```text
Transaction
    │
    ├── Identity available
    │       ↓
    │   Add identity features
    │
    └── Identity unavailable
            ↓
        Keep transaction
        Identity = NaN
```

This is important because:

> Missing identity information itself may contain predictive information.

---

## 6.3 Join Implementation

```python
import pandas as pd
import time

start_time = time.time()

train_master = pd.merge(
    train_transaction,
    train_identity,
    on='TransactionID',
    how='left'
)

end_time = time.time()

print(
    f"Join completed in "
    f"{end_time - start_time:.2f} seconds"
)

print(train_master.shape)
```

---

# 7. Phase 3 Results

### Source Tables

```text
train_transaction
Rows:    590,540
Columns: 394

train_identity
Rows:    144,233
Columns: 41
```

### Unified Dataset

```text
Rows:    590,540
Columns: 434
```

The mathematical column count is:

```text
394 + 41 - 1
= 434
```

The subtraction occurs because `TransactionID` is shared.

### Join Performance

```text
Join time: 0.98 seconds
```

### Fraud Distribution

```text
Total transactions: 590,540
Fraud transactions: 20,663
Fraud rate:         3.50%
```

---

# 8. Key Dataset Insights

## Insight 1 — Identity Information Gap

Only:

```text
144,233 / 590,540 ≈ 24.4%
```

of transactions have matching identity records.

Approximately:

```text
75.6%
```

do not.

This creates a highly sparse feature space.

### Production implication

The model must be able to operate when identity information is missing.

---

# 9. Insight 2 — Wide Feature Space

The final dataset contains:

```text
434 features
```

This is a wide tabular dataset.

The model therefore needs to efficiently process:

```text
590K rows
×
434 columns
```

This makes efficient tree-based algorithms attractive.

---

# 10. Insight 3 — Class Imbalance

The fraud rate is:

```text
3.50%
```

Therefore:

```text
Legitimate ≈ 96.50%
Fraudulent ≈ 3.50%
```

A naive model could predict:

```text
NOT FRAUD
```

for every transaction.

It would obtain approximately:

```text
96.5% accuracy
```

while catching:

```text
0% of fraud
```

Therefore, accuracy alone is not an appropriate primary metric.

---

# 11. Better Evaluation Metrics

Important fraud-detection metrics include:

### ROC-AUC

Measures how well the model ranks fraudulent transactions above legitimate transactions.

### Precision

Of transactions flagged as fraud, how many are actually fraud?

```text
Precision =
True Positives /
(True Positives + False Positives)
```

### Recall

Of all actual fraud cases, how many did we detect?

```text
Recall =
True Positives /
(True Positives + False Negatives)
```

### F1 Score

Balances precision and recall.

```text
F1 =
2 × Precision × Recall /
(Precision + Recall)
```

---

# 12. Phase 4 — Time-Based Validation

## 12.1 Why Random Splitting Is Dangerous

A normal random split might do this:

```text
January ─┐
February ├── Randomized ──> Training
March    │
April    └───────────────> Validation
```

This allows information from future transactions to influence the model used to predict earlier transactions.

That creates a temporal leakage problem.

---

# 13. Out-of-Time Validation

A more realistic strategy is:

```text
PAST
│
├── January
├── February
├── March
└── April
       ↓
   TRAIN MODEL

FUTURE
│
└── May
       ↓
   VALIDATE MODEL
```

The model is therefore evaluated on transactions that occur later in time.

---

# 14. Chronological Split

The dataset is sorted using:

```python
TransactionDT
```

Then:

```text
First 80% → Training
Last 20%  → Validation
```

### Implementation

```python
train_master = (
    train_master
    .sort_values('TransactionDT')
    .reset_index(drop=True)
)

split_idx = int(len(train_master) * 0.8)

X = train_master.drop(
    columns=[
        'TransactionID',
        'isFraud',
        'TransactionDT'
    ]
)

y = train_master['isFraud']

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]

X_val = X.iloc[split_idx:]
y_val = y.iloc[split_idx:]
```

---

# 15. Phase 4 Results

### Training

```text
Rows:       472,432
Features:   431
Fraud rate: 3.51%
```

### Validation

```text
Rows:       118,108
Features:   431
Fraud rate: 3.44%
```

The relatively stable fraud rate indicates that the target distribution did not dramatically change between the two periods.

---

# 16. Phase 5 — Baseline Machine Learning

# LightGBM

## 16.1 What Is LightGBM?

**LightGBM** is a gradient-boosting decision-tree framework developed by Microsoft.

Instead of creating one enormous decision tree, it builds many smaller trees sequentially.

Conceptually:

```text
Tree 1
  ↓
Prediction
  ↓
Find Errors
  ↓
Tree 2
  ↓
Correct Previous Errors
  ↓
Tree 3
  ↓
Correct Remaining Errors
  ↓
...
  ↓
Final Prediction
```

The final prediction combines the contributions of many trees.

---

# 17. Why LightGBM?

LightGBM is particularly useful for this project because it is designed for large-scale tabular machine learning.

Important characteristics include:

* Gradient-boosted decision trees
* High training efficiency
* Native missing-value handling
* Categorical feature support
* Low memory usage relative to many alternatives
* Parallel computation
* Strong performance on tabular datasets

---

# 18. Standard Machine Learning vs LightGBM

## Standard Approach

Suppose we use a simple classifier.

It may struggle because:

```text
590K rows
+
434 features
+
Many missing values
+
Categorical variables
+
Severe class imbalance
```

Traditional preprocessing can become complicated.

---

## LightGBM Approach

LightGBM can directly learn nonlinear relationships such as:

```text
Transaction Amount
        +
Card Information
        +
Device Information
        +
Time
        +
Product
        +
Missingness
        ↓
Fraud Risk
```

It can also use missing values as part of its decision-making process.

---

# 19. Phase 5 — Handling Categorical Variables

Categorical object columns are converted to Pandas categorical types.

```python
categorical_cols = (
    X_train_clean
    .select_dtypes(include=['object'])
    .columns
    .tolist()
)

for col in categorical_cols:

    X_train_clean[col] = (
        X_train_clean[col]
        .astype('category')
    )

    X_val_clean[col] = (
        X_val_clean[col]
        .astype('category')
    )
```

This allows the model to work with categorical information more efficiently.

---

# 20. The Core Problem — Class Imbalance

The training data contains approximately:

```text
96.5% legitimate
3.5% fraud
```

Therefore, the approximate class ratio is:

```text
Negative : Positive
     27.46 : 1
```

The model needs to understand that fraud errors are especially important.

---

# 21. Cost-Sensitive Learning

## What Is Cost-Sensitive Learning?

Cost-sensitive learning means assigning different importance to different types of mistakes.

Consider:

### False Negative

Actual fraud:

```text
Fraud → Model says Legitimate
```

The bank may lose money.

### False Positive

Actual legitimate:

```text
Legitimate → Model says Suspicious
```

The customer may experience additional verification.

These two mistakes have different business costs.

---

# 22. `scale_pos_weight`

LightGBM provides:

```python
scale_pos_weight
```

to increase the importance of the positive class.

The basic calculation is:

```text
scale_pos_weight =
Number of Negative Examples
──────────────────────────
Number of Positive Examples
```

In this project:

```text
scale_pos_weight ≈ 27.46
```

Therefore, the positive/fraud class receives substantially greater influence during training.

---

# 23. What Does the 27.46 Actually Mean?

It does **not** literally mean:

> "Every fraud mistake costs exactly 27.46 times more money."

Instead, it means the training objective gives greater weight to the positive class.

Conceptually:

```text
Normal training

Legitimate error → normal contribution
Fraud error      → normal contribution


Weighted training

Legitimate error → normal contribution
Fraud error      → much larger contribution
```

This changes the optimization process.

---

# 24. Why Does Standard Machine Learning Miss Fraud?

This answers one of the central questions in the project.

A naive classifier may minimize an overall loss such as:

```text
Total Error
=
Legitimate Errors
+
Fraud Errors
```

But legitimate transactions massively outnumber fraud transactions.

For example:

```text
10,000 transactions

9,650 legitimate
350 fraud
```

Suppose a model predicts everything as legitimate:

```text
Correct legitimate predictions = 9,650
Fraud detected                = 0
```

Accuracy:

```text
9,650 / 10,000
=
96.5%
```

The model appears highly accurate.

But from a fraud perspective:

```text
Fraud Recall = 0%
```

The model has completely failed its business objective.

---

# 25. Why Accuracy Is Misleading

This is the fundamental problem:

```text
High Accuracy
       ≠
Good Fraud Detection
```

A useful fraud model needs to distinguish between:

```text
Fraud
vs.
Legitimate
```

despite the extreme imbalance.

This is why we use:

```text
ROC-AUC
Precision
Recall
F1
```

rather than relying only on accuracy.

---

# 26. Baseline LightGBM Configuration

```python
clf = lgb.LGBMClassifier(

    n_estimators=100,

    learning_rate=0.05,

    num_leaves=31,

    scale_pos_weight=scale_factor,

    random_state=42,

    n_jobs=-1,

    verbose=-1
)
```

### Main Parameters

| Parameter          | Purpose                        |
| ------------------ | ------------------------------ |
| `n_estimators`     | Number of boosting trees       |
| `learning_rate`    | Contribution of each tree      |
| `num_leaves`       | Complexity of individual trees |
| `scale_pos_weight` | Handles class imbalance        |
| `n_jobs=-1`        | Uses available CPU cores       |
| `random_state`     | Reproducibility                |

---

# 27. Baseline Training

```python
start_time = time.time()

clf.fit(
    X_train_clean,
    y_train
)

end_time = time.time()

print(
    f"Training completed in "
    f"{end_time - start_time:.2f} seconds"
)
```

---

# 28. Model Output

The model produces probability scores:

```python
y_prob = clf.predict_proba(X_val_clean)[:, 1]
```

Example:

```text
Transaction A → 0.03
Transaction B → 0.42
Transaction C → 0.91
```

These are risk scores rather than simple binary decisions.

---

# 29. Baseline Results

## Training Time

```text
29.56 seconds
```

## Validation Dataset

```text
118,108 transactions
```

## ROC-AUC

```text
0.8979
```

## Classification Report

```text
                 precision   recall   f1-score

Legitimate          0.99      0.89      0.94

Fraud               0.19      0.75      0.31
```

---

# 30. Interpreting ROC-AUC = 0.8979

An ROC-AUC of:

```text
0.8979
```

means the model has strong ranking ability.

A useful interpretation is:

> If we randomly select one fraudulent transaction and one legitimate transaction, the model will rank the fraudulent transaction higher approximately 89.79% of the time.

This is why ROC-AUC is useful for a risk-ranking system.

---

# 31. Interpreting Fraud Recall = 75%

The fraud class has:

```text
Recall = 0.75
```

The validation set contains:

```text
4,064 fraudulent transactions
```

Therefore, approximately:

```text
4,064 × 0.75
≈ 3,048
```

fraudulent transactions were identified at the model's chosen classification threshold.

Approximately:

```text
1,016
```

were missed.

This is important because a production fraud system must explicitly consider the cost of those missed transactions.

---

# 32. Why Is Fraud Precision Only 19%?

Fraud precision:

```text
0.19
```

means:

> Among transactions classified as fraud by the selected threshold, approximately 19% were actually fraudulent.

The remaining flagged transactions were legitimate.

This is the **false-positive problem**.

The reason is closely related to the aggressive class weighting:

```text
scale_pos_weight = 27.46
```

The model is being encouraged to pay considerably more attention to the minority fraud class.

Therefore, we trade some precision for higher recall.

---

# 33. Production Interpretation

The model should not necessarily be interpreted as:

```text
Probability > threshold
        ↓
Automatically decline card
```

A better architecture is:

```text
Transaction
     ↓
LightGBM Risk Score
     ↓
Risk Decision Layer
     │
     ├── Low Risk
     │      ↓
     │   Approve
     │
     ├── Medium Risk
     │      ↓
     │   OTP / 2FA
     │
     └── High Risk
            ↓
        Manual Review /
        Decline
```

This allows the organization to balance:

```text
Fraud Loss
        ↕
Customer Friction
```

---

# 34. Complete Optimized Architecture

The complete system can be represented as:

```text
                 ┌──────────────────────┐
                 │ Transaction Data     │
                 │ 590K+ records        │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ Identity Data        │
                 │ 144K+ records        │
                 └──────────┬───────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │ Memory Optimization    │
                │ Numeric Downcasting    │
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │ Relational LEFT JOIN   │
                │ TransactionID          │
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │ Unified Feature Table  │
                │ 590,540 × 434         │
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │ Temporal Ordering      │
                │ TransactionDT          │
                └────────────┬───────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
        ┌────────────────┐      ┌────────────────┐
        │ Historical 80% │      │ Future 20%     │
        │ Training       │      │ Validation     │
        └───────┬────────┘      └───────┬────────┘
                │                       │
                ▼                       │
        ┌────────────────┐              │
        │ LightGBM       │              │
        │ + Class Weight │              │
        └───────┬────────┘              │
                │                       │
                └──────────┬────────────┘
                           ▼
                ┌────────────────────────┐
                │ Risk Probability       │
                │ 0.00 → 1.00            │
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │ Decision / Risk Layer  │
                └───────┬───────┬────────┘
                        │       │
                     Low Risk  High Risk
                        │       │
                     Approve   Review /
                               Verification
```

---

# 35. Production Edge Cases

## Edge Case 1 — Seasonal/Data Drift

### Problem

During events such as:

```text
Black Friday
Diwali
Christmas
Ramadan
Large e-commerce sales
```

customer behavior can change.

A large purchase that normally looks suspicious may become completely normal.

### Risk

```text
Normal behavior changes
        ↓
Model sees unfamiliar patterns
        ↓
False positives increase
```

### Solution

Use:

* Time-aware features
* Rolling statistics
* Recent customer behavior
* Monitoring
* Periodic retraining
* Drift detection

---

# 36. Edge Case 2 — Missing Identity Pipeline

Suppose the device-data pipeline suddenly fails.

Normally:

```text
Device information available
```

Suddenly:

```text
Device information = 100% missing
```

The model may still produce predictions, but the production system should recognize that this is an infrastructure anomaly.

### Solution

Monitor feature completeness:

```text
Missing DeviceInfo
        ↓
Expected range?
        │
        ├── Yes → Continue
        │
        └── No
             ↓
        Trigger Alert
```

---

# 37. Edge Case 3 — Latency Constraint

Fraud detection frequently operates under strict latency requirements.

The architecture should therefore avoid unnecessary operations during inference.

### Important distinction

The **training pipeline** can take seconds or minutes.

The **production inference pipeline** needs to be much faster.

Therefore:

```text
Offline Training
       ↓
Heavy computation acceptable

Online Inference
       ↓
Low latency required
```

The model should ideally receive already-prepared features rather than performing expensive historical joins for every transaction.

---

# 38. How Multi-Relational Data Should Work in Production

The current project performs:

```text
train_transaction
        +
train_identity
        ↓
train_master
```

This is appropriate for offline model training.

A production system would ideally separate:

### Offline Feature Engineering

```text
Historical Transactions
        ↓
Aggregations
        ↓
Feature Store
```

### Online Prediction

```text
New Transaction
       ↓
Retrieve Existing Features
       ↓
LightGBM
       ↓
Risk Score
```

This avoids repeatedly performing expensive large-table joins during live transactions.

---

# 39. Evolution of Fraud Detection Approaches

## Approach 1 — Hardcoded Rules

Example:

```python
if TransactionAmt > 5000 and DeviceType == "mobile":
    fraud = True
```

### Advantages

* Simple
* Explainable
* Easy to deploy

### Problems

* Brittle
* Difficult to maintain
* Easy for attackers to circumvent
* Cannot learn complex interactions

An attacker could simply change:

```text
$5,001
```

to:

```text
$4,999
```

---

# 40. Approach 2 — Standard Machine Learning

Examples:

```text
Logistic Regression
Random Forest
Basic Decision Trees
```

These models can learn relationships automatically.

However, problems can remain:

* Class imbalance
* Missing values
* High-dimensional categorical data
* Nonlinear interactions
* Large-scale processing

The biggest issue is that an unweighted model may prioritize overall accuracy rather than fraud detection.

---

# 41. Approach 3 — Cost-Sensitive Gradient Boosting

The proposed baseline combines:

```text
Efficient data engineering
        +
Relational data fusion
        +
Temporal validation
        +
LightGBM
        +
Class weighting
```

This is much closer to a production-oriented architecture.

---

# 42. Why the Optimal Approach Works

The system addresses the major problems independently.

| Problem                 | Solution                      |
| ----------------------- | ----------------------------- |
| Large memory footprint  | Downcasting                   |
| Multiple tables         | LEFT JOIN                     |
| Missing identity data   | Native missing-value handling |
| Class imbalance         | `scale_pos_weight`            |
| Temporal leakage        | OOT validation                |
| Nonlinear relationships | Gradient boosting             |
| Large tabular dataset   | LightGBM                      |
| Customer friction       | Risk-based decision layer     |
| Production drift        | Monitoring + retraining       |

---

# 43. LightGBM Performance

The baseline experiment produced:

```text
Training rows:       472,432
Features:            431
Trees:               100
Training time:       29.56 seconds
OOT ROC-AUC:         0.8979
```

This demonstrates that LightGBM can train a strong baseline on hundreds of thousands of rows in a relatively short time on the Kaggle environment used for this project.

However:

> Training speed is hardware-dependent.

Actual performance varies according to:

* CPU
* RAM
* Number of threads
* Dataset size
* Number of trees
* Number of leaves
* Feature cardinality
* Categorical features
* LightGBM version
* Hardware architecture

Therefore, **29.56 seconds should be reported as this experiment's measured training time**, not as a universal LightGBM speed.

---

# 44. Why LightGBM Is Fast

LightGBM was specifically designed to improve the efficiency of gradient-boosted decision trees.

Important techniques include:

### Histogram-Based Learning

Continuous feature values are grouped into bins.

Instead of considering every possible numerical threshold:

```text
1.000
1.001
1.002
1.003
...
```

the algorithm can work with a smaller set of histogram bins.

This reduces computational work.

### Leaf-Wise Tree Growth

LightGBM typically grows the leaf that provides the largest improvement in the objective.

Conceptually:

```text
Traditional level-wise:

        Root
       /    \
      /      \
     /        \
    ↓          ↓


LightGBM leaf-wise:

        Root
       /    \
      /      \
     ↓        ↓
    Best     Other
    Leaf     Leaf
     ↓
   Split
```

This can produce strong predictive performance efficiently.

---

# 45. LightGBM History

LightGBM was introduced by **Microsoft Research** as an efficient gradient boosting framework for large-scale machine learning.

It was designed to address limitations encountered when applying traditional gradient-boosted decision trees to very large datasets.

Its design emphasized:

```text
Speed
+
Memory efficiency
+
Scalability
+
Strong tabular performance
```

It subsequently became widely used in:

* Machine learning competitions
* Risk modeling
* Financial analytics
* Recommendation systems
* Ranking
* Customer analytics
* Classification
* Regression

---

# 46. Applications of LightGBM

## Financial Services

```text
Fraud detection
Credit risk
Loan default prediction
Customer risk scoring
Transaction monitoring
```

## E-Commerce

```text
Purchase prediction
Customer churn
Recommendation
Conversion prediction
Ad ranking
```

## Marketing

```text
Customer segmentation
Response prediction
Campaign optimization
Lead scoring
```

## Operations

```text
Demand forecasting
Anomaly detection
Failure prediction
Resource optimization
```

## Search and Ranking

```text
Search ranking
Recommendation ranking
Click-through prediction
```

---

# 47. Important Limitation of the Current Baseline

The current project is a **strong baseline**, but it should not yet be described as a complete production fraud system.

Important future improvements include:

```text
Feature Engineering
       ↓
Card-level behavioral features
       ↓
Time-window statistics
       ↓
Device relationships
       ↓
Transaction velocity
       ↓
Historical fraud patterns
       ↓
Threshold optimization
       ↓
Calibration
       ↓
Drift monitoring
       ↓
Production deployment
```

---

# 48. Recommended Next Phase — Feature Engineering

The next major improvement should focus on features such as:

### Transaction Velocity

```text
Number of transactions
in previous 5 minutes
```

### Card Spending Behavior

```text
Current amount
/
Historical average amount
```

### Device Behavior

```text
Number of cards
associated with device
```

### Temporal Behavior

```text
Transaction hour
Day of week
Time since previous transaction
```

### Aggregated Features

```text
Average transaction amount
Median amount
Transaction count
Unique devices
Unique addresses
Unique merchants
```

These features can allow the model to learn behavior rather than relying only on raw transaction attributes.

---

# 49. Final Project Results

## Data Engineering

```text
590,540 transactions
434 unified features
69.4% memory reduction
0.98-second relational join
```

## Validation

```text
80/20 chronological split
472,432 training rows
118,108 validation rows
```

## Machine Learning

```text
Algorithm: LightGBM
Class weighting: 27.46
Training time: 29.56 seconds
OOT ROC-AUC: 0.8979
Fraud recall: 75%
```

---

# 50. Final Engineering Narrative

The project demonstrates an end-to-end approach to fraud detection:

```text
Raw Data
   ↓
Memory-Constrained Ingestion
   ↓
Data Compression
   ↓
Relational Data Fusion
   ↓
Feature Matrix
   ↓
Temporal Validation
   ↓
Cost-Sensitive Learning
   ↓
LightGBM
   ↓
Fraud Risk Score
   ↓
Operational Decision
```

The key engineering principle is:

> **The goal is not simply to maximize accuracy. The goal is to build a system that detects fraud reliably while controlling customer friction, computational cost, and production risk.**

---

# 51. Portfolio Architecture Summary

```text
                    IEEE-CIS FRAUD DETECTION
                              │
                              ▼
                    ┌───────────────────┐
                    │ Data Ingestion    │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │ Memory Compression│
                    │ 69.4% Reduction   │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │ Relational Fusion │
                    │ LEFT JOIN         │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │ 590K × 434        │
                    │ Feature Matrix    │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │ Temporal Split    │
                    │ 80% / 20%         │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │ LightGBM          │
                    │ Cost-Sensitive    │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │ Fraud Probability │
                    │ 0.00 → 1.00       │
                    └─────────┬─────────┘
                              ▼
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          Low Risk        Medium Risk      High Risk
              │               │               │
              ▼               ▼               ▼
           Approve        Verification      Review/
                                            Decline
```

---

# 52. Portfolio Positioning

This project demonstrates knowledge across several engineering disciplines:

### Data Engineering

* Large CSV ingestion
* Memory optimization
* Data type optimization
* Relational joins

### Machine Learning

* Gradient boosting
* Class imbalance
* Cost-sensitive learning
* Probability prediction
* Model evaluation

### ML Engineering

* Temporal validation
* Leakage prevention
* Feature preparation
* Reproducible training

### Production Thinking

* Missing-data failures
* Data drift
* Latency
* False positives
* Risk-based decisions

---

# 53. Final Takeaway

The project evolved from:

```text
Raw Kaggle Dataset
```

into:

```text
Memory-Efficient
+
Relationally Integrated
+
Temporally Validated
+
Cost-Sensitive
+
Gradient-Boosted
Fraud Detection Pipeline
```

The baseline achieved:

```text
ROC-AUC = 0.8979
```

with:

```text
75% fraud recall
```

on an unseen chronological validation period.

The next engineering objective is not simply to make the model more complicated.

It is to improve the system intelligently through:

```text
Behavioral Feature Engineering
        +
Threshold Optimization
        +
Probability Calibration
        +
Drift Monitoring
        +
Feature Store Design
        +
Production Inference Architecture
```

This transforms the project from a Kaggle model into a stronger demonstration of **real-world fraud detection engineering**.

---

# 54. Proposed Repository Structure

```text
ieee-cis-fraud-lens/
│
├── README.md
│
├── src/
│   ├── ingestion.py
│   ├── memory_optimizer.py
│   ├── feature_engineering.py
│   ├── validation.py
│   ├── train.py
│   └── inference.py
│
├── notebooks/
│   ├── 01_ingestion.ipynb
│   ├── 02_memory_optimization.ipynb
│   ├── 03_relational_join.ipynb
│   ├── 04_temporal_validation.ipynb
│   └── 05_lightgbm_baseline.ipynb
│
├── configs/
│   └── model_config.yaml
│
├── reports/
│   └── model_evaluation.md
│
├── requirements.txt
│
└── .gitignore
```

---

# 55. Interview Questions to Master

Before presenting this project, be prepared to explain:

1. Why is accuracy a poor metric for fraud detection?
2. What is class imbalance?
3. What is `scale_pos_weight`?
4. Why does class weighting improve fraud recall?
5. What is LightGBM?
6. Why use LightGBM instead of logistic regression?
7. Why use a LEFT JOIN?
8. Why are so many identity features missing?
9. Why is missingness potentially predictive?
10. Why is random train/test splitting dangerous?
11. What is Out-of-Time validation?
12. What is `TransactionDT`?
13. What does ROC-AUC = 0.8979 mean?
14. Why is fraud precision only 19%?
15. How would you reduce false positives?
16. How would you detect data drift?
17. How would you handle a sudden identity-data outage?
18. How would you deploy this model for low-latency inference?
19. How would you monitor the model in production?
20. What features would you engineer next?

---

# 56. One-Sentence Project Summary

> **Designed an end-to-end IEEE-CIS fraud detection pipeline that compressed large tabular datasets by 69.4%, fused transaction and identity data through relational joins, prevented temporal leakage through Out-of-Time validation, and trained a cost-sensitive LightGBM baseline achieving 0.8979 ROC-AUC and 75% fraud recall in 29.56 seconds.**
