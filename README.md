# 💳 Credit Default Prediction

This project predicts whether a borrower will experience serious financial distress (default) within 2 years using **Machine Learning** and **Deep Learning**.

---

## 📊 Dataset

- **Source:** Kaggle — *"Give Me Some Credit"* dataset (`cs-training.csv`)
- **Target:** `Serious_Dlq_In_2Yrs` (1 = Default, 0 = No Default)
- **Size:** ~150,000 rows, 11 financial features (Age, Income, Debt Ratio, Late Payments, etc.)

---

## 📁 Project Structure

The project is divided into 3 main steps:

| File | Description |
|------|-------------|
| `EDA.py` | Data cleaning, preprocessing, and feature engineering |
| `Models.py` | Training traditional Machine Learning models |
| `ANN_.py` | Building a Deep Learning model using PyTorch |

---

## ⚙️ Pipeline

### 1. Data Preprocessing
- Renamed columns for readability
- Handled missing values and extreme outliers
- Created new features (e.g., `Debt_To_Income_Ratio`)

### 2. Handling Class Imbalance
The dataset is highly imbalanced — **93% No Default** vs. **7% Default**.
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) on the training data to balance the classes

### 3. Machine Learning Models (`Models.py`)
Trained and compared 4 algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### 4. Deep Learning Model (`ANN_.py`)
Built an **Artificial Neural Network** using PyTorch.

**Architecture:**
```
Input (11 features)
  → Dense(64) + ReLU + Dropout(30%)
  → Dense(32) + ReLU + Dropout(20%)
  → Dense(16) + ReLU
  → Dense(1)  + Sigmoid  →  Output: Default Probability
```

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy (BCELoss) |
| Early Stopping | Patience = 10 |
| Best ROC-AUC | **0.8285** |

---

## 📈 Key Metrics

Because the data is imbalanced, **Accuracy alone is misleading**. We focus on:

| Metric | Purpose |
|--------|---------|
| **ROC-AUC** | Overall model performance (0.82+ achieved) |
| **Recall** | Catching as many actual defaulters as possible |
| **Precision / F1-Score** | Balancing false alarms vs. missed defaulters |

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Language | Python |
| Data | Pandas, NumPy |
| Machine Learning | Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE) |
| Deep Learning | PyTorch |
| Visualization | Matplotlib, Seaborn |
