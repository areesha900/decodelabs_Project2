# DecodeLabs Internship - Project 2: Data Classification using AI

## 📌 Project Overview

This project implements a **supervised machine learning pipeline** to classify Iris flowers into three species - *Setosa*, *Versicolor*, and *Virginica* using the **K-Nearest Neighbors (KNN)** algorithm.

---

## 🧠 Concepts Covered

| Concept | Description |
|---|---|
| Supervised Learning | Model trained on labeled data to predict unseen samples |
| Feature Scaling | StandardScaler normalizes features to mean=0, variance=1 |
| Train-Test Split | 80% training / 20% testing with stratification & shuffle |
| KNN Algorithm | Classifies based on majority vote of K nearest neighbors |
| Elbow Method | Used to find the optimal value of K |
| Confusion Matrix | Breaks down TP, FP, FN, TN per class |
| F1 Score | Harmonic mean of Precision and Recall — robust to class imbalance |

---

## 📊 Dataset — The Iris Benchmark

| Property | Value |
|---|---|
| Source | `sklearn.datasets.load_iris()` |
| Samples | 150 (balanced - 50 per class) |
| Classes | Setosa · Versicolor · Virginica |
| Features | Sepal Length, Sepal Width, Petal Length, Petal Width |

---

## ⚙️ Pipeline (IPO Framework)

```
INPUT                  PROCESS                     OUTPUT
─────────────────      ─────────────────────────   ─────────────────────
Iris Dataset      →    StandardScaler          →   Confusion Matrix
Feature Scaling        Train-Test Split (80/20)    F1 Score: 0.9666
                       KNN (optimal K=1)           Accuracy: 96.67%
```

---

## 📈 Results

```
                precision    recall   f1-score   support
      setosa       1.00       1.00      1.00        10
  versicolor       0.91       1.00      0.95        10
   virginica       1.00       0.90      0.95        10

    accuracy                            0.97        30
   macro avg       0.97       0.97      0.97        30
weighted avg       0.97       0.97      0.97        30
```

**-> Only 1 misclassification out of 30 test samples.**

---

## 🖼️ Visualizations

The script automatically generates a 3-panel figure:
- **Elbow Curve** — Error rate vs K value to identify optimal K
- **Confusion Matrix** — Heatmap of predicted vs actual classes
- **Feature Chart** — Post-scaling feature magnitude comparison

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/areesha900/decodelabs_Project2.git
cd decodelabs_Project2
```

**2. Install dependencies**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

**3. Run the classifier**
```bash
python3 iris_classifier.py
```

---

## 🔍 Sample Prediction

```python
# Input: sepal_l = 5.1, sepal_w = 3.5, petal_l = 1.4, petal_w = 0.2
# Predicted: SETOSA
# Confidence:
    setosa       100.0%
    versicolor   0.0%
    virginica    0.0%
```

---

