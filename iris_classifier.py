# =============================================================================
# DecodeLabs Internship — Project 2: Data Classification Using AI
# Algorithm : K-Nearest Neighbors (KNN)
# Dataset   : Iris Benchmark (sklearn)
# Pipeline  : Load → Scale → Split → Train → Evaluate → Visualize
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

# ── ANSI colour helpers (terminal output) ────────────────────────────────────
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"

def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


# =============================================================================
# STEP 2 — UNDERSTAND THE DATASET
# =============================================================================
section("STEP 2 · Understand the Dataset")

iris      = load_iris()
X         = iris.data                        # shape (150, 4)
y         = iris.target                      # 0 = setosa, 1 = versicolor, 2 = virginica
class_names = iris.target_names
feature_names = iris.feature_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = [class_names[i] for i in y]

print(df.head(10).to_string(index=False))
print(f"Shape   : {X.shape}")
print(f"Classes : {class_names}\n")
print("\nBasic statistics:")
print(df.describe().round(2).to_string())


# =============================================================================
# STEP 3 — FEATURE SCALING & TRAIN/TEST SPLIT 
# =============================================================================
section("STEP 3 · Feature Scaling & Train/Test Split")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size    = 0.20,
    random_state = 42,
    shuffle      = True,
    stratify     = y,
)

print(f"Training samples : {X_train.shape[0]}  (80 %)")
print(f"Testing  samples : {X_test.shape[0]}   (20 %)")
print("\nTest-set class distribution:")
for name, count in zip(class_names, np.bincount(y_test)):
    print(f"  {name:<12} {count} samples")

# =============================================================================
# STEP 4 — ELBOW METHOD & TRAINING
# =============================================================================
section("STEP 4 · Tuning the Engine & Training the Model")

k_range     = range(1, 21)
error_rates = []

for k in k_range:
    knn   = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))

optimal_k = k_range[np.argmin(error_rates)]
print(f"Error rates by K: {[round(e, 4) for e in error_rates]}")
print(f"{GREEN}Optimal K = {optimal_k}  (lowest error rate = {min(error_rates):.4f}){RESET}")

# training
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)           # FIT
predictions = model.predict(X_test)   # PREDICT

print(f"Model trained with K = {optimal_k}")


# =============================================================================
# STEP 5 — EVALUATE: Confusion Matrix + F1 Score
# =============================================================================
section("STEP 5 · Output Validation")

acc    = accuracy_score(y_test, predictions)
f1_mac = f1_score(y_test, predictions, average="macro")
cm     = confusion_matrix(y_test, predictions)

print(f"\nAccuracy : {GREEN}{acc:.4f}  ({acc*100:.2f} %){RESET}")
print(f"F1 Score : {GREEN}{f1_mac:.4f}  (macro average){RESET}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{classification_report(y_test, predictions, target_names=class_names)}")

misclassified = np.sum(predictions != y_test)
print(f"->  {misclassified} misclassification(s) out of {len(y_test)} test samples.")


# =============================================================================
# STEP 6 — SAMPLE PREDICTION
# =============================================================================
section("STEP 6 · Sample Prediction")

sample      = np.array([[5.1, 3.5, 1.4, 0.2]])   # classic Setosa
sample_scaled = scaler.transform(sample)
pred_class  = model.predict(sample_scaled)[0]
pred_proba  = model.predict_proba(sample_scaled)[0]

print(f"Input   : sepal_l={sample[0,0]}, sepal_w={sample[0,1]}, "
      f"petal_l={sample[0,2]}, petal_w={sample[0,3]}")
print(f"Predicted : {GREEN}{class_names[pred_class].upper()}{RESET}")
print("\nConfidence:")
for name, prob in zip(class_names, pred_proba):
    print(f"  {name:<12} {prob*100:.1f}%")


# =============================================================================
# STEP 7 — VISUALIZATIONS  (3-panel figure)
# =============================================================================
section("STEP 7 · Generating Visualizations")

fig = plt.figure(figsize=(18, 6), facecolor="#F0F4F8")
fig.suptitle(
    "Data Classification Using AI\nIris KNN Pipeline",
    fontsize=16, fontweight="bold", color="#0D2B55", y=1.01
)

gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# ── Panel 1: Elbow Curve ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(list(k_range), error_rates, color="#0D2B55", marker="o",
         markerfacecolor="#E84E1B", linewidth=2, markersize=6)
ax1.axvline(x=optimal_k, color="#E84E1B", linestyle="--", linewidth=1.5,
            label=f"Optimal K = {optimal_k}")
ax1.scatter([optimal_k], [error_rates[optimal_k - 1]],
            s=200, color="#E84E1B", zorder=5)
ax1.set_title("Elbow Curve — Choosing K", fontweight="bold", color="#0D2B55")
ax1.set_xlabel("K Value", color="#0D2B55")
ax1.set_ylabel("Error Rate", color="#0D2B55")
ax1.legend(framealpha=0.8)
ax1.set_facecolor("#F8FAFC")
ax1.grid(True, linestyle="--", alpha=0.5)

# ── Panel 2: Confusion Matrix ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
sns.heatmap(
    cm,
    annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names,
    linewidths=1, linecolor="#CBD5E0",
    ax=ax2, cbar=False,
    annot_kws={"fontsize": 13, "fontweight": "bold"},
)
ax2.set_title("Confusion Matrix", fontweight="bold", color="#0D2B55")
ax2.set_xlabel("Predicted Label", color="#0D2B55")
ax2.set_ylabel("True Label", color="#0D2B55")
ax2.set_facecolor("#F8FAFC")

# ── Panel 3: Feature Scaling Comparison ─────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
means_raw    = X.mean(axis=0)
means_scaled = np.abs(X_scaled.mean(axis=0))
short_names  = ["Sepal L", "Sepal W", "Petal L", "Petal W"]
x_pos        = np.arange(len(short_names))
width        = 0.35

bars1 = ax3.bar(x_pos - width/2, means_raw,    width, label="Raw",    color="#0D2B55", alpha=0.85)
bars2 = ax3.bar(x_pos + width/2, means_scaled, width, label="Scaled", color="#E84E1B", alpha=0.85)

ax3.set_title("Feature Means: Raw vs Scaled", fontweight="bold", color="#0D2B55")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(short_names, fontsize=9)
ax3.set_ylabel("Mean Value", color="#0D2B55")
ax3.legend()
ax3.set_facecolor("#F8FAFC")
ax3.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("iris_results.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Visualisation saved → iris_results.png")
plt.show()
