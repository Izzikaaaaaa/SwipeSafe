# train_paysim_50_50.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

XGBClassifier = None
try:
    from xgboost import XGBClassifier
except Exception as e:
    print("Warning: xgboost is unavailable and will be skipped:", e)

from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, roc_curve,
                             precision_recall_curve, auc, accuracy_score)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA       = "paysim.csv"
OUT_PREFIX = "paysim_50"

# ── Load & prepare data ───────────────────────────────────────────────────────
print("Loading", DATA)
df = pd.read_csv(DATA)

if df['isFraud'].sum() > len(df) / 2:
    print("Detected majority label=1 — flipping labels")
    df['isFraud'] = df['isFraud'].map({0: 1, 1: 0})

print("isFraud counts:", df['isFraud'].value_counts().to_dict())

df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
df['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)

keep = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'type_TRANSFER', 'type_CASH_OUT', 'isFraud']
df = df[keep]

fraud = df[df['isFraud'] == 1]
legit = df[df['isFraud'] == 0]
n_fraud = len(fraud)
if n_fraud == 0:
    raise SystemExit("No fraud rows found. Check your isFraud column.")

legit_sample = legit.sample(n=min(len(legit), n_fraud), random_state=42)
balanced = pd.concat([fraud, legit_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
balanced.to_csv(f"{OUT_PREFIX}_balanced.csv", index=False)
print("Balanced counts:", balanced['isFraud'].value_counts().to_dict())

balanced['Time']   = balanced['step'] * 60
balanced['Amount'] = balanced['amount']

X = balanced[['Time', 'Amount', 'oldbalanceOrg', 'newbalanceOrig',
              'oldbalanceDest', 'newbalanceDest', 'type_TRANSFER', 'type_CASH_OUT']]
y = balanced['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']]  = scaler.transform(X_test[['Time', 'Amount']])

joblib.dump(scaler, f"{OUT_PREFIX}_scaler.pkl")
print("Saved scaler.")

# ── Model comparison ──────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  MODEL COMPARISON")
print("="*55)

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight='balanced',
        random_state=42, n_jobs=1),
}

if XGBClassifier is not None:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=1, random_state=42,
        use_label_encoder=False, eval_metric='logloss')

results = {}

for name, clf in models.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.28).astype(int)
    acc   = accuracy_score(y_test, preds)
    roc   = roc_auc_score(y_test, probs)
    results[name] = {"model": clf, "probs": probs, "acc": acc, "roc": roc}
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {roc:.6f}")

# ── Print comparison table ────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Model':<20} {'Accuracy':>10} {'ROC-AUC':>10}")
print("-"*55)
for name, r in results.items():
    print(f"{name:<20} {r['acc']:>10.4f} {r['roc']:>10.6f}")
print("="*55)

# ── Plot model comparison bar chart ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
names     = list(results.keys())
accs      = [results[n]['acc'] for n in names]
rocs      = [results[n]['roc'] for n in names]
colors    = ['#4C72B0', '#DD8452', '#55A868']

axes[0].bar(names, accs, color=colors)
axes[0].set_title("Model Comparison — Accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0.8, 1.0)
for i, v in enumerate(accs):
    axes[0].text(i, v + 0.002, f"{v:.4f}", ha='center', fontsize=10)

axes[1].bar(names, rocs, color=colors)
axes[1].set_title("Model Comparison — ROC-AUC")
axes[1].set_ylabel("ROC-AUC")
axes[1].set_ylim(0.8, 1.0)
for i, v in enumerate(rocs):
    axes[1].text(i, v + 0.002, f"{v:.4f}", ha='center', fontsize=10)

plt.suptitle("Model Comparison: Decision Tree vs XGBoost vs Random Forest",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_model_comparison.png", dpi=150)
plt.close()
print("\nSaved model comparison chart.")

# ── Pick best model by ROC-AUC ────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]['roc'])
print(f"\n✅ Best model: {best_name}  (ROC-AUC = {results[best_name]['roc']:.6f})")
print("→ Saving as the production model.\n")

best_model = results[best_name]['model']
best_probs = results[best_name]['probs']
best_preds = (best_probs >= 0.28).astype(int)

joblib.dump(best_model, f"{OUT_PREFIX}_model.pkl")
print("Saved model:", f"{OUT_PREFIX}_model.pkl")

# ── Evaluation of best model ──────────────────────────────────────────────────
report = classification_report(y_test, best_preds, digits=4)
roc_score = roc_auc_score(y_test, best_probs)
cm = confusion_matrix(y_test, best_preds)

with open(f"{OUT_PREFIX}_classification_report.txt", "w") as f:
    f.write(f"Best Model: {best_name}\n")
    f.write("Classification Report (threshold=0.28)\n\n")
    f.write(report)
    f.write(f"\nROC-AUC: {roc_score:.6f}\n")

# Confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix — {best_name} (PaySim 50-50)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_confusion_matrix.png")
plt.close()

# ROC curve (all models)
plt.figure(figsize=(7, 5))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['probs'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['roc']:.4f})")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curves — All Models")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_roc_curve.png")
plt.close()

# PR curve (best model)
precision, recall, _ = precision_recall_curve(y_test, best_probs)
pr_auc = auc(recall, precision)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.6f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title(f"PR Curve — {best_name}")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_pr_curve.png")
plt.close()

# Feature importance (Random Forest or XGBoost)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": importances
    }).sort_values("importance", ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x="importance", y="feature", data=imp_df)
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_feature_importance.png")
    plt.close()

print("Done. All outputs saved with prefix:", OUT_PREFIX)
print(f"\nSUMMARY: {best_name} selected as best model with ROC-AUC = {roc_score:.6f}")
