import os
import pandas as pd
from utils import data_preprocessing
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc, RocCurveDisplay)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r".\dataset\train\all compounds analysis.xlsx"
df = pd.read_excel(file_path)

X_train_scaled,X_test_scaled,X_train, X_test, y_train, y_test=data_preprocessing.preprocess(df)

save_dir = r".\theta\specen_moll"
os.makedirs(save_dir, exist_ok=True)

models={
"RF": joblib.load(os.path.join(save_dir, f"RF_model.pkl")),
"SVM": joblib.load(os.path.join(save_dir, f"SVM_model.pkl")),
"XGB": joblib.load(os.path.join(save_dir, f"XGB_model.pkl")),
"LR": joblib.load(os.path.join(save_dir, f"LR_model.pkl")),
"KNN": joblib.load(os.path.join(save_dir, f"KNN_model.pkl"))
}
results = {}
result_save_dir=r".\results\test"
for model_name, model in models.items():
    if model_name in ["SVM", "LR", "KNN"]:
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    results[model_name] = accuracy


    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(result_save_dir, f"{model_name}_cm.png"))
    plt.close()


    RocCurveDisplay.from_estimator(model, X_test_scaled if model_name in ["SVM", "LR", "KNN"] else X_test,
                                   y_test, name=model_name)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve (AUC = {roc_auc:.2f})")
    plt.savefig(os.path.join(result_save_dir, f"{model_name}_roc.png"))
    plt.close()


    plt.figure(1)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")