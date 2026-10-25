from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc, RocCurveDisplay,f1_score,precision_score)
import os
import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import plotting

def print_metrics(y_test,y_pred,y_proba,save_dir,model_name,clf):
    results_acc = {}
    results_f1 = {}
    results_presision={}
    results_specificity={}
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    presision = precision_score(y_test, y_pred)
    print(f"精确性（presision）: {presision:.4f}")
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    print(f"特异性 (Specificity): {specificity:.4f}")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    results_acc[model_name] = accuracy
    results_f1[model_name] = f1
    print(model_name, accuracy, report)

    # 保存报告
    with open(os.path.join(save_dir, f"{model_name}_report.txt"), "w") as f:
        f.write(f"Best Parameters: {clf.best_params_}\n\n{report}")