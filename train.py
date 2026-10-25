import os
import pandas as pd
from utils import data_preprocessing,spectral_embedding
from models import other_models
# import other_models
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc, RocCurveDisplay,f1_score,precision_score,recall_score)
import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import plotting
import numpy as np

def metrics(y_true, y_pred,y_probe=None):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    presision = precision_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    roc_auc=0
    fpr=0
    tpr=0
    if y_probe is not None:
        fpr, tpr, _ = roc_curve(y_true, y_probe)
        roc_auc = auc(fpr, tpr)
    return accuracy, f1, roc_auc,presision, specificity, recall,cm,fpr, tpr,report


plt.rcParams['font.family'] = 'Times New Roman'

file_path = r".\dataset\train\text_feature.xlsx"
file_path1=r".\dataset\train\all compounds analysis.xlsx"
graph_path=r".\dataset\train\pc-gnn.pt"
file_path_te = r".\dataset\test\text_feature.xlsx"
file_path1_te=r".\dataset\test\all compounds analysis.xlsx"
graph_path_te=r".\dataset\test\pc-gnn.pt"
k=2
df = pd.read_excel(file_path)
df_test = pd.read_excel(file_path_te)
x_embedding=spectral_embedding.get_embedding(graph_path,file_path1,k)
x_embedding_test=spectral_embedding.get_embedding(graph_path_te,file_path1_te,k)

X_train_scaled,X_test_scaled,X_train, X_test, y_train, y_test=data_preprocessing.preprocess(
    df, x_embedding, df_test, x_embedding_test)
models= other_models.get_models()

results_test_acc = {}
results_test_f1={}
results_test_presision={}
results_test_specificity={}
results_test_recall={}

results_train_acc = {}
results_train_f1={}
results_train_presision={}
results_train_specificity={}
results_train_recall={}

results={'RF':{'train':[],'test':[]},'SVM':{'train':[],'test':[]},'XGB':{'train':[],'test':[]},
         'LR':{'train':[],'test':[]},'KNN':{'train':[],'test':[]}}

# Serialized classifiers; figures go to results/
save_dir = r".\theta\specen_moll"
base_result_dir = r".\results\base_models"
comparison_dir = r".\results\comparison_holdout"
specen_result_dir = r".\results\ensemble\SpecEn-Moll"
os.makedirs(save_dir, exist_ok=True)
for _d in (base_result_dir, comparison_dir, specen_result_dir):
    os.makedirs(_d, exist_ok=True)
voting_results={'train':None,'test':None}
pred_cache={'train':np.zeros(len(X_train),dtype=np.int32),'test':np.zeros(len(X_test),dtype=np.int32)}
prob_cache={'train':None,'test':None}
for model_name, (model, params) in models.items():

    clf = GridSearchCV(model, params, cv=3, scoring="accuracy")
    if model_name in ["SVM", "LR", "KNN"]:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]

        y_train_pred=clf.predict(X_train_scaled)
        y_train_proba=clf.predict_proba(X_train_scaled)[:, 1]
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        y_train_pred = clf.predict(X_train)
        y_train_proba = clf.predict_proba(X_train)[:, 1]

    joblib.dump(clf, os.path.join(save_dir, f"{model_name}_model.pkl"))
    if prob_cache['train'] is None:
        prob_cache['train']=np.reshape(y_train_proba,(-1,1))
        prob_cache['test']=np.reshape(y_proba,(-1,1))
    else:
        prob_cache['train']=np.hstack((prob_cache['train'],np.reshape(y_train_proba,(-1,1))))
        prob_cache['test']=np.hstack((prob_cache['test'],np.reshape(y_proba,(-1,1))))

    pred_cache['train'] += y_train_pred
    pred_cache['test'] += y_pred

    #train dataset
    accuracy, f1, roc_auc, presision, specificity, recall, cm,fpr, tpr,report=metrics(y_train,y_train_pred,y_train_proba)
    results_train_acc[model_name] = accuracy
    results_train_f1[model_name] = f1
    results_train_presision[model_name] = presision
    results_train_specificity[model_name] = specificity
    results_train_recall[model_name]=recall
    # print(model_name,accuracy,report)
    results[model_name]['train'] = [accuracy, f1, roc_auc,presision, specificity, recall]
    #test dataset
    accuracy, f1, roc_auc, presision, specificity, recall, cm,fpr, tpr,report = metrics(y_test, y_pred, y_proba)
    results_test_acc[model_name] = accuracy
    results_test_f1[model_name] = f1
    results_test_presision[model_name] = presision
    results_test_specificity[model_name] = specificity
    results_test_recall[model_name] = recall

    results[model_name]['test']=[accuracy,f1,roc_auc,presision,specificity,recall]


    with open(os.path.join(save_dir, f"{model_name}_report_holdout.txt"), "w") as f:
        f.write(f"Best Parameters: {clf.best_params_}\n\n{report}")


    model_result_dir = os.path.join(base_result_dir, model_name)
    os.makedirs(model_result_dir, exist_ok=True)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d",cmap='Pastel1')
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(model_result_dir, "confusion_matrix.png"))
    plt.close()


    # RocCurveDisplay.from_estimator(clf, X_test_scaled if model_name in ["SVM", "LR", "KNN"] else X_test,
    #                                y_test, name=model_name)
    # plt.plot([0, 1], [0, 1], "k--")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title(f"{model_name} ROC Curve (AUC = {roc_auc:.2f})")
    # plt.savefig(os.path.join(result_save_dir, f"{model_name}_roc.png"))
    # plt.close()


    plt.figure(1)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

prob_cache['train']=np.mean(np.sort(prob_cache['train'],axis=-1)[:,:3],axis=-1)
prob_cache['test']=np.mean(np.sort(prob_cache['test'],axis=-1)[:,:3],axis=-1)
pred_cache['train'][pred_cache['train'] < 3] = 0
pred_cache['train'][pred_cache['train'] >= 3] = 1
pred_cache['test'][pred_cache['test'] < 3] = 0
pred_cache['test'][pred_cache['test'] >= 3] = 1

accuracy, f1, roc_auc, presision, specificity, recall, cm,fpr, tpr,report=metrics(y_test,pred_cache['test'],prob_cache['test'])
results_test_acc['VOTING'] = accuracy
results_test_f1["VOTING"] = f1
results_test_presision['VOTING'] = presision
results_test_specificity['VOTING'] = specificity
results_test_recall['VOTING']=recall
voting_results["test"] = [accuracy,f1,roc_auc,presision,specificity,recall]
with open(os.path.join(save_dir, f"Voting_report_holdout.txt"), "w") as f:
    f.write(f"{report}")
plt.plot(fpr, tpr, label=f"VOTING(AUC = {roc_auc:.2f})")
accuracy, f1, roc_auc, presision, specificity, recall, cm,fpr, tpr,report=metrics(y_train,pred_cache['train'],prob_cache['train'])

voting_results["train"] = [accuracy,f1,roc_auc,presision,specificity,recall]
_fpr_path = os.path.join(specen_result_dir, "roc_fpr.npy")
_tpr_path = os.path.join(specen_result_dir, "roc_tpr.npy")
if os.path.isfile(_fpr_path) and os.path.isfile(_tpr_path):
    fpr=np.load(_fpr_path)
    tpr=np.load(_tpr_path)
    plt.plot(fpr, tpr, label=f"SpecEn-Moll(AUC = {roc_auc:.2f})")
print(voting_results)

results_test_acc['SpecEn-Moll'] = 0.8718
results_test_f1["SpecEn-Moll"] = 0.84375
results_test_presision['SpecEn-Moll'] = 0.84375
results_test_specificity['SpecEn-Moll'] = 0.8913
results_test_recall['SpecEn-Moll']=0.8438

plotting.plot1(comparison_dir)
acc_base_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33F6', '#33FFF6']
acc_title="Model Accuracy Comparison"
acc_y_label="Accuracy"
acc_x_label="Model"
acc_file_name='accuracy'
plotting.plot4(comparison_dir,results_test_acc,acc_base_colors,acc_title,acc_y_label,acc_x_label,acc_file_name)

f1_base_colors = ['#1F77B4', '#FF7F0E', '#E377C2', '#D62728', '#17BECF']
f1_title="Model F1-Score Comparison"
f1_y_label='F1 Score'
f1_x_label='Model'
f1_file_name='f1_score'
plotting.plot4(comparison_dir,results_test_f1,f1_base_colors,f1_title,f1_y_label,f1_x_label,f1_file_name)

presision_base_colors = ['#1F77B4', '#FF7F0E', '#E377C2', '#D62728', '#17BECF']
presision_title="Model Presision Comparison"
presision_y_label='Presision'
presision_x_label='Model'
presision_file_name='precision'
plotting.plot4(comparison_dir,results_test_presision,presision_base_colors,presision_title,presision_y_label,presision_x_label,presision_file_name)

specificity_base_colors = ['#1F77B4', '#FF7F0E', '#E377C2', '#D62728', '#17BECF']
specificity_title="Model Specificity Comparison"
specificity_y_label='Specificity'
specificity_x_label='Model'
specificity_file_name='specificity'
plotting.plot4(comparison_dir,results_test_specificity,specificity_base_colors,specificity_title,specificity_y_label,specificity_x_label,specificity_file_name)

recall_base_colors = ['#1F77B4', '#FF7F0E', '#E377C2', '#D62728', '#17BECF']
recall_title="Model Recall Comparison"
recall_y_label='Recall'
recall_x_label='Model'
recall_file_name='recall'
plotting.plot4(comparison_dir,results_test_recall,recall_base_colors,recall_title,recall_y_label,recall_x_label,recall_file_name)

# colors = [(tuple(int(c[1:][i:i + 2], 16) / 255 for i in (0, 2, 4)) + (0.5,)) for c in
#               recall_base_colors]
# labels=['Acc','F1','Presision','Specificity','Recall']
# for model_name,c in zip(results.keys(),colors):
#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
#     train=results[model_name]['train']
#     test=results[model_name]['test']
#     print(c)
#     plotting.plot_radar(fig, ax,c,labels,train,model_name)
#     plotting.plot_radar(fig, ax,c+1,labels, test,model_name)
#     plt.show()

