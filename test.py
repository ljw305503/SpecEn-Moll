import os
import pandas as pd

from utils import data_preprocessing,spectral_embedding
from models import other_models,specen_moll
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc, RocCurveDisplay,f1_score,precision_score,recall_score)
import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from utils import pc_gnn
import numpy as np


input_file = "./dataset/test/text_feature.xlsx"
graph_path = "./dataset/test/pc-gnn.pt"
k=2
pc_gnn.get_pc_gnn_1(input_file,graph_path)
df = pd.read_excel(input_file)
x_embedding=spectral_embedding.get_embedding(graph_path,input_file,k)

X_scaled,X=data_preprocessing.preprocess_1(df,x_embedding)

save_dir = r".\theta\specen_moll"
os.makedirs(save_dir, exist_ok=True)

models={
"RF": joblib.load(os.path.join(save_dir, f"RF_model.pkl")),
"SVM": joblib.load(os.path.join(save_dir, f"SVM_model.pkl")),
"XGB": joblib.load(os.path.join(save_dir, f"XGB_model.pkl")),
"LR": joblib.load(os.path.join(save_dir, f"LR_model.pkl")),
"KNN": joblib.load(os.path.join(save_dir, f"KNN_model.pkl"))
}

models_proba_test={"RF": [],"SVM":[],"XGB":[],"LR":[],"KNN":[]}

results = {"A/IA":None}

voting_result_path = r".\results\prediction\ensemble\voting\result_ac.xlsx"
specen_result_path = r".\results\prediction\ensemble\SpecEn-Moll\result_ac.xlsx"
os.makedirs(os.path.dirname(voting_result_path), exist_ok=True)
os.makedirs(os.path.dirname(specen_result_path), exist_ok=True)
cache=np.zeros(len(X),dtype=np.int32)
for model_name, model in models.items():
    if model_name in ["SVM", "LR", "KNN"]:
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
    else:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
    models_proba_test[model_name].append(np.reshape(y_proba,(-1,1)))
    cache+=y_pred
cache[cache<4]=0
cache[cache>=4]=1
results["A/IA"]=cache

specen_results={"A/IA":None}
prior_proba_test=None
for proba_test in models_proba_test.values():
    X_scaled = np.hstack((X_scaled, proba_test[0]))
    if prior_proba_test is None:
        prior_proba_test=proba_test[0]
    else:
        prior_proba_test=np.hstack((prior_proba_test,proba_test[0]),dtype=np.float32)

input_shape=X_scaled.shape[1]
output_shape=5
specen=specen_moll.SpecEnMoll(input_shape, output_shape)
specen.load_weights(os.path.join(save_dir,f"SpecEn-Moll.ckpt"))
y_pred=specen.predict1(X_scaled,prior_proba_test,32)
specen_results["A/IA"]=y_pred

results=pd.DataFrame(results)
simle=pd.DataFrame(df['smiles'])
results=pd.concat([simle,results],axis=1)
results.to_excel(voting_result_path,index=False)

specen_results=pd.DataFrame(specen_results)
simle=pd.DataFrame(df['smiles'])
specen_results=pd.concat([simle,specen_results],axis=1)
specen_results.to_excel(specen_result_path,index=False)




