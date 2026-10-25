import numpy as np
import tensorflow as tf
from tensorflow.keras import Model,layers,losses,optimizers
from models import specen_moll
import pandas as pd
from utils import data_preprocessing,spectral_embedding
import joblib
import os
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc, RocCurveDisplay,f1_score,precision_score,recall_score)
import matplotlib.pyplot as plt
from visualization import plotting

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
y_train=np.array(y_train,dtype=np.int32)
y_test=np.array(y_test,dtype=np.int32)

save_dir = r".\theta\specen_moll"
os.makedirs(save_dir, exist_ok=True)

models={
"RF": joblib.load(os.path.join(save_dir, f"RF_model.pkl")),
"SVM": joblib.load(os.path.join(save_dir, f"SVM_model.pkl")),
"XGB": joblib.load(os.path.join(save_dir, f"XGB_model.pkl")),
"LR": joblib.load(os.path.join(save_dir, f"LR_model.pkl")),
"KNN": joblib.load(os.path.join(save_dir, f"KNN_model.pkl"))
}
models_proba_train={"RF": [],"SVM":[],"XGB":[],"LR":[],"KNN":[]}
models_proba_test={"RF": [],"SVM":[],"XGB":[],"LR":[],"KNN":[]}



for model_name, model in models.items():
    if model_name in ["SVM", "LR", "KNN"]:
        y_proba_train = np.reshape(model.predict_proba(X_train_scaled)[:, 1], (-1, 1))
        y_proba_test = np.reshape(model.predict_proba(X_test_scaled)[:, 1], (-1, 1))
    else:
        y_proba_train = np.reshape(model.predict_proba(X_train)[:, 1], (-1, 1))
        y_proba_test = np.reshape(model.predict_proba(X_test)[:, 1], (-1, 1))
    models_proba_train[model_name].append(y_proba_train)
    models_proba_test[model_name].append(y_proba_test)

prior_proba_train=None
prior_proba_test=None
for proba_train,proba_test in zip(models_proba_train.values(),models_proba_test.values()):

    X_train_scaled=np.hstack((X_train_scaled,proba_train[0]))
    X_test_scaled = np.hstack((X_test_scaled, proba_test[0]))
    if prior_proba_train is None:
        prior_proba_train=proba_train[0]
        prior_proba_test=proba_test[0]
    else:
        prior_proba_train=np.hstack((prior_proba_train,proba_train[0]),dtype=np.float32)
        prior_proba_test=np.hstack((prior_proba_test,proba_test[0]),dtype=np.float32)

input_shape=X_train_scaled.shape[1]
output_shape=5
optimizer=optimizers.Adam(learning_rate=0.0001)
model=specen_moll.SpecEnMoll(input_shape, output_shape,optimizer)
model.build((None,input_shape))
model.summary()
epochs=10
batch_size=32
train_db=tf.data.Dataset.from_tensor_slices((X_train_scaled,y_train,prior_proba_train)).batch(batch_size)
test_db=tf.data.Dataset.from_tensor_slices((X_test_scaled,y_test,prior_proba_test)).batch(batch_size)

results={'train':[],'test':[]}
save_dir = r".\theta\specen_moll"
# Serialized SpecEn-Moll weights; figures go to results/
specen_result_dir = r".\results\ensemble\SpecEn-Moll"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(specen_result_dir, exist_ok=True)

for e in range(epochs):
    l_all=[]
    for x,y,p in train_db:
        l=model.train_step(x,y,p)
        l_all.append(l)
    y_train_pred=tf.cast(tf.reduce_sum(model.predict(X_train_scaled,batch_size=batch_size)*prior_proba_train,axis=-1)>=0.5,dtype=tf.int32)
    y_test_pred = tf.cast(tf.reduce_sum(model.predict(X_test_scaled,batch_size=batch_size)*prior_proba_test,axis=-1)>=0.5,dtype=tf.int32)
    print("Epoch:{}/{}".format(e+1,epochs))
    print("Loss:{:.4}".format(np.mean(l_all)),"Training Accuracy:{:.4}".format(np.mean(y_train_pred==y_train)), "Testing Accuracy:{:.4}".format(np.mean(y_test_pred==y_test)))
model.save_weights(os.path.join(save_dir,f"SpecEn-Moll.ckpt"))

y_train_pred = tf.cast(
    tf.reduce_sum(model.predict(X_train_scaled, batch_size=batch_size) * prior_proba_train, axis=-1) >= 0.5,
    dtype=tf.int32)
y_train_proba=tf.reduce_sum(model.predict(X_train_scaled, batch_size=batch_size) * prior_proba_train, axis=-1)
y_test_pred = tf.cast(
    tf.reduce_sum(model.predict(X_test_scaled, batch_size=batch_size) * prior_proba_test, axis=-1) >= 0.5,
    dtype=tf.int32)
y_test_proba=tf.reduce_sum(model.predict(X_test_scaled, batch_size=batch_size) * prior_proba_test, axis=-1)


accuracy = accuracy_score(y_train, y_train_pred)
f1=f1_score(y_train, y_train_pred)
recall=recall_score(y_train, y_train_pred)

presision=precision_score(y_train, y_train_pred)
report = classification_report(y_train, y_train_pred)
cm = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
fpr, tpr, _ = roc_curve(y_train, y_train_proba)
roc_auc = auc(fpr, tpr)

# print(model_name,accuracy,report)
results['train'] = [accuracy, f1, roc_auc,presision, specificity, recall]
#test dataset
accuracy = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
presision = precision_score(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)
with open(os.path.join(save_dir, f"SpecEn-Moll_report_holdout.txt"), "w") as f:
    f.write(f"{report}")
results['test']=[accuracy,f1,roc_auc,presision,specificity,recall]
np.save(os.path.join(specen_result_dir, "roc_fpr.npy"), fpr)
np.save(os.path.join(specen_result_dir, "roc_tpr.npy"), tpr)
print(results)

