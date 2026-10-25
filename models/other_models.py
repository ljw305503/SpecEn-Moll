from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def get_models():
    models = {
        "RF": (RandomForestClassifier(),
               {"n_estimators": [100, 200], "max_depth": [None, 10]}),
        "SVM": (SVC(probability=True),
                {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
        "XGB": (XGBClassifier(),
                {"learning_rate": [0.01, 0.1], "max_depth": [3, 5]}),
        "LR": (LogisticRegression(max_iter=1000),  # 增加最大迭代次数
               {"C": [0.1, 1, 10], "penalty": ["l2"]}),
        "KNN": (KNeighborsClassifier(),
                {"n_neighbors": [3, 5, 7]})
    }
    return models