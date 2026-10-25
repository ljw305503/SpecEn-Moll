from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski

def _attach_embedding(df, x_embedding):
    feature_name = [f"feature_{i}" for i in range(x_embedding.shape[1])]
    df_embedding = pd.DataFrame(x_embedding, columns=feature_name)
    df = pd.concat([df.reset_index(drop=True), df_embedding], axis=1)
    print("=" * 40)
    print("原始数据缺失值统计：")
    print(df.isnull().sum())
    print("\n删除缺失值前的数据量:", len(df))
    df = df.dropna()
    print("删除缺失值后的数据量:", len(df))
    print("\n处理后的缺失值统计：")
    print(df.isnull().sum())
    print("=" * 40)
    return df, feature_name

def _encode_objects_from_train(X_train, X_test):
    object_cols = list(X_train.select_dtypes(include=["object"]).columns)
    for col in object_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        mapping = {cls: i for i, cls in enumerate(le.classes_)}
        X_test[col] = X_test[col].map(lambda v: mapping.get(v, -1))
    return X_train, X_test

def _scale_text_keep_graph(X_train, X_test, y_train, y_test, feature_name):
    train_graph_feature = X_train[feature_name]
    test_graph_feature = X_test[feature_name]
    train_text_feature = X_train.drop(columns=feature_name)
    test_text_feature = X_test.drop(columns=feature_name)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_text_feature)
    X_test_scaled = scaler.transform(test_text_feature)
    X_train_scaled = np.concatenate((X_train_scaled, train_graph_feature), axis=1)
    X_test_scaled = np.concatenate((X_test_scaled, test_graph_feature), axis=1)
    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test

def preprocess(df, x_embedding=None, df_test=None, x_embedding_test=None):
    """If df_test is provided, use the on-disk train/test split; otherwise split internally."""
    if df_test is not None:
        df_train, feature_name = _attach_embedding(df, x_embedding)
        df_test, _ = _attach_embedding(df_test, x_embedding_test)
        X_train = df_train.drop(columns=["lable"])
        y_train = df_train["lable"]
        X_test = df_test.drop(columns=["lable"])
        y_test = df_test["lable"]
        X_train, X_test = _encode_objects_from_train(X_train.copy(), X_test.copy())
        return _scale_text_keep_graph(X_train, X_test, y_train, y_test, feature_name)

    df, feature_name = _attach_embedding(df, x_embedding)
    X = df.drop(columns=["lable"])
    y = df["lable"]
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return _scale_text_keep_graph(X_train, X_test, y_train, y_test, feature_name)

def preprocess_1(df,x_embedding=None):
    feature_name=[]
    for i in range(x_embedding.shape[1]):
        feature_name.append('feature_'+str(i))
    df_embedding=pd.DataFrame(x_embedding,columns=feature_name)
    df=pd.concat([df,df_embedding],axis=1)
    # 检查缺失值
    print("=" * 40)
    print("原始数据缺失值统计：")
    print(df.isnull().sum())

    # 删除含有缺失值的行
    print("\n删除缺失值前的数据量:", len(df))
    df = df.dropna()
    print("删除缺失值后的数据量:", len(df))

    # 验证处理结果
    print("\n处理后的缺失值统计：")
    print(df.isnull().sum())
    print("=" * 40)

    # 2. 数据预处理
    X = df
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])

    X_graph_feature=X[feature_name]
    X_text_feature=X.drop(columns=feature_name)
    # 标准化数据（适用于需要标准化的模型）
    scaler = StandardScaler()
    X_text_scaled = scaler.fit_transform(X_text_feature)
    X_scaled=np.concatenate((X_text_scaled,X_graph_feature),axis=1)
    return X_scaled,X


def preprocess_2():
    # 读取原始数据
    df = pd.read_excel('../dataset/train/Pomacea canaliculata compounds analysis data.xlsx')
    # 初始化计算列
    df['MW'] = None
    df['logP'] = None
    df['logD'] = None
    df['HBA'] = None
    df['HBD'] = None

    # 定义logD计算函数（默认pH=7.4）
    def calculate_logd(smiles, pH=7.4):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Descriptors.MolLogD(mol, pH)
        except:
            return None

    # 遍历计算各属性
    error_count = 0
    for idx, smiles in enumerate(df['smiles']):
        try:
            mol = Chem.MolFromSmiles(smiles)
            # 分子量
            df.at[idx, 'MW'] = Descriptors.MolWt(mol)
            # logP
            df.at[idx, 'logP'] = Descriptors.MolLogP(mol)
            # logD (pH=7.4)
            df.at[idx, 'logD'] = calculate_logd(smiles)
            # 氢键受体和供体
            df.at[idx, 'HBA'] = Lipinski.NumHAcceptors(mol)
            df.at[idx, 'HBD'] = Lipinski.NumHDonors(mol)
        except Exception as e:
            error_count += 1
            print(f"Error processing SMILES {idx}: {str(e)}")
    print(f"成功处理 {len(df) - error_count} 条记录，{error_count} 条记录出现错误")
    # 保存结果
    output_path = '../dataset/train/fsl-1/Pomacea canaliculata compounds analysis data-1.csv'
    df.to_csv(output_path, index=False)
    print(f"结果已保存至：{output_path}")