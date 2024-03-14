import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline
import warnings
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
warnings.filterwarnings('ignore')

# Importing DataSet
df=pd.read_csv("./kidney_disease.csv")

df.isnull().values.any()
df.isnull().sum().sort_values(ascending=False)

mis_val_percent=((df.isnull().sum()/df.shape[0])*100).sort_values(ascending=False)

for x in ['rc','wc','pcv']:
    df[x] = df[x].str.extract('(\d+)').astype(float)

df.drop(["id"],axis=1,inplace=True)

# Filling missing numeric data in the dataset with mean
for x in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:
    df[x].fillna(df[x].mean(),inplace=True)

df.isnull().sum().sort_values(ascending=False)

df['cad'] = df['cad'].replace(to_replace='\tno',value='no')

df['dm'] = df['dm'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})

df['classification'] = df['classification'].replace(to_replace = 'ckd\t', value = 'ckd')

# Cleaning & Pre-Processing of Data.

df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
df['classification'] = df['classification'].replace(to_replace={'ckd':1,'notckd':0})
e = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
df[e] = df[e].astype("O")
df['classification'] = df['classification'].astype("int")

# Visualization
corr = df.corr()
plt.figure(figsize=(18,10))
sns.heatmap(corr, annot=True)

cat_cols = [col for col in df.columns if df[col].dtypes == 'O']

fig, axes = plt.subplots(5, 2, figsize=(12,18))
fs = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
for i, axi in enumerate(axes.flat):
    sns.countplot(x=fs[i], hue='classification', data=df, palette='prism', ax=axi) 
    axi.set(ylabel='Frequency')
    axi.legend(["Not Disease", "Disease"])
    
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in "id"]

numeric_cols1= ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']

def hist_for_nums(data, numeric_cols1):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols1:
        data[col].hist(bins=3)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")
hist_for_nums(df, numeric_cols1)

df["classification"].value_counts()

fig1, ax1 = plt.subplots()
ax1.pie(df["classification"].value_counts(),  labels=['CKD','NOTCKD'], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Number of Categorical Variables : ', len(cat_cols))


def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df= pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
df, cat_cols = one_hot_encoder(df, nan_as_category= True)


X = df.drop('classification', axis=1)
y = df[["classification"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)
X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_scaled)

lgbm = LGBMClassifier(random_state=46)
lgbm.fit(X_train, y_train)
Lgbm_pred = lgbm.predict(X_test)
acc_lgbm = round(lgbm.score(X_train, y_train) * 100, 2)


xgboost = XGBClassifier(random_state=46)
xgboost.fit(X_train, y_train)
xgboost_pred = xgboost.predict(X_test)
acc_xgboost = round(xgboost.score(X_train, y_train) * 100, 2)


DecisionTree = XGBClassifier(random_state=46)
DecisionTree.fit(X_train, y_train)
DecisionTree_pred = DecisionTree.predict(X_test)
acc_DecisionTree = round(DecisionTree.score(X_train, y_train) * 100, 2)


models = pd.DataFrame({
    'Model': ["LightGBM",'Decision Tree', "XGBOOST"],
    'Score': [acc_lgbm,acc_DecisionTree, acc_xgboost]})
models.sort_values(by='Score', ascending=False)


# Web Page

st.title("Chronic Kidney Disease.")
input_df = st.text_input("1")
input_df = st.text_input("2")
input_df = st.text_input("3")
input_df = st.text_input("4")
input_df = st.text_input("5")
input_df = st.text_input("6")
input_df = st.text_input("7")
input_df = st.text_input("8")
input_df = st.text_input("9")
input_df = st.text_input("10")
input_df = st.text_input("11")
input_df = st.text_input("12")
input_df = st.text_input("13")
input_df = st.text_input("14")
input_df = st.text_input("15")
input_df = st.text_input("16")
input_df = st.text_input("17")
input_df = st.text_input("18")
input_df = st.text_input("19")
input_df = st.text_input("20")
input_df = st.text_input("21")
input_df = st.text_input("22")
input_df = st.text_input("23")
input_df = st.text_input("24")

submit = st.button("Submit")

if submit:
    features = np.asarray(input_df)
