import pandas as pd
import numpy as np
df=pd.read_csv("Bank Customer Churn Prediction.csv")
df.columns
df.shape
df.describe()
df.head()
df.drop("customer_id",axis=1,inplace=True)
df.isna().sum() # no missing values are here in this 
df["products_number"].unique() 
df["gender"].unique()

df.columns
# data visualization 
import matplotlib.pyplot as plt
import seaborn as sns

df["country"].value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.show()

sns.countplot(data=df,x='country',hue='churn')
plt.show()  

sns.countplot(data=df,x='gender',hue='churn')
plt.show() 
 
sns.histplot(data=df,x='age',hue='churn',bins=20,kde=True)
plt.show()

sns.countplot(data=df,x="products_number",hue='churn')
plt.show()

sns.boxplot(x='churn',y='estimated_salary',data=df)
plt.show() 

sns.histplot(data=df,x='credit_score',bins=20,kde=True)
plt.show()

numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Final Correlation Heatmap')
plt.show()


df["gender"]=df["gender"].map({'Male':1,'Female':0})
df["country"].unique() # france , spain , germany 
#label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder() 
df["country"]=le.fit_transform(df["country"]) 
df["country"]

df.head()
df.iloc[0]
df.dtypes
x=df.drop("churn",axis=1)
y=df["churn"] 
x.head()
x.columns
# train and splitting the data set 
from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.columns
# scaling 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_train 
x_test=scaler.transform(x_test)
x_test

# training the model ...
#1) using logistic regression 
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test) # poor score !!
#using RandomforestClasssifier
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,
                             max_depth=10,random_state=42,class_weight='balanced')
model.fit(x_train,y_train)
model.score(x_test,y_test) 

# doing grid search cv for finding the best huper tunning parameters of randomforestclassifier
from sklearn.model_selection import RandomizedSearchCV
param_grid={
    'n_estimators':[100,50,150],
    'max_depth':[10,12,15],
    'criterion':['gini','entropy'],
    'class_weight':['balanced']
}
grid_search=RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                         param_distributions=param_grid,cv=5,n_jobs=5,return_train_score=False,n_iter=5)
grid_search.fit(x_train,y_train) 
# checking the results 
grid_search.best_params_
grid_search.best_score_



# now training the final model 
final_model=RandomForestClassifier(
    n_estimators=150,max_depth=150,
    criterion='entropy',class_weight='balanced',
    random_state=42)
final_model.fit(x_train,y_train)
final_model.score(x_test,y_test) 

# changing the threshold....necessary for bank churn
y_probs = final_model.predict_proba(x_test)[:, 1] 
custom_threshold = 0.38 
y_pred_custom = (y_probs >= custom_threshold).astype(int)


# confusion matrix plot 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_custom) 
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('truth')
plt.show()

# doing prediction 
new_customer=[[600,0,1,42,3,60000,1,1,0,50000]]
new_customer_scaled=scaler.transform(new_customer)

prob = final_model.predict_proba(new_customer_scaled)[:, 1]
prediction = (prob >= 0.38).astype(int)
prediction 

 
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_custom)

from sklearn.metrics import classification_report
classification_report(y_test, y_pred_custom) 



# saving the model 
import joblib

joblib.dump(final_model, 'bank_churn_model.pkl',compress=3)

# 2. Save the scaler 
joblib.dump(scaler, 'scaler.pkl')
