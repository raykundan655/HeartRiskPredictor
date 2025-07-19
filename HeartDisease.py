import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle


data=pd.read_csv("C:\\Users\\USER\\Downloads\\4th sem\\py\\heartProblem.csv")

# print(data.head(5))

# print(data.isnull().sum())

# print(data.describe())

print(data.dtypes)

Sex_le=LabelEncoder()

ChestPainType_le=LabelEncoder()

RestingECG_le=LabelEncoder()

ExerciseAngina_le=LabelEncoder()

ST_Slope_le=LabelEncoder()

# label_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
# for col in label_cols:
#     le = LabelEncoder()
#     data[col] = le.fit_transform(data[col])


# i have to do like this beacuse i need labelencoder for each categorical ->that required for backend

data['Sex']=Sex_le.fit_transform(data['Sex'])

data['ChestPainType']=ChestPainType_le.fit_transform(data['ChestPainType'])

data['RestingECG']=RestingECG_le.fit_transform(data['RestingECG'])

data['ExerciseAngina']=ExerciseAngina_le.fit_transform(data['ExerciseAngina'])

data['ST_Slope']=ST_Slope_le.fit_transform(data['ST_Slope'])

# sns.heatmap(data.corr(),annot=True)
# plt.show()

# NOTE
# No — a negative correlation does not mean the feature should be removed.
# What matters is how strongly it is related to the target and whether it adds useful predictive information.

# If feature A has a negative correlation with target Y:
# When A increases, Y tends to decrease (on average)
# Example: ExerciseAngina and HeartDisease
# If someone has angina while exercising, their risk of heart disease might increase, leading to a negative correlation with the "no disease" class.
# But this doesn’t mean the feature is useless.
# In fact, it might be very valuable for prediction.


#  Main Bases to Remove Features
#  1. Constant or Low Variance Features
# A feature whose values are all the same, or almost the same.

#  2. High Missing Values
# If a column has too many nulls/NaNs, its information is unreliable.
# Rule of thumb:
# More than 40% missing values → consider removin

# data.isnull().mean() * 100  # % of missing values

#  3. Irrelevant Features
# Features that have no logical connection to the target
# Don't affect the output in real life
# Examples:
# Patient ID, Name, File Path, Timestamp, Notes, etc.
# Random unique values with no repeat = no pattern

# 4.Highly Correlated Features (Multicollinearity)
# When two or more features give the same information

# Drop one of each correlated pair


# What is VIF?
# VIF measures how much a feature is linearly dependent on the other features.
# High VIF = multicollinearity.
# A VIF of:
# 1 → no correlation with other variables 
# 1–5 → moderate correlation, acceptable 
# >5 or >10 → high multicollinearity 

y=data['HeartDisease']

data.drop(columns='HeartDisease',axis=1,inplace=True)

vif=pd.DataFrame()

vif['features']=data.columns

# VIF works on arrays, but specifically on a 2D
vif['VIF']=[variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
# A 2D NumPy array of the entire DataFrame

# X: 2D array or DataFrame of independent variables.
# i: the index of the column (feature) in X for which you want to compute the VIF

# print(vif)

# MaxHR  26.142683 has high multi. vif

data.drop(columns='MaxHR',axis=1,inplace=True)

vif=pd.DataFrame() ##reset the vif

vif['features'] = data.columns

vif['VIF']=[variance_inflation_factor(data.values,i) for i in range(data.shape[1])]

# print(vif)

# RestingBP  39.969669 has high multi. vif

data.drop(columns='RestingBP',axis=1,inplace=True)

vif=pd.DataFrame() ##reset the vif

vif['features'] = data.columns

vif['VIF']=[variance_inflation_factor(data.values,i) for i in range(data.shape[1])]

# print(vif)

print(data.columns)

x=data[['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'RestingECG',
       'ExerciseAngina', 'Oldpeak', 'ST_Slope']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000)

# When you train a logistic regression model, the computer tries to find the best values (called coefficients) 
# by repeating calculations again and again

# max_iter=1000 means:
# “Allow the model to try up to 1000 times to find the best solution.”
# While learning, the model can repeat (iterate) its learning process up to 1000 times to find the best solution."

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("Accurate Score:",accuracy_score(y_pred,y_test))

# print("Classification report\n",classification_report(y_pred,y_test))


with open("HeartDiseaseModel.pkl","wb") as f:
    pickle.dump(model,f) 

with open("Sex_le.pkl","wb") as f:
    pickle.dump(Sex_le,f) 

with open("ChestPainType_le.pkl","wb") as f:
    pickle.dump(ChestPainType_le,f) 

with open("ExerciseAngina_le.pkl","wb") as f:
    pickle.dump(ExerciseAngina_le,f) 

with open("ST_Slope_le.pkl","wb") as f:
    pickle.dump(ST_Slope_le,f) 

with open("RestingECG_le.pkl","wb") as f:
    pickle.dump(RestingECG_le,f) 


















