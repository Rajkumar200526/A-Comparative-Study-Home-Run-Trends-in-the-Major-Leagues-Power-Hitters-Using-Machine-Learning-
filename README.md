# A-comparative-study-Home-run-Trends-in-the-Major-Leagues-Power-Hitters-Using-machine-Learning-
A comparative study:Home run Trends in the Major Leagues Power Hitters Using machine Learning 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score,mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv(r"C:\Users\ajaya\Downloads\Batting_converted.csv")
X=df.drop(columns=['ISO'])
y=df['ISO']
label_encoder = LabelEncoder()
y= label_encoder.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train_scaler=scaler.fit_transform(X_train)
X_test_scaler=scaler.transform(X_test)
LR=LinearRegression()
LR.fit(X_train_scaler,y_train)
Y_pred=LR.predict(X_test_scaler)
DT=DecisionTreeClassifier()
DT.fit(X_train,y_train)
y_pred=DT.predict(X_test)
print(f"Accuracy {accuracy_score(y_test,y_pred)}")
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1{f1}")
mae = mean_absolute_error(y_test,Y_pred)
mse = mean_squared_error(y_test,Y_pred)
r2 = r2_score(y_test,Y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
Accuracy 0.031746031746031744
F10.023280423280423283
Mean Absolute Error: 12.806381627309618
Mean Squared Error: 297.69340316634583
R-squared: 0.7810208953193403
