import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
data=pd.read_csv(r"/content/diabetes (1).csv")
correlation=data.corr()
from sklearn.model_selection import train_test_split
X=data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
Y=data.Outcome
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
import pickle
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(model, file)