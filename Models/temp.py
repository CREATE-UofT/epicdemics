import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import xgboost as xgb 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 

data = pd.read_csv("Data/FID/good.csv")
print(data.head(6))

x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=['ILI_CASE']), data["ILI_CASE"], test_size=.2)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
bst = xgb.XGBClassifier(objective="multi:softmax")
bst.fit(x_train,y_train)
print(bst.predict(xtest))
