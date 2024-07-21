import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics
from xgboost import XGBRegressor  


data = pd.read_csv("dataset.csv")


X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values    


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


xgb = XGBRegressor()


xgb.fit(X_train, y_train)


pickle.dump(xgb, open('model.pkl', 'wb'))
