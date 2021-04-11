import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('Lending_Club_Loan_approval_Optimization.csv')

from sklearn.model_selection import train_test_split

X = df.drop("Target", axis=1)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=30)

from sklearn.ensemble import RandomForestClassifier

Model_RF = RandomForestClassifier(n_estimators=500, max_features=None, max_depth=6, bootstrap=True)
Model_RF.fit(X_train, y_train)
y_pred = Model_RF.predict(X_test)
pickle.dump(Model_RF, open('model.pkl','wb'))

# def prediction(config, model):

#     if type()
#     return y_pred