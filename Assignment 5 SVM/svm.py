import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


data_path = "./"
train = pd.read_csv(data_path + 'train_sample.csv')
test = pd.read_csv(data_path + 'test_sample.csv')
sc = StandardScaler()

feats = ["x0", "x1"]
train[feats] = sc.fit_transform(train[feats])
test[feats] = sc.transform(test[feats])

params = {'gamma':.1**np.arange(-1,4), 'C': 5*np.arange(3,6)}
svc = GridSearchCV(svm.SVC(), params, scoring="accuracy", cv=[(slice(None), slice(None))])
svc.fit(train[feats], train["class"])

test_prediction = svc.predict(test[feats])

test['class'] = test_prediction
test[['ID','class']].to_csv('submission.csv', index=False)