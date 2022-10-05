import numpy as np
import matplotlib as plt
import pandas as pd
from  sklearn import  svm

data = pd.read_csv('iris.csv')
# X = data.columns()
X = data.iloc[:,0:2]
X = np.array(X)
Y = data.iloc[:,-1]
Y = np.array(Y)

C = 1.0
svc = svm.SVC(kernel='linear', C = 1, gamma = "auto").fit(X, Y)

