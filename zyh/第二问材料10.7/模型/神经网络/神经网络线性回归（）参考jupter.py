import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchensemble import GradientBoostingRegressor,BaggingRegressor,FusionRegressor,VotingRegressor,SnapshotEnsembleRegressor

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# 2. Define the ensemble
mlp_model = BaggingRegressor(
    estimator=MLP,
    n_estimators=10,
    cuda=True,
)

data = pd.read_csv("clean_data1.csv", encoding="utf_8_sig")[0:123]
data1 = pd.read_csv("clean_data1.csv", encoding="utf_8_sig")[123:]
# print(data1)
X=data.iloc[:,data.keys()!='10cm湿度(kg/m2)']

Res_x = data1.iloc[:,data.keys()!='10cm湿度(kg/m2)']
Res_x=torch.FloatTensor(np.array(Res_x))


Y = data["10cm湿度(kg/m2)"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
print("X Train: ", X_train.shape)
print("Y Train: ", Y_train.shape)
print("X Test: ", X_test.shape)
print("Y Test: ", Y_test.shape)
X_train=torch.FloatTensor(np.array(X_train))
X_test=torch.FloatTensor(np.array(X_test))
Y_train=torch.FloatTensor(np.array(Y_train)).reshape(-1, 1)
Y_test=torch.FloatTensor(np.array(Y_test)).reshape(-1, 1)
train_data = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

test_data = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# 4. 设置优化器optimizer
lr = 1e-3
weight_decay = 5e-4
mlp_model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)
mlp_model.fit(train_loader,epochs=100)
testing_mse = mlp_model.evaluate(test_loader)
print(testing_mse)

Y_pred = mlp_model.predict(X_test)
Res_pred = mlp_model.predict(Res_x)
model_MLP_vote = pd.DataFrame(np.array(X_test))
model_MLP_vote['10cm湿度(kg/m2)'] = np.array(Y_test)
model_MLP_vote['Predicted 10cm湿度(kg/m2)'] = np.array(Y_pred)
print(np.array(Res_pred))
print(model_MLP_vote)