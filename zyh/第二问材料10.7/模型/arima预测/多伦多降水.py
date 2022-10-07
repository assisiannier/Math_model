import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")
#将时间列转为标准格式 %Y-%m-%d
def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')
#加载数据
rainfall = pd.read_csv("rain_21_22.csv",parse_dates=[0], index_col=0, date_parser=parser)
#设定数据的长度
start_date = datetime(2012,1,1)
end_date = datetime(2022,3,1)
rainfall = rainfall[start_date:end_date]
#绘制降水量的折线图
def plot_series(time_series):
    plt.figure(figsize=(20,8))
    plt.plot(time_series)
    plt.title('降水量折线图', fontsize=20)
    plt.ylabel('降水量（mm）', fontsize=16)
    #mplcursors.cursor(hover=True)
    #Gera uma linha indicando novo ano, fortalecendo a suspeita de sasonalidade da população
    for year in range(start_date.year,2023):
        plt.axvline(pd.to_datetime('31-1-' + str(year)), color='k', linestyle='--', alpha=0.2)
#绘制相关性图
def plot_corr(time_series):
    fig, axes = plt.subplots(1,2,figsize=(16,5), dpi= 300)
    plot_acf(time_series, lags=24, ax=axes[0])
    plot_pacf(time_series, lags=24, ax=axes[1])
    plt.show()
#给出arima的使用条件 其中ADF < 10%,20%,50% p-valor<1
def stationarity_check(time_series):
    result = adfuller(time_series)
    print(f'Estatísticas ADF : {result[0]}')
    print(f'p-valor: {result[1]}')
    print("Valores Críticos:\n")
    for key, value in result[4].items():
        print(f'{key}: {value}')

#画图
plot_series(rainfall)
plt.show()
print(stationarity_check(rainfall))


plot_corr(rainfall)
plt.show()
train, test = rainfall.iloc[:-14], rainfall.iloc[-15:]
x_train, x_test = np.array(range(train.shape[0])), np.array(range(train.shape[0], rainfall.shape[0]))
fig, axes = plt.subplots(1,1,figsize=(16,5), dpi= 300)
plt.plot(train)
plt.plot(test)
for year in range(start_date.year,2023):
    plt.axvline(pd.to_datetime('31-1-' + str(year)), color='k', linestyle='--', alpha=0.2)
plt.show()

#加载armia模型
from pmdarima.arima import auto_arima
model = auto_arima(train,start_p=1, start_q=1,
                      test='adf',
                      max_p=5, max_q=5,
                      m=12,
                      d=0,
                      seasonal='TRUE',
                      start_P=1,
                      start_Q=1,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
#输出模型的相关参数
print(model.summary())
#预测后15个月的结果
prediction, confint = model.predict(n_periods=15, return_conf_int=True)
#获取真值后面比较
dates_pred = pd.read_csv("rain_21_22.csv",parse_dates=[0], index_col=0, date_parser=parser)
dates_pred = dates_pred.iloc[-15:].reset_index()
#绘图
for i in range(0,15):
    dates_pred.at[i,'降水量(mm)']= prediction[i]
print("Valores Preditos\n")
print(dates_pred)
print("Valores Reais\n")
print(test)
cf= pd.DataFrame(confint)
prediction_series = pd.Series(prediction,index=test.index)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(rainfall)
ax.plot(prediction_series)
plt.title('预测趋势图', fontsize=20)
plt.ylabel('降水量（mm）', fontsize=16)
ax.fill_between(prediction_series.index,
                cf[0],
                cf[1],color='grey',alpha=.3)
for year in range(start_date.year,2023):
    plt.axvline(pd.to_datetime('31-1-' + str(year)), color='k', linestyle='--', alpha=0.2)
plt.show()

prediction_series = pd.Series(prediction,index=test.index)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(test)
ax.plot(prediction_series)
plt.title('局部放大图', fontsize=20)
plt.ylabel('降水量（mm）', fontsize=16)
ax.fill_between(prediction_series.index,
                cf[0],
                cf[1],color='grey',alpha=.3)
plt.show()

from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

real_value = np.array(test['降水量(mm)'])
pred_value = np.array(dates_pred['降水量(mm)'])
t_value = np.array(train['降水量(mm)'])
smape = MeanAbsolutePercentageError(symmetric=True)
print(smape(real_value, pred_value))


