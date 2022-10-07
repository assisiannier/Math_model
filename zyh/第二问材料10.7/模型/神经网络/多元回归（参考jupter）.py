from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd

def polynomial_model(degree=1):
	polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
	linear_regression = LinearRegression(normalize=True)
	# 这是一个流水线，先增加多项式阶数，然后再用线性回归算法来拟合数据
	pipeline = Pipeline([("polynomial_features", polynomial_features),
						 ("linear_regression", linear_regression)])
	return pipeline
data = pd.read_csv("clean_data1.csv", encoding="utf_8_sig")[0:123]
data1 = pd.read_csv("clean_data1.csv", encoding="utf_8_sig")[123:]
X=data.iloc[:,data.keys()!='100cm湿度(kg/m2)']
Y = data["100cm湿度(kg/m2)"]
Res_x = data1.iloc[:,data.keys()!='100cm湿度(kg/m2)']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
poly_model = polynomial_model(degree=2)
poly_model.fit(X_train, Y_train)
Y_pred = poly_model.predict(X_test)
model_poly = pd.DataFrame(X_test)
model_poly['MEDV'] = Y_test
model_poly['Predicted MEDV'] = Y_pred
# Get Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, Y_pred)
# Get Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_test, Y_pred)
print(f"mse:{mse},mae:{mae}")
Y_pred = poly_model.predict(Res_x)
print(Y_pred)