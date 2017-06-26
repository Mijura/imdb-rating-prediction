import pandas as pd
from pandas import DataFrame,Series
import numpy as np
from sklearn import linear_model,neighbors , svm ,tree,ensemble
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
def ridge_regression(x_train, y_train, x_test):
	model = linear_model.Ridge()
	model.fit(x_train, y_train)

	return model.predict(x_test)
	
def simple_linear_regression(x_train, y_train, x_test):
	model = linear_model.LinearRegression()
	model.fit(x_train, y_train)
	
	return model.predict(x_test)

def knn_regression(x_train,y_train,x_test):
	model=neighbors.KNeighborsRegressor(10,weights='uniform')
	model.fit(x_train,y_train)

	return model.predict(x_test)

def lasso(x_train,y_train,x_test):
	model = linear_model.Lasso(alpha=0.1)
	model.fit(x_train,y_train)

	return model.predict(x_test)

def bayesian_regression(x_train,y_train,x_test):
	model=linear_model.BayesianRidge()
	model.fit(x_train,y_train)
	
	return model.predict(x_test)

def suppor_vector_machine(x_train,y_train,x_test):
	model = svm.SVR()
	model.fit(x_train,y_train)
	
	return model.predict(x_test)
	
def decision_tree(x_train,y_train,x_test):
	model = tree.DecisionTreeRegressor(max_depth=7)
	model.fit(x_train,y_train)
	
	return model.predict(x_test)

def random_forest(x_train,y_train,x_test):
	model = ensemble.RandomForestRegressor(n_estimators=10)
	model.fit(x_train,y_train)
	
	return model.predict(x_test)

def calc_error(y_test, y_predict):

	err = 0
	s = 0

	for y, yp in zip(y_test, y_predict) :
		s += abs(y - yp)
		err += (y - yp) ** 2

	print("MSE:")
	print(err / len(x_test))
	print("Average error:")
	print(s / len(x_test))

def residual_plot(y_test,y_predict):
	preds = pd.DataFrame({"preds":y_predict, "true":y_test})
	preds["residuals"] = preds["true"] - preds["preds"]
	preds.plot(x = "preds", y = "residuals",kind = "scatter")
	plt.title("Residual plot")
	plt.show()


if __name__ == "__main__":
	f = pd.read_csv("movie_metadata.csv")

	data=DataFrame(f)

	cols=data.dtypes[data.dtypes!='object'].index
	#cols=['duration','num_voted_users','imdb_score']
	x=data[cols]

	x=x.fillna(0)	


	y=x['imdb_score']
	x.drop(['imdb_score'],axis=1,inplace=True)

	x=x.values
	y=np.asarray(y)

	x=np.asarray(x)

	number_of_samples = len(y)
	np.random.seed(15)
	random_indices = np.random.permutation(number_of_samples)
	num_training_samples = int(number_of_samples*0.75)
	x_train = x[random_indices[:num_training_samples]]
	y_train=y[random_indices[:num_training_samples]]
	x_test=x[random_indices[num_training_samples:]]
	y_test=y[random_indices[num_training_samples:]]
	
	choice = -1
	while True:
		choice = int(input("\nChoose an algorithm: \n1.Simple Linear Regression\n2.KNN Regression\n"
													+"3.Bayesian Regression.\n4.SVR\n5.Ridge Regression\n"
													+"6.Decision Tree\n7.Lasso\n8.Random Forest\n"
													+"0. exit\n"))

		if choice == 1:
			y_predict = simple_linear_regression(x_train, y_train, x_test)
			calc_error(y_test, y_predict)
			residual_plot(y_test, y_predict)
		elif choice == 2:
			y_predict = knn_regression(x_train,y_train,x_test)
			calc_error(y_test,y_predict)
			residual_plot(y_test,y_predict)
		elif choice == 3:
			y_predict= bayesian_regression(x_train,y_train,x_test)
			calc_error(y_test,y_predict)
			residual_plot(y_test,y_predict)
		elif choice == 4:
			y_predict=suppor_vector_machine(x_train,y_train,x_test)
			calc_error(y_test,y_predict)
			residual_plot(y_test,y_predict)
		elif choice == 5:
			y_predict = ridge_regression(x_train, y_train, x_test)
			calc_error(y_test, y_predict)
			residual_plot(y_test,y_predict)
		elif choice == 6:
			y_predict = decision_tree(x_train, y_train, x_test)
			calc_error(y_test, y_predict)
			residual_plot(y_test,y_predict)
		elif choice == 7 :
			y_predict = lasso(x_train, y_train, x_test)
			calc_error(y_test, y_predict)
			residual_plot(y_test,y_predict)
		elif choice == 8 :
			y_predict = random_forest(x_train, y_train, x_test)
			calc_error(y_test, y_predict)
			residual_plot(y_test,y_predict)
		elif choice == 0:
			break