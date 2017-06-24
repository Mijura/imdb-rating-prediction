import pandas as pd
from pandas import DataFrame,Series
import numpy as np
from sklearn import linear_model,neighbors , svm ,tree

def ridge_regression(x_train, y_train, x_test):
	model = linear_model.Ridge()
	model.fit(x_train, y_train)

	return model.predict(x_test)
	
def simple_linear_regression(x_train, y_train, x_test):
	model = linear_model.LinearRegression()
	model.fit(x_train, y_train)
	
	return model.predict(x_test)

def knn_regression(x_train,y_train,x_test):
	model=neighbors.KNeighborsRegressor(5,weights='uniform')
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
	model = tree.DecisionTreeRegressor(max_depth=3)
	model.fit(x_train,y_train)
	
	return model.predict(x_test)

def calc_error(y_test, y_predict):

	err = 0
	s = 0

	for y, yp in zip(y_test, y_predict) :
		print(y, yp)
		s += abs(y - yp)
		err += (y - yp) ** 2

	print(err, len(x_test))
	print(err / len(x_test))
	print(s / len(x_test))

if __name__ == "__main__":
	data = DataFrame(pd.read_csv('movie_metadata.csv'))
	data = data[data.dtypes[data.dtypes!='object'].index]
	data = data.fillna(0)
	df_norm = (data - data.mean()) / (data.max() - data.min()) + 1


	x_train = data[data.dtypes[data.dtypes != 'object'].index]
	y_train = data['imdb_score']
	x_train.drop(['imdb_score'],axis = 1, inplace = True)
	x_train = x_train.fillna(0)

	x_test = data[4000:]
	y_test = x_test['imdb_score']
	x_test = x_test.drop(['imdb_score'], axis = 1)

	x_train = x_train[:4000]
	y_train = y_train[:4000]
	
	choice = -1
	while True:
		choice = int(input("\nChoose an algorithm: \n1.Simple Linear Regression\n2.KNN Regression\n"
													+"3.Bayesian Regression.\n4.SVM\n5.Ridge Regression\n"
													+"6.Decision Tree\n"
													+"0. exit\n"))

		if choice == 1:
			y_predict = simple_linear_regression(x_train, y_train, x_test)
			calc_error(y_test, y_predict)
		elif choice == 2:
			y_predict = knn_regression(x_train,y_train,x_test)
			calc_error(y_test,y_predict)
		elif choice == 3:
			y_predict= bayesian_regression(x_train,y_train,x_test)
			calc_error(y_test,y_predict)
		elif choice == 4:
			y_predict=suppor_vector_machine(x_train,y_train,x_test)
			calc_error(y_test,y_predict)
		elif choice == 5:
			y_predict = ridge_regression(x_train, y_train, x_test)
			calc_error(y_test, y_predict)
		elif choice == 6:
			y_predict = decision_tree(x_train, y_train, x_test)
			calc_error(y_test, y_predict)
		elif choice == 0:
			break