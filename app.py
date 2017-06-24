import pandas as pd
from pandas import DataFrame,Series
import numpy as np
from sklearn import linear_model

data = DataFrame(pd.read_csv('movie_metadata.csv'))
data = data[data.dtypes[data.dtypes!='object'].index]
data = data.fillna(0)
df_norm = (data - data.mean()) / (data.max() - data.min()) + 1


x_train = data[data.dtypes[data.dtypes != 'object'].index]
y_train = data['imdb_score']
x_train.drop(['imdb_score'],axis = 1, inplace = True)
x_train = x_train.fillna(0)

x_test = data[2000:]
y_test = x_test['imdb_score']
x_test.drop(['imdb_score'], axis = 1, inplace = True)

x_train = x_train[:2000]
y_train = y_train[:2000]

model = linear_model.Ridge()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

err = 0
s = 0

for y, yp in zip(y_test, y_predict) :
	print(y, yp)
	s += abs(y - yp)
	err += (y - yp) ** 2

print(err, len(x_test))
print(err / len(x_test))
print(s / len(x_test))