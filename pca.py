from sklearn import linear_model,neighbors , svm ,tree
import numpy as np
import pandas as pd
from pandas import DataFrame
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

f = pd.read_csv("movie_metadata.csv")

data=DataFrame(f)

cols=data.dtypes[data.dtypes!='object'].index

x=data[cols]

x=x.fillna(0)	


x.drop(['imdb_score'],axis=1,inplace=True)

x=x.values
x=np.asarray(x)


X_std=StandardScaler().fit_transform(x)
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance


plt.figure(figsize=(10, 5))
plt.bar(range(len(var_exp)), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')
plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

pca = PCA(n_components=10)
x_9d = pca.fit_transform(X_std)


