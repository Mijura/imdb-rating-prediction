# IMDBRatingPrediction

In this experiment we try to predict imdb movie ratings based on the 27 different movie-related attributes. At first, we analyzed the data. We analyzed individual affect of attributes on the imdb rating, and their mutual correlations so that we can reduce the number of attributes during the prediction process. Methods used in the predictions are: Linear Regression, Ridge Regression, Lasso Regression, Support Vector Regression, Bayesian Regressor, KNN Regression, Decision Tree Regresor and Random Forest. After trying several number of attributes used in the prediction process, size of training set, and other parameters of the mentioned algorithams, Random Forest algoritham turned out to be the most efficient method with the lowest error value.

Since the variable we are predicting is continual (float) there’s no need for using any of the classification algorithms. We will use several regression models for prediction such as Linear Regression, Lasso, Ridge, Bayesian, K-Nearest-Neighbours, SVM Regressor, Decision Tree and Random Forest Regressor. For regression we use all numerical data from dataset. Then we compare mean squared error these regressors produce on test set (Table 1). Also, we have compared average error which represents difference from real and predicted value of imdb score for every movie.

![result table](/Screenshots/results.png)
