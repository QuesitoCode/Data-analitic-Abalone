import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

data = 'DataSet/color_iris.csv'

dataset = pd.read_csv(data)

column = [
    'sepal length',
    'sepal width',
    'petal length',
    'petal width',
    'name'
]

dataset.columns=column

corr = dataset.corr()


fig = plot.figure(figsize=(8,8))
plot.matshow(corr,cmap = 'RdBu' ,fignum=fig.number)
plot.xticks(range(len(corr.columns)),corr.columns,rotation='vertical');
plot.yticks(range(len(corr.columns)),corr.columns);
