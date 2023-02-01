import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

abalone = 'DataSet/abalone.csv'

datasetAbalone = pd.read_csv(abalone)

column = [
    'Sex',
    'Length',
    'Diameter',
    'Height',
    'Whole Weight',
    'Shucked weigth',
    'Viscera weight',
    'Shell weight',
    'Rings'
]

datasetAbalone.columns=column

#* Graficas

plot.hist(x=datasetAbalone['Diameter'])
plot.subplots()
plot.hist(x=datasetAbalone['Height'])
plot.subplots()
plot.boxplot(x=datasetAbalone['Viscera weight'])
plot.subplots()
plot.scatter(x=datasetAbalone['Rings'],y=datasetAbalone['Height'])
#plot.hist(x=datasetAbalone['Sex'])


#new_df = datasetAbalone
#caracteristicas = ['c3','c4','c5','c6']
#for i in caracteristicas:
#    x = i
#
#    # TODO calculo de cuartil 1 y 3
#    Q1 = new_df[x].quantile(0.25)
#    Q3 = new_df[x].quantile(0.75)
#
#    IQR = Q3 - Q1
#
#    u_limit = Q3 + 1.5 * IQR
#    l_limit = Q3 + 1.5 * IQR
#    ubicacionNoAtipicos = (new_df[x] >= l_limit) & (new_df[x] <= u_limit)
#    new_df = new_df[ubicacionNoAtipicos]

#new_df


#*

x = datasetAbalone['Diameter']
y = datasetAbalone['Rings']

#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.5)


#modelo = LinearRegression()
#modelo.fit(x = np.array(x_train).reshape(-1, 1), y = y_train
          

          

          
X = datasetAbalone[['Rings','Diameter']]
Y = datasetAbalone['Height']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.5)

modelo2 = LinearRegression()
modelo2.fit(X = (x_train), y = y_train)
predicciones = modelo2
