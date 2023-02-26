import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

auto = 'DataSet/AirQualityUCI.csv'

datasetAuto = pd.read_csv(auto, sep=';')

columns_names = datasetAuto.columns.values


boxplots_cols = columns_names[2:]
datasetAuto[boxplots_cols].plot(kind='box', subplots = True, layout = (3,5), figsize=(14,8))



new_df = datasetAuto
caracteristicas = boxplots_cols

atipicos= pd.DataFrame()

for i in caracteristicas:
    # TODO calculo de cuartil 1 y 3
    Q1 = new_df[i].quantile(0.25)
    Q3 = new_df[i].quantile(0.75)
    IQR = Q3 - Q1
    u_limit = Q3 + 1.5 * IQR
    l_limit = Q1 - 1.5 * IQR
    
    arriba= np.where(datasetAuto[i]>u_limit)
    abajo= np.where(datasetAuto[i]<l_limit)
    
    atp = arriba + abajo
    
    atipicos[i] = atp
    
    ubicacionNoAtipicos = (new_df[i] >= l_limit) & (new_df[i] <= u_limit)
    new_df = new_df[ubicacionNoAtipicos]
    
df_caracteristicas = new_df[[columns_names[11],columns_names[6]]]
df_resultado = new_df[columns_names[5]]

x_train, x_test,y_train,y_test = train_test_split(df_caracteristicas, df_resultado, train_size=0.7)
modelo = LinearRegression()

modelo.fit(X=np.array(x_train), y= y_train)

predicciones = modelo.predict(X= np.array(x_test))

r2 = r2_score(y_true= y_test, y_pred= predicciones)

print(r2)

#La diferencia entre los desempeños de cada una de las posibilidades radica en la correlación de cada uno de estos datos, ya que realizamos el entrenamiento con unos datos similares al realizar la regresión nos genera que los datos más acertados frente a estos parametros son aquellos que generan una concordancia mas acertada.



new_df[boxplots_cols].plot(kind='box', subplots = True, layout = (3,5), figsize=(14,8))

    

