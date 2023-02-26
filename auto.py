import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

auto = 'DataSet/auto-mpgv2.csv'

datasetAuto = pd.read_csv(auto)

column = [
    'mpg',
    'cilidraje',
    'desplazamiento',
    'caballos',
    'peso',
    'aceleración',
    'año',
    'origen',
    'nombre'
]

datasetAuto.columns=column


datasetAuto.loc[datasetAuto['caballos']=='?']
datasetAuto.drop(datasetAuto.loc[datasetAuto.caballos == '?'].index, inplace = True)
datasetAuto['caballos'] = datasetAuto['caballos'].astype('float64')


datasetAuto[column].plot(kind='box', subplots = True, layout = (3,5), figsize=(14,8))


new_df = datasetAuto
caracteristicas = ['caballos','aceleración']
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


plot.boxplot(x=new_df['caballos'])
plot.subplots()
plot.boxplot(x=new_df['aceleración'])

df_caracteristicas = new_df[[column[0],column[4]]]
df_resultado = new_df[column[2]]

x_train, x_test,y_train,y_test = train_test_split(df_caracteristicas, df_resultado, train_size=0.6)
modelo = LinearRegression()

modelo.fit(X=np.array(x_train), y= y_train)

predicciones = modelo.predict(X= np.array(x_test))

r2 = r2_score(y_true= y_test, y_pred= predicciones)

corr = new_df.corr()

fig = plot.figure(figsize=(8,8))
plot.matshow(corr,cmap = 'RdBu' ,fignum=fig.number)
plot.xticks(range(len(corr.columns)),corr.columns,rotation='vertical');
plot.yticks(range(len(corr.columns)),corr.columns);

print(r2)


new_df[column].plot(kind='box', subplots = True, layout = (3,5), figsize=(14,8))