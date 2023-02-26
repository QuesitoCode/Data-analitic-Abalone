import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error

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
plot.subplots()