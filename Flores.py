import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import math

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

#------------------------------------------------------------------------------------------------------------------>

#cambiando lso nombres de las flores por numeros para poder identificar y tener un tratamiendo de datos mas sencillo

#------------------------------------------------------------------------------------------------------------------>

dataset['name'] = dataset['name'].replace('Iris-setosa', 0)
dataset['name'] = dataset['name'].replace('Iris-versicolor', 1)
dataset['name'] = dataset['name'].replace('Iris-virginica', 2)


fig = plot.figure(figsize=(8,8))
plot.matshow(corr,cmap = 'RdBu' ,fignum=fig.number)
plot.xticks(range(len(corr.columns)),corr.columns,rotation='vertical');
plot.yticks(range(len(corr.columns)),corr.columns);

#------------------------------------------------------------------------------------------------------------------>

#Separación y mezcla de la data según la flor

#------------------------------------------------------------------------------------------------------------------>

data_setosa = dataset[:49]
data_versicolor = dataset[49:99]
data_virginica = dataset[99:]

data_setosa = data_setosa.to_numpy()
np.random.shuffle(data_setosa)
data_setosa = pd.DataFrame(data_setosa)
data_setosa.columns = column

data_versicolor = data_versicolor.to_numpy()
np.random.shuffle(data_versicolor)
data_versicolor = pd.DataFrame(data_versicolor)
data_versicolor.columns = column

data_virginica = data_virginica.to_numpy()
np.random.shuffle(data_virginica)
data_virginica = pd.DataFrame(data_virginica)
data_virginica.columns = column

#------------------------------------------------------------------------------------------------------------------>

#Definición de datos para evaluación y estimación

#------------------------------------------------------------------------------------------------------------------>

def datasets(porcentajes):
    cant = int(50*porcentajes)
    
    
    data_setosa_test = data_setosa[:cant]
    data_versicolor_test = data_versicolor[:cant]
    data_virginica_test = data_virginica[:cant]
    
    data_test = pd.DataFrame()
    data_test = pd.concat([data_setosa_test,data_versicolor_test,data_virginica_test])
    
    data_setosa_train = data_setosa[:cant]
    data_versicolor_train = data_versicolor[:cant]
    data_virginica_train = data_virginica[:cant]
    
    data_train = pd.DataFrame()
    data_train = pd.concat([data_setosa_train,data_versicolor_train,data_virginica_train])
    
    data_train = data_train.to_numpy()
    np.random.shuffle(data_train)
    data_train = pd.DataFrame(data_train)
    data_train.columns = column
    
    data_test = data_test.to_numpy()
    np.random.shuffle(data_test)
    data_test = pd.DataFrame(data_test)
    data_test.columns = column
    return data_test,data_train

def euclidiana (test,train):
    distancia = 0
    for i in range(len(test)):
        distancia += (test[i]-train[i])**2
    return math.sqrt(distancia)

def prediccion (x_train, y_train, x_test, k):
    y_pred = []
    for i in range(len(x_test)):
        distancias = []
        for j in range(len(x_train)):
            distancia = euclidiana(x_test[i], x_train[j])
            distancias.append((distancia, y_train[j]))
        distancias.sort()
        vecinos = distancias[:k]
        #print(vecinos)
        #print(i)
        cont = {}
        for vecino in vecinos:
            if vecino[1] in cont:
                cont[vecino[1]] += 1
            else:
                cont[vecino[1]] = 1
        
        y_pred.append(max(cont, key = cont.get))
    return y_pred

#------------------------------------------------------------------------------------------------------------------>

#Modelo 80-20

#Se generan los dataset de prueba y entrenamiento 

#------------------------------------------------------------------------------------------------------------------>

data_test, data_train = datasets(0.2)

x_train = data_train.iloc[:,[2,3]].values
y_train = data_train.iloc[:,4].values

x_test = data_test.iloc[:,[2,3]].values
y_test = data_test.iloc[:,4].values

#------------------------------------------------------------------------------------------------------------------>

#se ejecuta el KNN y se reportan la cantidad de aciertos y fallos

#------------------------------------------------------------------------------------------------------------------>

k=3
y_pred = prediccion(x_train, y_train, x_test, k)
cont = 0

for i in range(len(y_test)):
    if(y_test[i]!=y_pred[i]):
        cont +=1
    
print('fallos: ', cont)
print('aciertos: ',(len(y_test)-cont) )


k=5
y_pred = prediccion(x_train, y_train, x_test, k)
cont = 0

for i in range(len(y_test)):
    if(y_test[i]!=y_pred[i]):
        cont +=1
    
print('fallos: ', cont)
print('aciertos: ',(len(y_test)-cont) )

k=7
y_pred = prediccion(x_train, y_train, x_test, k)
cont = 0

for i in range(len(y_test)):
    if(y_test[i]!=y_pred[i]):
        cont +=1
    
print('fallos: ', cont)
print('aciertos: ',(len(y_test)-cont) )

k=9
y_pred = prediccion(x_train, y_train, x_test, k)
cont = 0

for i in range(len(y_test)):
    if(y_test[i]!=y_pred[i]):
        cont +=1
    
print('fallos: ', cont)
print('aciertos: ',(len(y_test)-cont) )
                

k=11
y_pred = prediccion(x_train, y_train, x_test, k)
cont = 0

for i in range(len(y_test)):
    if(y_test[i]!=y_pred[i]):
        cont +=1
    
print('fallos: ', cont)
print('aciertos: ',(len(y_test)-cont) )


k=13
y_pred = prediccion(x_train, y_train, x_test, k)
cont = 0

for i in range(len(y_test)):
    if(y_test[i]!=y_pred[i]):
        cont +=1
    
print('fallos: ', cont)
print('aciertos: ',(len(y_test)-cont) )


print()
#------------------------------------------------------------------------------------------------------------------>

#MODELO 50-50

#------------------------------------------------------------------------------------------------------------------>

data_test_50, data_train_50 = datasets(0.5)

x_train_50 = data_train_50.iloc[:,[2,3]].values
y_train_50 = data_train_50.iloc[:,4].values

x_test_50 = data_test_50.iloc[:,[2,3]].values
y_test_50 = data_test_50.iloc[:,4].values

k=7
y_pred_50 = prediccion(x_train_50, y_train_50, x_test_50, k)
cont = 0

for i in range(len(y_test_50)):
    if(y_test_50[i]!=y_pred_50[i]):
        cont +=1
    
print('fallos: ', cont)
print('aciertos: ',(len(y_test_50)-cont) )

#------------------------------------------------------------------------------------------------------------------>

#MODELO 75-25

#------------------------------------------------------------------------------------------------------------------>

data_test_75, data_train_75 = datasets(0.75)

x_train_75 = data_train_75.iloc[:,[2,3]].values
y_train_75 = data_train_75.iloc[:,4].values

x_test_75 = data_test_75.iloc[:,[2,3]].values
y_test_75 = data_test_75.iloc[:,4].values

k=7
y_pred_75 = prediccion(x_train_75, y_train_75, x_test_75, k)
cont = 0

for i in range(len(y_test_75)):
    if(y_test_75[i]!=y_pred_75[i]):
        cont +=1
    
print('fallos: ', cont)
print('aciertos: ',(len(y_test_75)-cont) )
            
    

    


