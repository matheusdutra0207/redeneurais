# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

# Bibliotecas de ajuda
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv("iris.data", header=None)
dataset = data.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

'''
Separar o tipo ‘Iris-setosa’, ‘Iris-versicolor’, ‘Iris-virginica’ em um array binario para identificar o tipo.

'''
Y = pd.get_dummies(Y)
Y = Y.values

#Agora, vamos dividir o conjunto de dados em conjunto de treino e teste. Neste caso, utilizaremos 80% do dataset em treino e 20% em teste

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# pré-processamento dos dados
scaler = preprocessing.StandardScaler().fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)

def create_model(X_train, X_test, y_train, y_test):
    model = Sequential()    
    model.add(Dense(16, input_dim=4)) #primeira camada (input)
    model.add(Activation('relu'))     #primeira camada - func. ativ.
    model.add(Dense(16))              #camada intermediária
    model.add(Activation('relu'))     #camada intermediária - func. ativ.
    model.add(Dense(3))               #última camada (output)
    model.add(Activation('softmax'))  #última camada - func. ativ.
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',     metrics=['accuracy'])
    return model

model = create_model(X_train, y_train, X_test, y_test)
name = "{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='tensorboard/{}'.format(name))
model.fit(X_train, y_train, epochs=200, verbose=2, 
                 validation_data=(X_test, y_test),callbacks=[tensorboard])
model.summary()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('\nAcuácia de Teste', test_acc)

predictions = model.predict(X_test)

class_names = ['setosa','versicolor','virginica']

for teste in predictions:
  print(f'''
% de ser {class_names[0]} = {teste[0]*100}
% de ser {class_names[1]} = {teste[1]*100}
% de ser {class_names[2]} = {teste[2]*100}
        ''' )