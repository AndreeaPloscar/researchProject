import math
from datetime import timedelta
import pandas as pd
from keras import models
from keras.layers import Dense, Dropout
import numpy as np
import livelossplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib

from cnn.load_data import load_data

matplotlib.use('TkAgg')
plot_losses = livelossplot.PlotLossesKeras()

NUM_ROWS = 1
NUM_COLS = 12
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 10

train_meteo = './train/meteo.xlsx'
train_med = './train/med.txt'
test_meteo = './test/meteo.xlsx'
test_med = './test/med.txt'

batch_size = 32

X_train, y_train = load_data(train_meteo, train_med)
X_test, y_test = load_data(test_meteo, test_med)

model = models.Sequential()
model.add(Dense(512, activation='relu', input_shape=(NUM_ROWS * NUM_COLS,)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
# Train model
model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=[plot_losses],
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])