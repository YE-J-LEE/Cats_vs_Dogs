import random
from google.colab import drive
drive.mount("/content/gdrive/")

# /content/gdrive/My Drive/
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from keras.layers import Dense, Activation
from keras.utils import np_utils
from imutils import build_montages
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import cv2

trainCats = []
for i in os.listdir("/content/gdrive/My Drive/training_set/cats/"):
    if '(' not in i and '_' not in i:
        i = "/content/gdrive/My Drive/training_set/cats/" + i
        trainCats.append(i)
trainDogs = []
for i in os.listdir("/content/gdrive/My Drive/training_set/dogs/"):
    if '_' not in i and '(' not in i:
        i = "/content/gdrive/My Drive/training_set/dogs/" + i
        trainDogs.append(i)
testCats = []
for i in os.listdir("/content/gdrive/My Drive/test_set/cats/"):
    if '(' not in i and '_' not in i:
        i = "/content/gdrive/My Drive/test_set/cats/" + i
        testCats.append(i)
testDogs = []
for i in os.listdir("/content/gdrive/My Drive/test_set/dogs/"):
    if '(' not in i and '_' not in i:
        i = "/content/gdrive/My Drive/test_set/dogs/" + i
        testDogs.append(i)

Cats, Dogs, All = [], [], []
for i, j in zip(trainCats, trainDogs):
#     Cats.append(i)
#     Dogs.append(j)
    All.append(i)
    All.append(j)
for i, j in zip(testCats, testDogs):
#     Cats.append(i)
#     Dogs.append(j)
    All.append(i)
    All.append(j)

# print("Cats", len(Cats))
# print("Dogs", len(Dogs))
print('All', len(All))
imagePaths = All

data = []
labels = []

for imagePath in imagePaths:
    # Extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the image, and resize it to be a fixed 100x100 pixels, ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (100, 100))

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# # Converting data into Numpy Array, scaling it to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
# # Reshaping for channel dimension
data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 3))

# Encoding Labels
LE = LabelEncoder()
labels = LE.fit_transform(labels)
# One-hot Encoding
labels = np_utils.to_categorical(labels, 2)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, stratify=labels, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

import keras.backend.tensorflow_backend as K
from keras.callbacks import EarlyStopping
with K.tf_ops.device('/device:GPU:0'): 
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(256, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dropout(0.5)) # DROPOUT
  model.add(Dense(512, activation='relu'))
  model.add(Dense(2, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  
  early_stopping = EarlyStopping(patience = 7) # early stop callback
model.summary()

hist = model.fit(X_train, y_train, epochs=3000, batch_size=100, validation_data=(X_valid, y_valid), callbacks=[early_stopping])

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

loss_and_metrics = model.evaluate(X_test, y_test, batch_size=100)

print('')
print('loss : ' + str(loss_and_metrics[0]))
print('accuray : ' + str(loss_and_metrics[1]))
