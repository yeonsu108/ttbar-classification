import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from myutils import *

trainInput = "../SNN_sample/ttbar.h5"
data = pd.read_hdf(trainInput)

epochs = 3
trainOutput = "keras_ttbar_epoch"+str(epochs)
trainOutput = "test"
try: os.mkdir(trainOutput)
except: pass

# make the number of events for each category equal
pd_tth = data[data['category'] == 0].sample(n=40330)
#pd_tth = data[data['category'] == 0].sample(n=0)
pd_ttlf = data[data['category'] == 1].sample(n=40330)
pd_ttb = data[data['category'] == 2].sample(n=40330)
pd_ttbb = data[data['category'] == 3].sample(n=40330)
pd_ttc = data[data['category'] == 4].sample(n=40330)
pd_ttcc = data[data['category'] == 5].sample(n=40330)

# merge data and reset index
pd_data = pd.concat([pd_tth, pd_ttlf, pd_ttb, pd_ttbb, pd_ttc, pd_ttcc])
pd_data = pd_data.sample(frac=1).reset_index(drop=True)

# pickup only interesting variables
variables = ["ngoodjets","nbjets_m", "nbjets_t", "ncjets_l",
    "ncjets_m", "ncjets_t", "deltaR_j12", "deltaR_j34",
    "sortedjet_pt1", "sortedjet_pt2", "sortedjet_pt3", "sortedjet_pt4",
    "sortedjet_eta1", "sortedjet_eta2", "sortedjet_eta3", "sortedjet_eta4",
    "sortedjet_phi1", "sortedjet_phi2", "sortedjet_phi3", "sortedjet_phi4",
    "sortedjet_mass1", "sortedjet_mass2", "sortedjet_mass3", "sortedjet_mass4",
    "sortedjet_btag1", "sortedjet_btag2", "sortedjet_btag3", "sortedjet_btag4"]
class_names = ["tth", "ttlf", "ttb", "ttbb", "ttc", "ttcc"]
#class_names = ["ttlf", "ttb", "ttbb", "ttc", "ttcc"]

pd_train_out  = pd_data.filter(items = ['category'])
pd_train_data = pd_data.filter(items = variables)

#covert from pandas to array
train_out = np.array( pd_train_out )
train_data = np.array( pd_train_data )

numbertr = len(train_out)
trainlen = int(0.8*numbertr) # Fraction used for training

# Splitting between training set and cross-validation set
valid_data=train_data[trainlen:,0::]
valid_data_out=train_out[trainlen:]

train_data=train_data[:trainlen,0::]
train_data_out=train_out[:trainlen]

# model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
])

batch_size = 512

model.compile(loss='sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy', 'sparse_categorical_accuracy'])
hist = model.fit(train_data, train_data_out, batch_size=batch_size, epochs=epochs, validation_data=(valid_data,valid_data_out))

model.summary()

pred = np.argmax(model.predict(valid_data), axis=1)
comp = np.reshape(valid_data_out,(-1))

acc = 100 * np.mean( pred == comp )

plot_confusion_matrix(comp, pred, classes=class_names,
                    title='Confusion matrix, without normalization, acc=%.2f'%acc, savename=trainOutput+"/confusion_matrix.pdf")

plot_confusion_matrix(comp, pred, classes=class_names, normalize=True,
                    title='Normalized confusion matrix, acc=%.2f'%acc, savename=trainOutput+"/norm_confusion_matrix.pdf")

plot_performance(hist=hist, savedir=trainOutput)

