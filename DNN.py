from __future__ import division
import sys, os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
from ROOT import TFile, TTree
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, add
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from plot_confusion_matrix import plot_confusion_matrix

trainInput = "../SNN_sample/ttbar.h5"
trainOutput = "keras_ttbar_epoch_30"
try: os.mkdir(trainOutput)
except: pass
data = pd.read_hdf(trainInput)

# make the number of events for each category equal
pd_tth = data[data['category'] == 0].sample(n=40330)
#pd_tth = data[data['category'] == 0].sample(n=0)
pd_ttlf = data[data['category'] == 1].sample(n=40330)
pd_ttb = data[data['category'] == 2].sample(n=40330)
pd_ttbb = data[data['category'] == 3].sample(n=40330)
pd_ttc = data[data['category'] == 4].sample(n=40330)
pd_ttcc = data[data['category'] == 5].sample(n=40330)

# merge data and reset index
pd_data = pd.concat([pd_tth, pd_ttlf, pd_ttb, pd_ttbb, pd_ttc, pd_ttcc], ignore_index=True)
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

numbertr=len(train_out)

print numbertr

trainnb=0.8 # Fraction used for training

# Splitting between training set and cross-validation set
valid_data=train_data[int(trainnb*numbertr):numbertr,0::]
valid_data_out=train_out[int(trainnb*numbertr):numbertr]
valid_data_out = to_categorical(valid_data_out)

train_data_out=train_out[0:int(trainnb*numbertr)]
train_data=train_data[0:int(trainnb*numbertr),0::]
train_data_out = to_categorical(train_data_out)

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
 # tf.keras.layers.Dense(100, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(100, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(100, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])

#modelshape = "10L_300N"
batch_size = 512
epochs = 30
model_output_name = 'model_ttbar_%dE' %(epochs)

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy', 'categorical_accuracy'])
hist = model.fit(train_data, train_data_out, batch_size=batch_size, epochs=epochs, validation_data=(valid_data,valid_data_out))

    #using only fraction of data
    #evaluate = model.predict( valid_data ) 

model.summary()

pred = model.predict(valid_data)
pred = np.argmax(pred, axis=1)
#pred = to_categorical(pred)
comp = np.argmax(valid_data_out, axis=1)

acc = 100 * np.mean( pred == comp )
print (str(acc))

plot_confusion_matrix(comp, pred, classes=class_names,
                    title='Confusion matrix, without normalization, acc=%.2f'%acc)
plt.savefig(trainOutput+"/confusion_matrix.pdf")
plt.gcf().clear()

plot_confusion_matrix(comp, pred, classes=class_names, normalize=True,
                    title='Normalized confusion matrix, acc=%.2f'%acc)
plt.savefig(trainOutput+"/norm_confusion_matrix.pdf")
plt.gcf().clear()

print("Plotting scores")
plt.plot(hist.history['categorical_accuracy'])
plt.plot(hist.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Test'], loc='lower right')
plt.savefig(os.path.join(trainOutput+'/fig_score_acc.pdf'))
plt.gcf().clear()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper right')
plt.savefig(os.path.join(trainOutput+'/fig_score_loss.pdf'))
plt.gcf().clear()

