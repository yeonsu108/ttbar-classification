# use: python train.py inputdir

import os
import sys

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import uproot
import pandas as pd
import numpy as np
import tensorflow as tf
from myutils import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.setrecursionlimit(10**7)

indir = sys.argv[1]
train_outdir = indir+"/results_conv/"
os.makedirs(train_outdir, exist_ok=True)
class_names = ["ttbb","tthbb"]

print("Start multi Training")
epochs = 1000
inputvars = [
    "njet", "nbjet", "nmuon", "nelectron", "MET_met", "MET_eta", "MET_phi",
    "lep_pt", "lep_eta", "lep_phi", "lep_e",
    "Jet1_pt", "Jet1_eta", "Jet1_phi", "Jet1_mass", "Jet1_btag",
    "Jet2_pt", "Jet2_eta", "Jet2_phi", "Jet2_mass", "Jet2_btag",
    "Jet3_pt", "Jet3_eta", "Jet3_phi", "Jet3_mass", "Jet3_btag",
    "Jet4_pt", "Jet4_eta", "Jet4_phi", "Jet4_mass", "Jet4_btag",
    "Jet5_pt", "Jet5_eta", "Jet5_phi", "Jet5_mass", "Jet5_btag",
    "Jet6_pt", "Jet6_eta", "Jet6_phi", "Jet6_mass", "Jet6_btag",
    "nonbJet1_pt", "nonbJet1_eta", "nonbJet1_phi", "nonbJet1_mass",
    "nonbJet2_pt", "nonbJet2_eta", "nonbJet2_phi", "nonbJet2_mass",
    "bJet1_pt", "bJet1_eta", "bJet1_phi", "bJet1_mass",
    "bJet2_pt", "bJet2_eta", "bJet2_phi", "bJet2_mass",
    "bJet3_pt", "bJet3_eta", "bJet3_phi", "bJet3_mass",
    "bJet4_pt", "bJet4_eta", "bJet4_phi", "bJet4_mass",
    "seljet1_idx", "seljet2_idx", "selbjet1_idx", "selbjet2_idx", "selnonbjet1_idx", "selnonbjet2_idx",
    "mindR_jjPt", "mindR_jjEta", "mindR_jjPhi", "mindR_jjMass", "mindR_jjdR",
    "mindR_bbPt", "mindR_bbEta", "mindR_bbPhi", "mindR_bbMass", "mindR_bbdR",
    "mindR_nnPt", "mindR_nnEta", "mindR_nnPhi", "mindR_nnMass", "mindR_nndR",
    "chi2jet1_idx", "chi2jet2_idx", "chi2bjet1_idx", "chi2bjet2_idx",
    "chi2_jjPt", "chi2_jjEta", "chi2_jjPhi", "chi2_jjMass", "chi2_jjdR",
    "chi2_bbPt", "chi2_bbEta", "chi2_bbPhi", "chi2_bbMass", "chi2_bbdR",
]


# MODIFY!! Input
#df_tthbb = uproot.open(indir+"dnn_tthbb_mu.root")["dnn_input"].arrays(inputvars,library="pd")
#df_ttbb  = uproot.open(indir+"dnn_ttbb_mu.root")["dnn_input"].arrays(inputvars,library="pd")
df_tthbb = uproot.open(indir+"dnn_tthbb.root:dnn_input").arrays(inputvars,library="pd")
df_ttbb  = uproot.open(indir+"dnn_ttbb.root:dnn_input").arrays(inputvars,library="pd")
print(type(df_tthbb))

ntthbb = len(df_tthbb)
nttbb  = len(df_ttbb)

ntrain = min(ntthbb, nttbb)
print(ntrain)

df_tthbb = df_tthbb.sample(n=ntrain).reset_index(drop=True)
df_ttbb  = df_ttbb.sample(n=ntrain).reset_index(drop=True)

df_tthbb["category"] = 0
df_ttbb["category"]  = 1

pd_data = pd.concat([df_tthbb, df_ttbb])
colnames = pd_data.columns
print(pd_data.head())
print("Col names:",colnames)

print("Plotting corr_matrix total")
plot_corrMatrix(pd_data,train_outdir,"total")
print("Plotting corr_matrix tthh")
plot_corrMatrix(df_tthbb,train_outdir,"tthbb")
print("Plotting corr_matrix ttbb")
plot_corrMatrix(df_ttbb,train_outdir,"ttbb")

pd_data = pd_data.sample(frac=1).reset_index(drop=True)

print(pd_data.head())

x_total = np.array(pd_data.filter(items = inputvars))
y_total = np.array(pd_data.filter(items = ['category']))

# Splitting between training set and cross-validation set
x_train, x_val, y_train, y_val = train_test_split(x_total, y_total, test_size=0.3)
x_train = x_train.reshape(-1, 10, 10, 1)
x_val = x_val.reshape(-1, 10, 10, 1)
print(len(x_train),len(x_val),len(y_train),len(y_val))

patience_epoch = 10
# Early Stopping with Validation Loss for Best Model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience_epoch)
mc = ModelCheckpoint(train_outdir+'/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
print("xtrain shape:",x_train.shape)



###################################################
#                      Model                      #
###################################################
activation_function='relu'
weight_initializer = 'random_normal'

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, activation=tf.nn.relu, use_bias=False, input_shape=(10, 10, 1)),
    tf.keras.layers.Conv2D(filters=3, kernel_size=2, strides=2, activation=tf.nn.relu, use_bias=False),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=2, activation=tf.nn.relu, use_bias=False),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

batch_size = 200
print("batch size :", batch_size)
model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=0.5), loss="sparse_categorical_crossentropy", metrics = ["accuracy"])

model.summary()
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                validation_data=(x_val,y_val), callbacks=[es, mc])


pred_train = model.predict_classes(x_train)
print("pred_train", pred_train)
print("orig train", y_train.T)
#train_result = pd.DataFrame(np.array([y_train.T[0], pred_train.T[1]]).T, columns=["True", "Pred"])
pred_val = model.predict_classes(x_val)
print("pred_val", pred_val)
print("orig train", y_val.T)
print("conf matrix on train set ")
print(confusion_matrix(y_train, pred_train))
print("conf matrix on val set ")
print(confusion_matrix(y_val, pred_val))

plot_confusion_matrix(y_val, pred_val, classes=class_names,
                    title='Confusion matrix, without normalization', savename=train_outdir+"/confusion_matrix_val.pdf")
plot_confusion_matrix(y_val, pred_val, classes=class_names, normalize=True,
                    title='Normalized confusion matrix', savename=train_outdir+"/norm_confusion_matrix_val.pdf")
plot_confusion_matrix(y_train, pred_train, classes=class_names,
                    title='Confusion matrix, without normalization', savename=train_outdir+"/confusion_matrix_train.pdf")
plot_confusion_matrix(y_train, pred_train, classes=class_names, normalize=True,
                    title='Normalized confusion matrix', savename=train_outdir+"/norm_confusion_matrix_train.pdf")

#pred_val = model.predict(x_val)
#pred_train = model.predict(x_train)

#print(pred_val)
#print(pred_val.T)
#print(y_val)
#print(y_val.T)
#val_result = pd.DataFrame(np.array([y_val.T, pred_val.T[1]]).T, columns=["True", "Pred"])
#train_result = pd.DataFrame(np.array([y_train.T, pred_train.T[1]]).T, columns=["True", "Pred"])
#plot_output_dist(train_result, val_result, sig="tt",savedir=train_outdir)
#val_result = pd.DataFrame(np.array([y_val.T, pred_val.T[2]]).T, columns=["True", "Pred"])
#train_result = pd.DataFrame(np.array([y_train.T, pred_train.T[2]]).T, columns=["True", "Pred"])
#plot_output_dist(train_result, val_result, sig="st",savedir=train_outdir)
