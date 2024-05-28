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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.setrecursionlimit(10**7)

indir = sys.argv[1]
train_outdir = indir+"/results_DNN/"
os.makedirs(train_outdir, exist_ok=True)

print("Start multi Training")
epochs = 1000
inputvars = [
    'njets', 'nbjets', 'ncjets', 'nElectron', 'nMuon', 'MET_met',# 'nLepton', 'MET_px', 'MET_py', 
    'Lepton_pt', 'Lepton_eta', 'Lepton_e',# 'Lepton_phi',
    'Jet1_pt', 'Jet1_eta', 'Jet1_e', 'Jet1_btag', 'Jet2_pt', 'Jet2_eta', 'Jet2_e', 'Jet2_btag',# 'Jet_phi1', 'Jet_phi2',
    'Jet3_pt', 'Jet3_eta', 'Jet3_e', 'Jet3_btag', 'Jet4_pt', 'Jet4_eta', 'Jet4_e', 'Jet4_btag',# 'Jet_phi3', 'Jet_phi4',
    #'bjet1_pt', 'bjet1_eta', 'bjet1_e', 'bjet2_pt', 'bjet2_eta', 'bjet2_e',# 'bjet1_phi', 'bjet2_phi',
    'selbjet1_pt', 'selbjet1_eta', 'selbjet1_e', 'selbjet2_pt', 'selbjet2_eta', 'selbjet2_e',# 'selbjet1_phi', 'selbjet2_phi',

    'bbdR',   'bbdEta',   'bbdPhi',   'bbPt',   'bbEta',   'bbMass',   'bbHt',   'bbMt',  # 'bbPhi',   
    'nub1dR', 'nub1dEta', 'nub1dPhi', 'nub1Pt', 'nub1Eta', 'nub1Mass', 'nub1Ht', 'nub1Mt',# 'nub1Phi', 
    'nub2dR', 'nub2dEta', 'nub2dPhi', 'nub2Pt', 'nub2Eta', 'nub2Mass', 'nub2Ht', 'nub2Mt',# 'nub2Phi', 
    'nubbdR', 'nubbdEta', 'nubbdPhi', 'nubbPt', 'nubbEta', 'nubbMass', 'nubbHt', 'nubbMt',# 'nubbPhi', 
    'lb1dR',  'lb1dEta',  'lb1dPhi',  'lb1Pt',  'lb1Eta',  'lb1Mass',  'lb1Ht',  'lb1Mt', # 'lb1Phi',  
    'lb2dR',  'lb2dEta',  'lb2dPhi',  'lb2Pt',  'lb2Eta',  'lb2Mass',  'lb2Ht',  'lb2Mt', # 'lb2Phi',  
    'lbbdR',  'lbbdEta',  'lbbdPhi',  'lbbPt',  'lbbEta',  'lbbMass',  'lbbHt',  'lbbMt', # 'lbbPhi',  
    'Wjb1dR', 'Wjb1dEta', 'Wjb1dPhi', 'Wjb1Pt', 'Wjb1Eta', 'Wjb1Mass', 'Wjb1Ht', 'Wjb1Mt',# 'Wjb1Phi', 
    'Wjb2dR', 'Wjb2dEta', 'Wjb2dPhi', 'Wjb2Pt', 'Wjb2Eta', 'Wjb2Mass', 'Wjb2Ht', 'Wjb2Mt',# 'Wjb2Phi', 
    'Wlb1dR', 'Wlb1dEta', 'Wlb1dPhi', 'Wlb1Pt', 'Wlb1Eta', 'Wlb1Mass', 'Wlb1Ht', 'Wlb1Mt',# 'Wlb1Phi', 
    'Wlb2dR', 'Wlb2dEta', 'Wlb2dPhi', 'Wlb2Pt', 'Wlb2Eta', 'Wlb2Mass', 'Wlb2Ht', 'Wlb2Mt',# 'Wlb2Phi', 
]


# MODIFY!! Input
#df_tthbb = uproot.open(indir+"dnn_tthbb_mu.root")["dnn_input"].arrays(inputvars,library="pd")
#df_ttbb  = uproot.open(indir+"dnn_ttbb_mu.root")["dnn_input"].arrays(inputvars,library="pd")
#df_tthbb = uproot.open(indir+"tthbb+.root:dnn_input").arrays(inputvars,library="pd")
#df_ttbb  = uproot.open(indir+"ttbb+.root:dnn_input").arrays(inputvars,library="pd")
df_tthbb = uproot.open(indir+"tthbb+.root:dnn_input").arrays(library="pd")
ttbb  = uproot.open(indir+"ttbb+.root:dnn_input").arrays(library="pd")
ttcc  = uproot.open(indir+"ttcc+.root:dnn_input").arrays(library="pd")
ttlf  = uproot.open(indir+"ttjj+.root:dnn_input").arrays(library="pd")
class_names = ["tthbb","ttbb", "ttbj", "ttcc", "ttlf"]
nClass, nVars = len(class_names), len(inputvars)

df_tthbb = df_tthbb[df_tthbb["category"] == 1]
df_tthbb["category"] = 0
df_ttbb  = ttbb[ttbb["category"] == 1]
df_ttbj  = ttbb[ttbb["category"] == 2]
df_ttcc  = ttcc[ttcc["category"] == 3]
df_ttlf  = ttlf[ttlf["category"] == 4]

print ("tthbb", df_tthbb)
print ("ttbb", df_ttbb)
print ("ttbj", df_ttbj)
print ("ttcc", df_ttcc)
print ("ttlf", df_ttlf)
ntrain = min(len(df_tthbb), len(df_ttbb), len(df_ttbj), len(df_ttcc), len(df_ttlf))
print(ntrain)

df_tthbb = df_tthbb.sample(n=ntrain).reset_index(drop=True)
df_ttbb  = df_ttbb.sample(n=ntrain).reset_index(drop=True)
df_ttbj  = df_ttbj.sample(n=ntrain).reset_index(drop=True)
df_ttcc  = df_ttcc.sample(n=ntrain).reset_index(drop=True)
df_ttlf  = df_ttlf.sample(n=ntrain).reset_index(drop=True)

print ("tthbb", df_tthbb)
print ("ttbb", df_ttbb)
print ("ttbj", df_ttbj)
print ("ttcc", df_ttcc)
print ("ttlf", df_ttlf)

pd_data = pd.concat([df_tthbb, df_ttbb, df_ttbj, df_ttcc, df_ttlf])
colnames = pd_data.columns
print(pd_data.head())
print("Col names:",colnames)

print("Plotting corr_matrix total")
plot_corrMatrix(pd_data,train_outdir,"total")
print("Plotting corr_matrix tthbb")
plot_corrMatrix(df_tthbb,train_outdir,"tthbb")

pd_data = pd_data.sample(frac=1).reset_index(drop=True)

print(pd_data.head())

x_total = np.array(pd_data.filter(items = inputvars))
y_total = np.array(pd_data.filter(items = ['category']))

# Splitting between training set and cross-validation set
x_train, x_val, y_train, y_val = train_test_split(x_total, y_total, test_size=0.3)
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

model = tf.keras.models.Sequential()
###############    Input Layer      ###############
model.add(tf.keras.layers.Flatten(input_shape = (x_train.shape[1],)))
###############    Hidden Layer     ###############
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation=activation_function))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation=activation_function, kernel_regularizer='l2', kernel_initializer=weight_initializer))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation=activation_function, kernel_regularizer='l2', kernel_initializer=weight_initializer))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dense(100, activation=activation_function, kernel_regularizer='l2', kernel_initializer=weight_initializer))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dense(100, activation=activation_function, kernel_regularizer='l2', kernel_initializer=weight_initializer))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dense(100, activation=activation_function, kernel_regularizer='l2', kernel_initializer=weight_initializer))
###############    Output Layer     ###############
model.add(tf.keras.layers.Dense(len(class_names), activation="softmax"))
###################################################


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

pred_val = model.predict(x_val)
pred_train = model.predict(x_train)

print(pred_val.shape)
print(pred_val.T.shape)
print(pred_val.T[0].shape)
print(y_val.shape)
print(y_val.T.shape)
val_result = pd.DataFrame(np.array([y_val.T[0], pred_val.T[0]]).T, columns=["True", "Pred"])
train_result = pd.DataFrame(np.array([y_train.T[0], pred_train.T[0]]).T, columns=["True", "Pred"])
plot_output_dist(train_result, val_result, sig="tthbb",savedir=train_outdir)
