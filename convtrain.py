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
class_names = ["tthbb","ttbb"]

print("Start multi Training")
epochs = 1000
inputvars = [
    'njets', 'nbjets', 'nElectron', 'nMuon', 'MET_met',# 'nLepton', 'MET_px', 'MET_py', 
    'Lepton_pt', 'Lepton_eta', 'Lepton_e',# 'Lepton_phi',
    'bjet1_pt', 'bjet1_eta', 'bjet1_phi', 'bjet1_e', 'bjet2_pt', 'bjet2_eta', 'bjet2_phi', 'bjet2_e',
    'Jet_pt1', 'Jet_eta1', 'Jet_phi1', 'Jet_e1', 'Jet_pt2', 'Jet_eta2', 'Jet_phi2', 'Jet_e2',
    'Jet_pt3', 'Jet_eta3', 'Jet_phi3', 'Jet_e3', 'Jet_pt4', 'Jet_eta4', 'Jet_phi4', 'Jet_e4',
    'selbjet1_pt', 'selbjet1_eta', 'selbjet1_phi', 'selbjet1_e', 'selbjet2_pt', 'selbjet2_eta', 'selbjet2_phi', 'selbjet2_e',

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
nClass, nVars = len(class_names), len(inputvars)


# MODIFY!! Input
#df_tthbb = uproot.open(indir+"dnn_tthbb_mu.root")["dnn_input"].arrays(inputvars,library="pd")
#df_ttbb  = uproot.open(indir+"dnn_ttbb_mu.root")["dnn_input"].arrays(inputvars,library="pd")
#df_tthbb = uproot.open(indir+"tthbb+.root:dnn_input").arrays(inputvars,library="pd")
#df_ttbb  = uproot.open(indir+"ttbb+.root:dnn_input").arrays(inputvars,library="pd")
df_tthbb = uproot.open(indir+"tthbb+.root:dnn_input").arrays(library="pd")
df_ttbb  = uproot.open(indir+"ttbb+.root:dnn_input").arrays(library="pd")
#df_ttcc  = uproot.open(indir+"ttcc+.root:dnn_input").arrays(library="pd")
#df_ttlf  = uproot.open(indir+"ttjj+.root:dnn_input").arrays(library="pd")

df_tthbb = df_tthbb[df_tthbb["category"] == 0]
df_ttbb  = df_ttbb[df_ttbb["category"] == 0]

ntrain = min(len(df_tthbb), len(df_ttbb))
print(ntrain)

df_tthbb = df_tthbb.sample(n=ntrain).reset_index(drop=True)
df_ttbb  = df_ttbb.sample(n=ntrain).reset_index(drop=True)

df_tthbb["category"] = 0
print ("before, ttbb: ", df_ttbb)
df_ttbb["category"]  = 1
print ("after, ttbb: ",df_ttbb)
#df_ttcc["category"]  = 2
#df_ttlf["category"]  = 3

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
print (x_train.shape)
print (y_train.shape)
x_train = x_train.reshape(-1, 11, 12, 1)
x_val = x_val.reshape(-1, 11, 12, 1)
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
    tf.keras.layers.Conv2D(filters=3, kernel_size=2, strides=2, activation=tf.nn.relu, use_bias=False, input_shape=(11, 12, 1)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=2, activation=tf.nn.relu, use_bias=False),
    tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=2, activation=tf.nn.relu, use_bias=False),
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
