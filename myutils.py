import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False,  title=None, cmap=plt.cm.Blues, savename="./cm.pdf"):
    # This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True label', xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.savefig(savename)
    plt.gcf().clear()
    return ax

def plot_performance(hist, savedir="./"):
    print("Plotting scores")
    plt.plot(hist.history['sparse_categorical_accuracy'])
    plt.plot(hist.history['val_sparse_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'], loc='lower right')
    plt.savefig(os.path.join(savedir+'/fig_score_acc.pdf'))
    plt.gcf().clear()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'],loc='upper right')
    plt.savefig(os.path.join(savedir+'/fig_score_loss.pdf'))
    plt.gcf().clear()

def plot_output_dist(train, test,sig="tthbb", savedir="./"):
    sig_class = {"tthbb":0,"ttbb":1,"ttbj":2,"ttcc":3,"ttlf":4}
    sigtrain = np.array(train[train["True"]==sig_class[sig]]["Pred"])
    bkgtrain = np.array(train[train["True"]!=sig_class[sig]]["Pred"])
    sigtest = np.array(test[test["True"]==sig_class[sig]]["Pred"])
    bkgtest = np.array(test[test["True"]!=sig_class[sig]]["Pred"])
    bins=40
    scores = [sigtrain, sigtest, bkgtrain, bkgtest]
    #print (scores)
    low = min(np.min(d) for d in scores)
    high = max(np.max(d) for d in scores)

    # test is dotted
    plt.hist(sigtrain, color="b", alpha=0.5, range=(low, high), bins=bins, histtype="stepfilled", density=True, label=sig+" train")
    plt.hist(bkgtrain, color="r", alpha=0.5, range=(low, high), bins=bins, histtype="stepfilled", density=True, label="bkg train")
    # train is filled
#    plt.hist(sigtest, color="b", range=(low, high), bins=bins, histtype="step", density=True, label="sig test")
#    plt.hist(bkgtest, color="r", range=(low, high), bins=bins, histtype="step", density=True, label="bkg test")

    hist, bins = np.histogram(sigtest, bins=bins, range=(low,high), density=True)
    scale = len(sigtest) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label=sig+' test')
    hist, bins = np.histogram(bkgtest, bins=bins, range=(low,high), density=True)
    scale = len(bkgtest) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='bkg test')

    plt.title("Output distribution")
    plt.ylabel("entry")
    plt.xlabel("probability")
    plt.legend(loc='best')
    plt.savefig(os.path.join(savedir+'/fig_output_dist_'+sig+'.pdf'))
    plt.gcf().clear()

def plot_corrMatrix(dataframe, savedir="./", outname=""):
    corrdf = dataframe.corr()
    fig, ax1 = plt.subplots(ncols=1, figsize=(10,9))

    opts = {'cmap': plt.get_cmap("RdBu"),
            'vmin': -1, 'vmax': +1}
    heatmap1 = ax1.pcolor(corrdf, **opts)
    plt.colorbar(heatmap1, ax=ax1)

    labels = corrdf.columns.values
    for ax in (ax1,):
        ax.tick_params(labelsize=8)
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False, ha='right', rotation=90)
        ax.set_yticklabels(labels, minor=False)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(savedir+'/correlation_'+outname+'.pdf'))
    plt.gcf().clear()

def plot_roc_curve(fpr,tpr,auc,savedir="./"):
    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1])
    plt.title('AUC = '+str(auc))
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(savedir+'/fig_roc.pdf'))
    plt.gcf().clear()

def feature_importance(df_data, name_inputvar, outputDir):
    """
    Extract importance of features
            then save it in outside
            load it from outside if it is exist
    Plotting importance by ranking / by order (?)
    """

    model = load_model(outputDir+'/model.h5')
    model.summary()

    input_data = df_data.filter(name_inputvar)

    ################### Feature importance ##################

    mean_1st = []
    mean_2nd = []

    if not os.path.exists(outputDir+'/feature_importance.txt'):

        mean_grads = np.zeros(len(name_inputvar))
        mean_jacobian = np.zeros(len(name_inputvar))

        n_evts = len(input_data)
        fraction = 1
        for idx, row in input_data.iterrows():
            if idx % fraction != 0: continue
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape:
                    inputs = tf.Variable([row.to_numpy()])
                    tape.watch(inputs)
                    tape2.watch(inputs)
                    output = model(inputs)
                g = tape.gradient(output, inputs)
                grads = g.numpy()[0]
            jacobian = tape2.jacobian(g, inputs).numpy()[0]
            for i in range(len(name_inputvar)):
                mean_grads[i] += abs(grads[i])/n_evts*fraction
                mean_jacobian[i] += abs(jacobian[i][0][i])/n_evts*fraction

        print("Average of first order gradient: \n"+str(mean_grads))
        print("Average of second order gradient: \n"+str(mean_jacobian))

        ##save it
        f_out = open(outputDir+'/feature_importance.txt','w')
        f_out.write("Feature importance with model: "+model_name+"\n")
        f_out.write("Average of first order gradient: \n" + str(mean_grads)+"\n")
        f_out.write("Average of second order gradient: \n" + str(mean_jacobian)+"\n")
        f_out.close()

        mean_grads = " ".join(map(str,mean_grads))
        mean_jacobian = " ".join(map(str,mean_jacobian))

        mean_1st = [float(num) for num in mean_grads.split(" ")]
        mean_2nd = [float(num) for num in mean_jacobian.split(" ")]

    else:
        mean_grads = []
        mean_jacobian = []
        count = 0

        print("Reading exisiting importances..")
        with open(outputDir+'/feature_importance.txt','r') as f_in:
            for line in f_in:
                if "." in line:
                  if "[" in line: count = count + 1
                  if count == 1: mean_grads.append(line.strip('[\n]'))
                  elif count == 2: mean_jacobian.append(line.strip('[\n]'))
                  else: print("huh I shouldn't have 3rd term")

        mean_grads = "".join(mean_grads)
        mean_jacobian = "".join(mean_jacobian)

        mean_1st = [float(num) for num in mean_grads.split(" ")]
        mean_2nd = [float(num) for num in mean_jacobian.split(" ")]

    ################### Plotting importance ##################

    #TODO hessian matrix from 2nd order
    #df_impact = pd.DataFrame({'second-order':mean_2nd,'first-order':mean_1st}, index=name_inputvar)

    df_impact = pd.DataFrame({'first-order':mean_1st}, index=name_inputvar)
    df_impact = df_impact.sort_values(['first-order'], ascending = True)
    #df_impact = pd.DataFrame({'second-order':mean_2nd}, index=name_inputvar)
    #df_impact = df_impact.sort_values(['second-order'], ascending = True)

    #df_impact = (df_impact-df_impact.min())/(df_impact.max()-df_impact.min())
    df_impact = df_impact/df_impact.sum()

    #ax = df_impact.plot.barh(color={'first-order':'#002b54', 'second-order':'#b38e50'}, width=0.9, alpha = 0.9)
    ax = df_impact.plot.barh(color={'first-order':'#002b54'}, width=0.9, alpha = 0.9)

    plt.ylim(-0.6, ax.get_yticks()[-1] + 0.6)

    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.xlabel('Normalized Importance')

    #handles,labels = ax.get_legend_handles_labels()
    #handles = [handles[1], handles[0]]
    #labels = [labels[1], labels[0]]
    #ax.legend(handles,labels,loc='best')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(outputDir+"/fig_feature_importance.pdf")
    plt.gcf().clear()

    print("Feature importance extracted!")
