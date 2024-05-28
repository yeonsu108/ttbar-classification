import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

indir = sys.argv[1]
outdir = indir + "/hists/"
if not os.path.exists(outdir): os.makedirs(outdir)


pd_tthbb = uproot.open(indir+"/tthbb.root:dnn_input").arrays(library="pd")
pd_ttbb = uproot.open(indir+"/ttbb.root:dnn_input").arrays(library="pd")
pd_ttcc = uproot.open(indir+"/ttcc.root:dnn_input").arrays(library="pd")
pd_ttjj = uproot.open(indir+"/ttcc.root:dnn_input").arrays(library="pd")

tthbb = pd_tthbb[pd_tthbb["category"]==1]
tthbb["category"] = 0
ttbb  = pd_ttbb[pd_ttbb["category"]==1]
ttbj  = pd_ttbb[pd_ttbb["category"]==2]
ttcc  = pd_ttcc[pd_ttcc["category"]==3]
ttjj  = pd_ttjj[pd_ttjj["category"]==4]

variables = tthbb.columns
print (variables)

for var in variables:
    print (var)
    nbin, xmin, xmax = 20, 0, 400
    _xmax = tthbb[var].max()
    if _xmax < 300: xmax = 300
    if _xmax < 200: xmax = 200
    if _xmax < 100: xmax = 100
    if _xmax < 20: nbin, xmax = int(_xmax+2), int(_xmax+2)
    if _xmax < 5: nbin, xmin, xmax = 20, -4, 4
    if "dR" in var: nbin, xmin, xmax = 20, 0, 4

    plt.hist(tthbb[var], color="tab:red", bins=nbin, range=[xmin, xmax], density=True, histtype="step", label="tthbb")
    plt.hist(ttbb[var], color="tab:blue", bins=nbin, range=[xmin, xmax], density=True, histtype="step", label="ttbb")
    plt.hist(ttbj[var], color="tab:cyan", bins=nbin, range=[xmin, xmax], density=True, histtype="step", label="ttbj")
    plt.hist(ttcc[var], color="tab:green", bins=nbin, range=[xmin, xmax], density=True, histtype="step", label="ttcc")
    plt.hist(ttjj[var], color="tab:purple", bins=nbin, range=[xmin, xmax], density=True, histtype="step", label="ttjj")
    plt.xlabel(var)
    plt.legend()

    plt.savefig(outdir+var+".pdf")
    plt.clf()
