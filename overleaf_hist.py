import uproot
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import os, sys

indir = sys.argv[1]
outdir = indir + "/overleaf/"
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

var = 'bjet1_pt'
nbin, xmin, xmax = 20, 0, 400

plt.hist(tthbb[var], color="tab:red", bins=nbin, range=[xmin, xmax], density=True, histtype="step", label="tthbb")
plt.hist(ttbb[var], color="tab:blue", bins=nbin, range=[xmin, xmax], density=True, histtype="step", label="ttbb")
plt.hist(ttbj[var], color="tab:cyan", bins=nbin, range=[xmin, xmax], density=True, histtype="step", label="ttbj")
plt.hist(ttcc[var], color="tab:green", bins=nbin, range=[xmin, xmax], density=True, histtype="step", label="ttcc")
plt.hist(ttjj[var], color="tab:purple", bins=nbin, range=[xmin, xmax], density=True, histtype="step", label="ttjj")
#plt.title("leading b-tagged jet $p_T$")
#plt.xlabel("$p_T (GeV)$", loc='right')
plt.ylabel("Normalized Entries")
plt.legend()

plt.savefig(outdir+var+".pdf")
plt.clf()
