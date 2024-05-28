# how to use: python cutflow.py inputdir lepton_channel
# root files should be in inputdir/
# root files should be outputs of ana_ttbar_df.C 
# lepton_channel: mu or elec

import ROOT
import numpy as np
import os, sys

TreeName = "Delphes"

indir = sys.argv[1]
tthbb = ROOT.RDataFrame(TreeName, indir+"/tthbblj*.root")
ttbb  = ROOT.RDataFrame(TreeName, indir+"/ttbblj*.root")

# Luminosity [pb^-1]
L = 3000 * 1000

# cross section (pb)
xsec_tthbb = 0.06786 + 0.06783
xsec_ttbb  = 0.3963  + 0.3964

# Number of events
ntthbb = L * xsec_tthbb
nttbb  = L * xsec_ttbb


# MODIFY!! E.S.
def Acceptance(df):
    Accept = []
    nocut = float(df.Count().GetValue())
    df = df.Filter("nGenAddbJet >= 2")
    S0 = float(df.Count().GetValue())
    S1 = float(df.Filter("nmuon + nelectron == 1").Count().GetValue())
    S2 = float(df.Filter("nmuon + nelectron == 1 && njet >= 4").Count().GetValue())
    S3 = float(df.Filter("nmuon + nelectron == 1 && njet >= 4 && nbjet >= 2").Count().GetValue())

    S1mu = float(df.Filter("nmuon == 1 && nelectron == 0").Count().GetValue())
    S2mu = float(df.Filter("nmuon == 1 && nelectron == 0 && njet >= 4").Count().GetValue())
    S3mu = float(df.Filter("nmuon == 1 && nelectron == 0 && njet >= 4 && nbjet >= 2").Count().GetValue())

    S1elec = float(df.Filter("nmuon == 0 && nelectron == 1").Count().GetValue())
    S2elec = float(df.Filter("nmuon == 0 && nelectron == 1 && njet >= 4").Count().GetValue())
    S3elec = float(df.Filter("nmuon == 0 && nelectron == 1 && njet >= 4 && nbjet >= 2").Count().GetValue())
    Accept.extend([[nocut, S0, S1, S2, S3], [S1mu, S2mu, S3mu], [S1elec, S2elec, S3elec]])
    print(Accept)
    return Accept
    '''Yeild
    print("noCut : ", S0 , (S0/S0)*100, "%")
    print("S1 : ", S1, (S1/S0)*100, "%")
    print("S2 : ", S2, (S2/S0)*100, "%")
    print("S3 : ", S3, (S3/S0)*100, "%")
    '''

print("________ACCEPTANCE________")
print("tthbb")
tthbb = Acceptance(tthbb)[0]
print("ttbb")
ttbb = Acceptance(ttbb)[0]

Acc = {
    "tthbb" : [tthbb, ntthbb/tthbb[0]],
    "ttbb"  : [ttbb,  nttbb/ttbb[0]]
}

# Cutflow
def Cutflow(Acc):
    for key, value in Acc.items():
        value[0] = [element * value[1] for element in value[0]]
        print(key, value[0])
    return Acc

print("__________CUTFLOW__________")        
Acc_re = Cutflow(Acc)

print(" ")
print("________SIGNIFICANCE________")

# Significance
for i in range(5):
    print("Siignificance of ES :", i)
    print("%.1f" % (Acc_re["tthbb"][0][i]/np.sqrt(Acc_re["ttbb"][0][i])))



