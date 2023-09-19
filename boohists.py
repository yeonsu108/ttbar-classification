# how to use: python bookhists.py inputdir lepton_channel
# root files should be in inputdir/
# lepton_channel: mu or elec
# output files (histogram root files) are generated at inputdir/hist/

import ROOT
import numpy as np
import os, sys
ROOT.gStyle.SetOptStat(0)

# RDF
TreeName = "Delphes"

# Modify!! 
indir = sys.argv[1]
print(indir)
outdir = indir+"/hist/"
if not os.path.exists(outdir):
    os.makedirs(outdir)
lep = sys.argv[2]    # mu or elec

tthbb = ROOT.RDataFrame(TreeName, indir+"/tthbblj*.root")
ttbb  = ROOT.RDataFrame(TreeName, indir+"/ttbblj*.root")
hist_tthbb = ROOT.TFile.Open(outdir+"hist_tthbb_"+lep+".root", "RECREATE")
hist_ttbb  = ROOT.TFile.Open(outdir+"hist_ttbb_"+lep+".root", "RECREATE")

def add1Dhist(hists, df, outfile, flag="_S0"):
    outfile.cd()
    for h in hists:
        hist = df.Histo1D(ROOT.RDF.TH1DModel(h[0]+flag, h[1], h[2], h[3], h[4]), h[5])
        hist.Write()
        print (hist)

def addhists(df, outfile):
    hist_S0 = [
        ("njet", "Jet multiplicity", 14, 0, 14, "njet"),
        ("nbjet", "bJet multiplicity", 10, 0, 10, "nbjet"),
        ("nmuon", "Muon multiplicity", 3, 0, 3, "nmuon"),
        ("nelectron", "Electron multiplicity", 3, 0, 3, "nelectron"),
        ("nlepton", "lepton multiplicity", 3, 0, 3, "nlepton"),
        ("MET_met", "MET", 30, 0, 300, "MET_met"),
        ("MET_eta", "MET eta", 20, -4, 4, "MET_eta"),
        ("MET_phi", "MET phi", 20, -4, 4, "MET_phi"),
    ]
    add1Dhist(hist_S0, df, outfile, "_S0")


    if lep == "mu":
        tmp = "Muon"
        df = df.Filter("nmuon==1 && nelectron==0")
    else:
        tmp = "Electron"
        df = df.Filter("nmuon==0 && nelectron==1")
    hist_S1 = hist_S0+[
        (tmp+"_pt",  tmp+" pt",  30,  0, 300, tmp+"_pt" ),
        (tmp+"_eta", tmp+" eta", 20, -4,   4, tmp+"_eta"),
        (tmp+"_phi", tmp+" phi", 20, -4,   4, tmp+"_phi"),
        (tmp+"_e",   tmp+" e",   30,  0, 300, tmp+"_e"  ),
    ]

    add1Dhist(hist_S1, df, outfile, "_S1")

    hist_S2jet1 = hist_S1 + [
        ("Jet1_pt",  "Jet1 pt",  40,  0, 400, "Jet1_pt" ),
        ("Jet1_eta", "Jet1 eta", 40, -4,   4, "Jet1_eta"),
        ("Jet1_phi", "Jet1 phi", 40, -4,   4, "Jet1_phi"),
        ("Jet1_mass", "Jet1 mass", 40,  0, 400, "Jet1_mass"),
    ]
    df = df.Filter("njet>=1") \
           .Define("Jet1_pt", "Jet_pt[0]").Define("Jet1_eta", "Jet_eta[0]").Define("Jet1_phi", "Jet_phi[0]").Define("Jet1_mass", "Jet_mass[0]")
    add1Dhist(hist_S2jet1, df, outfile, "_S2jet1")

    hist_S2jet2 = hist_S2jet1 + [
        ("Jet2_pt",  "Jet2 pt",  40,  0, 400, "Jet2_pt" ),
        ("Jet2_eta", "Jet2 eta", 40, -4,   4, "Jet2_eta"),
        ("Jet2_phi", "Jet2 phi", 40, -4,   4, "Jet2_phi"),
        ("Jet2_mass", "Jet2 mass", 40,  0, 400, "Jet2_mass"),
    ]
    df = df.Filter("njet>=2") \
           .Define("Jet2_pt", "Jet_pt[1]").Define("Jet2_eta", "Jet_eta[1]").Define("Jet2_phi", "Jet_phi[1]").Define("Jet2_mass", "Jet_mass[1]")
    add1Dhist(hist_S2jet2, df, outfile, "_S2jet2")

    hist_S2jet3 = hist_S2jet2 + [
        ("Jet3_pt",  "Jet3 pt",  40,  0, 400, "Jet3_pt" ),
        ("Jet3_eta", "Jet3 eta", 40, -4,   4, "Jet3_eta"),
        ("Jet3_phi", "Jet3 phi", 40, -4,   4, "Jet3_phi"),
        ("Jet3_mass", "Jet3 mass", 40,  0, 400, "Jet3_mass"),
    ]
    df = df.Filter("njet>=3") \
           .Define("Jet3_pt", "Jet_pt[2]").Define("Jet3_eta", "Jet_eta[2]").Define("Jet3_phi", "Jet_phi[2]").Define("Jet3_mass", "Jet_mass[2]")
    add1Dhist(hist_S2jet3, df, outfile, "_S2jet3")

    hist_S2jet4 = hist_S2jet3 + [
        ("Jet4_pt",  "Jet4 pt",  40,  0, 400, "Jet4_pt" ),
        ("Jet4_eta", "Jet4 eta", 40, -4,   4, "Jet4_eta"),
        ("Jet4_phi", "Jet4 phi", 40, -4,   4, "Jet4_phi"),
        ("Jet4_mass", "Jet4 mass", 40,  0, 400, "Jet4_mass"),
    ]
    df = df.Filter("njet>=4") \
           .Define("Jet4_pt", "Jet_pt[3]").Define("Jet4_eta", "Jet_eta[3]").Define("Jet4_phi", "Jet_phi[3]").Define("Jet4_mass", "Jet_mass[3]")
    add1Dhist(hist_S2jet4, df, outfile, "_S2jet4")

    hist_S2jet5 = hist_S2jet4 + [
        ("Jet5_pt",  "Jet5 pt",  40,  0, 400, "Jet5_pt" ),
        ("Jet5_eta", "Jet5 eta", 40, -4,   4, "Jet5_eta"),
        ("Jet5_phi", "Jet5 phi", 40, -4,   4, "Jet5_phi"),
        ("Jet5_mass", "Jet5 mass", 40,  0, 400, "Jet5_mass"),
    ]
    tmp = df.Filter("njet>=5") \
            .Define("Jet5_pt", "Jet_pt[4]").Define("Jet5_eta", "Jet_eta[4]").Define("Jet5_phi", "Jet_phi[4]").Define("Jet5_mass", "Jet_mass[4]")
    add1Dhist(hist_S2jet5, tmp, outfile, "_S2jet5")

    hist_S2jet6 = hist_S2jet5 + [
        ("Jet6_pt",  "Jet6 pt",  40,  0, 400, "Jet6_pt" ),
        ("Jet6_eta", "Jet6 eta", 40, -4,   4, "Jet6_eta"),
        ("Jet6_phi", "Jet6 phi", 40, -4,   4, "Jet6_phi"),
        ("Jet6_mass", "Jet6 mass", 40,  0, 400, "Jet6_mass"),
    ]
    tmp = tmp.Filter("njet>=6") \
             .Define("Jet6_pt", "Jet_pt[5]").Define("Jet6_eta", "Jet_eta[5]").Define("Jet6_phi", "Jet_phi[5]").Define("Jet6_mass", "Jet_mass[5]")
    add1Dhist(hist_S2jet6, tmp, outfile, "_S2jet6")

    hist_S3bjet1 = hist_S2jet4 + [
        ("bJet1_pt",  "bJet1 pt",  40,  0, 400, "bJet1_pt" ),
        ("bJet1_eta", "bJet1 eta", 40, -4,   4, "bJet1_eta"),
        ("bJet1_phi", "bJet1 phi", 40, -4,   4, "bJet1_phi"),
        ("bJet1_mass", "bJet1 mass", 40,  0, 400, "bJet1_mass"),
    ]
    df = df.Filter("nbjet>=1") \
           .Define("bJet1_pt", "bJet_pt[0]").Define("bJet1_eta", "bJet_eta[0]").Define("bJet1_phi", "bJet_phi[0]").Define("bJet1_mass", "bJet_mass[0]")
    add1Dhist(hist_S3bjet1, df, outfile, "_S3bjet1")

    hist_S3bjet2 = hist_S3bjet1 + [
        ("bJet2_pt",  "bJet2 pt",  40,  0, 400, "bJet2_pt" ),
        ("bJet2_eta", "bJet2 eta", 40, -4,   4, "bJet2_eta"),
        ("bJet2_phi", "bJet2 phi", 40, -4,   4, "bJet2_phi"),
        ("bJet2_mass", "bJet2 mass", 40,  0, 400, "bJet2_mass"),
    ]
    df = df.Filter("njet>=2") \
           .Define("bJet2_pt", "bJet_pt[1]").Define("bJet2_eta", "bJet_eta[1]").Define("bJet2_phi", "bJet_phi[1]").Define("bJet2_mass", "bJet_mass[1]")
    add1Dhist(hist_S3bjet2, df, outfile, "_S3bjet2")

    hist_S3bjet3 = hist_S3bjet2 + [
        ("bJet3_pt",  "bJet3 pt",  40,  0, 400, "bJet3_pt" ),
        ("bJet3_eta", "bJet3 eta", 40, -4,   4, "bJet3_eta"),
        ("bJet3_phi", "bJet3 phi", 40, -4,   4, "bJet3_phi"),
        ("bJet3_mass", "bJet3 mass", 40,  0, 400, "bJet3_mass"),
    ]
    df = df.Filter("nbjet>=3") \
           .Define("bJet3_pt", "bJet_pt[2]").Define("bJet3_eta", "bJet_eta[2]").Define("bJet3_phi", "bJet_phi[2]").Define("bJet3_mass", "bJet_mass[2]")
    add1Dhist(hist_S3bjet3, df, outfile, "_S3bjet3")

    hist_S3bjet4 = hist_S3bjet3 + [
        ("bJet4_pt",  "bJet4 pt",  40,  0, 400, "bJet4_pt" ),
        ("bJet4_eta", "bJet4 eta", 40, -4,   4, "bJet4_eta"),
        ("bJet4_phi", "bJet4 phi", 40, -4,   4, "bJet4_phi"),
        ("bJet4_mass", "bJet4 mass", 40,  0, 400, "bJet4_mass"),
    ]
    df = df.Filter("nbjet>=4") \
           .Define("bJet4_pt", "bJet_pt[3]").Define("bJet4_eta", "bJet_eta[3]").Define("bJet4_phi", "bJet_phi[3]").Define("bJet4_mass", "bJet_mass[3]")
    add1Dhist(hist_S3bjet4, df, outfile, "_S3bjet4")

addhists(tthbb, hist_tthbb)
addhists(ttbb, hist_ttbb)
