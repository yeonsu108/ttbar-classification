# how to use: python bookhists.py inputdir
# root files should be in inputdir/
# root files should be output files of ana_ttbar_df.C
# output files (histogram root files) will be generated at inputdir/hist/

import ROOT
import numpy as np
import os, sys
ROOT.gStyle.SetOptStat(0)

# RDF
TreeName = "dnn_input"

indir = sys.argv[1]
print(indir)
outdir = indir+"/hist/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

tthbb = ROOT.RDataFrame(TreeName, indir+"/tthbb*.root")
ttbb  = ROOT.RDataFrame(TreeName, indir+"/ttbb*.root")
ttcc  = ROOT.RDataFrame(TreeName, indir+"/ttcc*.root")
ttjj  = ROOT.RDataFrame(TreeName, indir+"/ttjj*.root")


def add1Dhist(hists, df, outfile, flag=""):
    outfile.cd()
    for h in hists:
        hist = df.Histo1D(ROOT.RDF.TH1DModel(h[0]+flag, h[1], h[2], h[3], h[4]), h[5])
        hist.Write()
        print (hist)

def addhists(df, outfile, lep="mu"):
    hist = [
        ("njets", "Jet multiplicity", 10, 4, 14, "njets"),
        ("nbjets", "bJet multiplicity", 8, 2, 10, "nbjets"),
        #("ncjets", "bJet multiplicity", 8, 0, 8, "ncjets"),
        ("nMuon", "Muon multiplicity", 3, 0, 3, "nMuon"),
        ("nElectron", "Electron multiplicity", 3, 0, 3, "nElectron"),
        ("nLepton", "lepton multiplicity", 3, 0, 3, "nLepton"),
        ("MET_px", "MET px", 30, 0, 300, "MET_px"),
        ("MET_py", "MET py", 30, 0, 300, "MET_py"),
        ("MET_met", "MET", 30, 0, 300, "MET_met"),
        ("MET_eta", "MET eta", 20, -4, 4, "MET_eta"),
        ("MET_phi", "MET phi", 20, -4, 4, "MET_phi"),
        ("Lepton_pt",  "Lepton pt",  30,  0, 300, "Lepton_pt" ),
        ("Lepton_eta", "Lepton eta", 20, -4,   4, "Lepton_eta"),
        ("Lepton_phi", "Lepton phi", 20, -4,   4, "Lepton_phi"),
        ("Lepton_e",   "Lepton e",   30,  0, 300, "Lepton_e"  ),

        ("Jet1_pt",  "Jet1 pt",  40,  0, 400, "Jet1_pt" ),
        ("Jet1_eta", "Jet1 eta", 40, -4,   4, "Jet1_eta"),
        ("Jet1_phi", "Jet1 phi", 40, -4,   4, "Jet1_phi"),
        ("Jet1_e", "Jet1 energy", 20,  0, 200, "Jet1_e"),
        ("Jet1_btag", "Jet1 btag", 2,  0, 2, "Jet1_btag"),
        ("Jet2_pt",  "Jet2 pt",  40,  0, 400, "Jet2_pt" ),
        ("Jet2_eta", "Jet2 eta", 40, -4,   4, "Jet2_eta"),
        ("Jet2_phi", "Jet2 phi", 40, -4,   4, "Jet2_phi"),
        ("Jet2_btag", "Jet2 btag", 2,  0, 2, "Jet2_btag"),
        ("Jet2_e", "Jet2 energy", 20,  0, 200, "Jet2_e"),
        ("Jet3_pt",  "Jet3 pt",  40,  0, 400, "Jet3_pt" ),
        ("Jet3_eta", "Jet3 eta", 40, -4,   4, "Jet3_eta"),
        ("Jet3_phi", "Jet3 phi", 40, -4,   4, "Jet3_phi"),
        ("Jet3_e", "Jet3 energy", 20,  0, 200, "Jet3_e"),
        ("Jet3_btag", "Jet3 btag", 2,  0, 2, "Jet3_btag"),
        ("Jet4_pt",  "Jet4 pt",  40,  0, 400, "Jet4_pt" ),
        ("Jet4_eta", "Jet4 eta", 40, -4,   4, "Jet4_eta"),
        ("Jet4_phi", "Jet4 phi", 40, -4,   4, "Jet4_phi"),
        ("Jet4_e", "Jet4 energy", 20,  0, 200, "Jet4_e"),
        ("Jet4_btag", "Jet4 btag", 2,  0, 2, "Jet4_btag"),

        ("bjet1_pt",  "bjet1 pt",  40,  0, 400, "bjet1_pt" ),
        ("bjet1_eta", "bjet1 eta", 40, -4,   4, "bjet1_eta"),
        ("bjet1_phi", "bjet1 phi", 40, -4,   4, "bjet1_phi"),
        ("bjet1_e", "bjet1 energy", 20,  0, 200, "bjet1_e"),
        ("bjet2_pt",  "bjet2 pt",  40,  0, 400, "bjet2_pt" ),
        ("bjet2_eta", "bjet2 eta", 40, -4,   4, "bjet2_eta"),
        ("bjet2_phi", "bjet2 phi", 40, -4,   4, "bjet2_phi"),
        ("bjet2_e", "bjet2 energy", 20,  0, 200, "bjet2_e"),

        ("selbjet1_pt",  "selbjet1 pt",  40,  0, 400, "selbjet1_pt" ),
        ("selbjet1_eta", "selbjet1 eta", 40, -4,   4, "selbjet1_eta"),
        ("selbjet1_phi", "selbjet1 phi", 40, -4,   4, "selbjet1_phi"),
        ("selbjet1_e", "selbjet1 energy", 20,  0, 200, "selbjet1_e"),
        ("selbjet2_pt",  "selbjet2 pt",  40,  0, 400, "selbjet2_pt" ),
        ("selbjet2_eta", "selbjet2 eta", 40, -4,   4, "selbjet2_eta"),
        ("selbjet2_phi", "selbjet2 phi", 40, -4,   4, "selbjet2_phi"),
        ("selbjet2_e", "selbjet2 energy", 20,  0, 200, "selbjet2_e"),

        ("bbdR", "dR(bb)", 40, -4, 4, "bbdR"),
        ("bbdEta", "dEta(bb)", 40, -4, 4, "bbdEta"),
        ("bbdPhi", "dPhi(bb)", 40, -4, 4, "bbdPhi"),
        ("bbPt", "pT(bb)", 40, 0, 400, "bbPt"),
        ("bbEta", "Eta(bb)", 40, -4, 4, "bbEta"),
        ("bbPhi", "Phi(bb)", 40, -4, 4, "bbPhi"),
        ("bbMass", "M(bb)", 20, 0, 200, "bbMass"),
        ("bbHt", "Ht(bb)", 40, 0, 400, "bbPt"),
        ("bbMt", "Mt(bb)", 40, 0, 400, "bbPt"),

        ("nub1dR", "dR(nub1)", 40, -4, 4, "nub1dR"),
        ("nub1dEta", "dEta(nub1)", 40, -4, 4, "nub1dEta"),
        ("nub1dPhi", "dPhi(nub1)", 40, -4, 4, "nub1dPhi"),
        ("nub1Pt", "pT(nub1)", 40, 0, 400, "nub1Pt"),
        ("nub1Eta", "Eta(nub1)", 40, -4, 4, "nub1Eta"),
        ("nub1Phi", "Phi(nub1)", 40, -4, 4, "nub1Phi"),
        ("nub1Mass", "M(nub1)", 20, 0, 200, "nub1Mass"),
        ("nub1Ht", "Ht(nub1)", 40, 0, 400, "nub1Pt"),
        ("nub1Mt", "Mt(nub1)", 40, 0, 400, "nub1Pt"),

        ("nub2dR", "dR(nub2)", 40, -4, 4, "nub2dR"),
        ("nub2dEta", "dEta(nub2)", 40, -4, 4, "nub2dEta"),
        ("nub2dPhi", "dPhi(nub2)", 40, -4, 4, "nub2dPhi"),
        ("nub2Pt", "pT(nub2)", 40, 0, 400, "nub2Pt"),
        ("nub2Eta", "Eta(nub2)", 40, -4, 4, "nub2Eta"),
        ("nub2Phi", "Phi(nub2)", 40, -4, 4, "nub2Phi"),
        ("nub2Mass", "M(nub2)", 20, 0, 200, "nub2Mass"),
        ("nub2Ht", "Ht(nub2)", 40, 0, 400, "nub2Pt"),
        ("nub2Mt", "Mt(nub2)", 40, 0, 400, "nub2Pt"),

        ("nubbdR", "dR(nubb)", 40, -4, 4, "nubbdR"),
        ("nubbdEta", "dEta(nubb)", 40, -4, 4, "nubbdEta"),
        ("nubbdPhi", "dPhi(nubb)", 40, -4, 4, "nubbdPhi"),
        ("nubbPt", "pT(nubb)", 40, 0, 400, "nubbPt"),
        ("nubbEta", "Eta(nubb)", 40, -4, 4, "nubbEta"),
        ("nubbPhi", "Phi(nubb)", 40, -4, 4, "nubbPhi"),
        ("nubbMass", "M(nubb)", 20, 0, 200, "nubbMass"),
        ("nubbHt", "Ht(nubb)", 40, 0, 400, "nubbPt"),
        ("nubbMt", "Mt(nubb)", 40, 0, 400, "nubbPt"),

        ("lb1dR", "dR(lb1)", 40, -4, 4, "lb1dR"),
        ("lb1dEta", "dEta(lb1)", 40, -4, 4, "lb1dEta"),
        ("lb1dPhi", "dPhi(lb1)", 40, -4, 4, "lb1dPhi"),
        ("lb1Pt", "pT(lb1)", 40, 0, 400, "lb1Pt"),
        ("lb1Eta", "Eta(lb1)", 40, -4, 4, "lb1Eta"),
        ("lb1Phi", "Phi(lb1)", 40, -4, 4, "lb1Phi"),
        ("lb1Mass", "M(lb1)", 20, 0, 200, "lb1Mass"),
        ("lb1Ht", "Ht(lb1)", 40, 0, 400, "lb1Pt"),
        ("lb1Mt", "Mt(lb1)", 40, 0, 400, "lb1Pt"),

        ("lb2dR", "dR(lb2)", 40, -4, 4, "lb2dR"),
        ("lb2dEta", "dEta(lb2)", 40, -4, 4, "lb2dEta"),
        ("lb2dPhi", "dPhi(lb2)", 40, -4, 4, "lb2dPhi"),
        ("lb2Pt", "pT(lb2)", 40, 0, 400, "lb2Pt"),
        ("lb2Eta", "Eta(lb2)", 40, -4, 4, "lb2Eta"),
        ("lb2Phi", "Phi(lb2)", 40, -4, 4, "lb2Phi"),
        ("lb2Mass", "M(lb2)", 20, 0, 200, "lb2Mass"),
        ("lb2Ht", "Ht(lb2)", 40, 0, 400, "lb2Pt"),
        ("lb2Mt", "Mt(lb2)", 40, 0, 400, "lb2Pt"),

        ("lbbdR", "dR(lbb)", 40, -4, 4, "lbbdR"),
        ("lbbdEta", "dEta(lbb)", 40, -4, 4, "lbbdEta"),
        ("lbbdPhi", "dPhi(lbb)", 40, -4, 4, "lbbdPhi"),
        ("lbbPt", "pT(lbb)", 40, 0, 400, "lbbPt"),
        ("lbbEta", "Eta(lbb)", 40, -4, 4, "lbbEta"),
        ("lbbPhi", "Phi(lbb)", 40, -4, 4, "lbbPhi"),
        ("lbbMass", "M(lbb)", 20, 0, 200, "lbbMass"),
        ("lbbHt", "Ht(lbb)", 40, 0, 400, "lbbPt"),
        ("lbbMt", "Mt(lbb)", 40, 0, 400, "lbbPt"),

        ("Wjb1dR", "dR(Wjb1)", 40, -4, 4, "Wjb1dR"),
        ("Wjb1dEta", "dEta(Wjb1)", 40, -4, 4, "Wjb1dEta"),
        ("Wjb1dPhi", "dPhi(Wjb1)", 40, -4, 4, "Wjb1dPhi"),
        ("Wjb1Pt", "pT(Wjb1)", 40, 0, 400, "Wjb1Pt"),
        ("Wjb1Eta", "Eta(Wjb1)", 40, -4, 4, "Wjb1Eta"),
        ("Wjb1Phi", "Phi(Wjb1)", 40, -4, 4, "Wjb1Phi"),
        ("Wjb1Mass", "M(Wjb1)", 20, 0, 200, "Wjb1Mass"),
        ("Wjb1Ht", "Ht(Wjb1)", 40, 0, 400, "Wjb1Pt"),
        ("Wjb1Mt", "Mt(Wjb1)", 40, 0, 400, "Wjb1Pt"),

        ("Wjb2dR", "dR(Wjb2)", 40, -4, 4, "Wjb2dR"),
        ("Wjb2dEta", "dEta(Wjb2)", 40, -4, 4, "Wjb2dEta"),
        ("Wjb2dPhi", "dPhi(Wjb2)", 40, -4, 4, "Wjb2dPhi"),
        ("Wjb2Pt", "pT(Wjb2)", 40, 0, 400, "Wjb2Pt"),
        ("Wjb2Eta", "Eta(Wjb2)", 40, -4, 4, "Wjb2Eta"),
        ("Wjb2Phi", "Phi(Wjb2)", 40, -4, 4, "Wjb2Phi"),
        ("Wjb2Mass", "M(Wjb2)", 20, 0, 200, "Wjb2Mass"),
        ("Wjb2Ht", "Ht(Wjb2)", 40, 0, 400, "Wjb2Pt"),
        ("Wjb2Mt", "Mt(Wjb2)", 40, 0, 400, "Wjb2Pt"),

        ("Wlb1dR", "dR(Wlb1)", 40, -4, 4, "Wlb1dR"),
        ("Wlb1dEta", "dEta(Wlb1)", 40, -4, 4, "Wlb1dEta"),
        ("Wlb1dPhi", "dPhi(Wlb1)", 40, -4, 4, "Wlb1dPhi"),
        ("Wlb1Pt", "pT(Wlb1)", 40, 0, 400, "Wlb1Pt"),
        ("Wlb1Eta", "Eta(Wlb1)", 40, -4, 4, "Wlb1Eta"),
        ("Wlb1Phi", "Phi(Wlb1)", 40, -4, 4, "Wlb1Phi"),
        ("Wlb1Mass", "M(Wlb1)", 20, 0, 200, "Wlb1Mass"),
        ("Wlb1Ht", "Ht(Wlb1)", 40, 0, 400, "Wlb1Pt"),
        ("Wlb1Mt", "Mt(Wlb1)", 40, 0, 400, "Wlb1Pt"),

        ("Wlb2dR", "dR(Wlb2)", 40, -4, 4, "Wlb2dR"),
        ("Wlb2dEta", "dEta(Wlb2)", 40, -4, 4, "Wlb2dEta"),
        ("Wlb2dPhi", "dPhi(Wlb2)", 40, -4, 4, "Wlb2dPhi"),
        ("Wlb2Pt", "pT(Wlb2)", 40, 0, 400, "Wlb2Pt"),
        ("Wlb2Eta", "Eta(Wlb2)", 40, -4, 4, "Wlb2Eta"),
        ("Wlb2Phi", "Phi(Wlb2)", 40, -4, 4, "Wlb2Phi"),
        ("Wlb2Mass", "M(Wlb2)", 20, 0, 200, "Wlb2Mass"),
        ("Wlb2Ht", "Ht(Wlb2)", 40, 0, 400, "Wlb2Pt"),
        ("Wlb2Mt", "Mt(Wlb2)", 40, 0, 400, "Wlb2Pt"),
    ]
    add1Dhist(hist, df, outfile, "")
    return df


print("tthbb")
hist_tthbb = ROOT.TFile.Open(outdir+"hist_tthbb.root", "RECREATE")
df_tthbb = addhists(tthbb, hist_tthbb)

print("ttbb")
hist_ttbb  = ROOT.TFile.Open(outdir+"hist_ttbb.root", "RECREATE")
df_ttbb  = addhists(ttbb, hist_ttbb)

print("ttcc")
hist_ttcc  = ROOT.TFile.Open(outdir+"hist_ttcc.root", "RECREATE")
df_ttcc  = addhists(ttcc, hist_ttcc)

print("ttjj")
hist_ttjj  = ROOT.TFile.Open(outdir+"hist_ttjj.root", "RECREATE")
df_ttjj  = addhists(ttjj, hist_ttjj)
