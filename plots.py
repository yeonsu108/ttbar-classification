# how to use: python plots.py inputdir lepton_channel
# histogram root files should be in inputdir/hist/
# lepton_channel: mu or elec

import ROOT
import numpy as np
import os, sys
ROOT.gStyle.SetOptStat(0)

# RDF
TreeName = "Delphes"

# Modify!! 
indir = sys.argv[1]
lep = sys.argv[2] # mu or elec
print(indir)
outdir = indir+"plots/"
if not os.path.exists(outdir):
    os.makedirs(outdir)
tthbb  = ROOT.TFile(indir + "/hist/" + "hist_tthbb_"+lep+".root")
ttbb   = ROOT.TFile(indir + "/hist/" + "hist_ttbb_"+lep+".root")

def drawhisto(hists, files, flag="_S0"):
    canvas = ROOT.TCanvas("c", "c", 400, 400)
    for hname in hists:
        hist_dict = {}
        legend = ROOT.TLegend(0.75, 0.9, 0.9, 0.8)
        ymax, color = 0, 1
        for fname, f in files.items():
            h = f.Get(hname)
            if ymax < h.GetMaximum(): ymax = h.GetMaximum()
            h.GetXaxis().SetTitle(h.GetTitle())
            h.GetYaxis().SetTitle("Normalized entries")
            h.GetYaxis().SetTitleOffset(1.5)
            h.SetLineColor(color)
            color+=1
            h.SetLineWidth(2)
            legend.AddEntry(h, fname, "l")
            hist_dict[hname + "_" + fname] = h

        first = True
        for _hname, h in hist_dict.items():
            h.SetMaximum(ymax * 1.2)
            if first:
                h.DrawNormalized("hist")
                #h.Draw("hist")
                first = False
            else:
                h.DrawNormalized("histsame")
                #h.Draw("histsame")
        legend.Draw()
        canvas.Print(outdir + hname + ".pdf")
        canvas.Clear()

if lep == mu: lep = "Muon"
else: lep = "Electron"
## Histogram Features  
hists_lep = [
    lep+"_pt_S1", lep+"_eta_S1", lep+"_phi_S1", lep+"_e_S1",
    lep+"_pt_S2jet1", lep+"_eta_S2jet1", lep+"_phi_S2jet1", lep+"_e_S2jet1",
    lep+"_pt_S2jet2", lep+"_eta_S2jet2", lep+"_phi_S2jet2", lep+"_e_S2jet2",
    lep+"_pt_S2jet3", lep+"_eta_S2jet3", lep+"_phi_S2jet3", lep+"_e_S2jet3",
    lep+"_pt_S2jet4", lep+"_eta_S2jet4", lep+"_phi_S2jet4", lep+"_e_S2jet4",
    lep+"_pt_S2jet5", lep+"_eta_S2jet5", lep+"_phi_S2jet5", lep+"_e_S2jet5",
    lep+"_pt_S2jet6", lep+"_eta_S2jet6", lep+"_phi_S2jet6", lep+"_e_S2jet6",
    lep+"_pt_S3bjet1", lep+"_eta_S3bjet1", lep+"_phi_S3bjet1", lep+"_e_S3bjet1",
    lep+"_pt_S3bjet2", lep+"_eta_S3bjet2", lep+"_phi_S3bjet2", lep+"_e_S3bjet2",
    lep+"_pt_S3bjet3", lep+"_eta_S3bjet3", lep+"_phi_S3bjet3", lep+"_e_S3bjet3",
    lep+"_pt_S3bjet4", lep+"_eta_S3bjet4", lep+"_phi_S3bjet4", lep+"_e_S3bjet4",
]
hists = hists_lep+[
    "njet_S0", "nbjet_S0", "nmuon_S0", "nelectron_S0", "nlepton_S0", "MET_met_S0", "MET_eta_S0", "MET_phi_S0",

    "njet_S1", "nbjet_S1", "nmuon_S1", "nelectron_S1", "nlepton_S1", "MET_met_S1", "MET_eta_S1", "MET_phi_S1",

    "njet_S2jet1", "nbjet_S2jet1", "nmuon_S2jet1", "nelectron_S2jet1", "nlepton_S2jet1", "MET_met_S2jet1", "MET_eta_S2jet1", "MET_phi_S2jet1",
    "Jet1_pt_S2jet1", "Jet1_eta_S2jet1", "Jet1_phi_S2jet1", "Jet1_mass_S2jet1",

    "njet_S2jet2", "nbjet_S2jet2", "nmuon_S2jet2", "nelectron_S2jet2", "nlepton_S2jet2", "MET_met_S2jet2", "MET_eta_S2jet2", "MET_phi_S2jet2",
    "Jet1_pt_S2jet2", "Jet1_eta_S2jet2", "Jet1_phi_S2jet2", "Jet1_mass_S2jet2",
    "Jet2_pt_S2jet2", "Jet2_eta_S2jet2", "Jet2_phi_S2jet2", "Jet2_mass_S2jet2",

    "njet_S2jet3", "nbjet_S2jet3", "nmuon_S2jet3", "nelectron_S2jet3", "nlepton_S2jet3", "MET_met_S2jet3", "MET_eta_S2jet3", "MET_phi_S2jet3",
    "Jet1_pt_S2jet3", "Jet1_eta_S2jet3", "Jet1_phi_S2jet3", "Jet1_mass_S2jet3",
    "Jet2_pt_S2jet3", "Jet2_eta_S2jet3", "Jet2_phi_S2jet3", "Jet2_mass_S2jet3",
    "Jet3_pt_S2jet3", "Jet3_eta_S2jet3", "Jet3_phi_S2jet3", "Jet3_mass_S2jet3",

    "njet_S2jet4", "nbjet_S2jet4", "nmuon_S2jet4", "nelectron_S2jet4", "nlepton_S2jet4", "MET_met_S2jet4", "MET_eta_S2jet4", "MET_phi_S2jet4",
    "Jet1_pt_S2jet4", "Jet1_eta_S2jet4", "Jet1_phi_S2jet4", "Jet1_mass_S2jet4",
    "Jet2_pt_S2jet4", "Jet2_eta_S2jet4", "Jet2_phi_S2jet4", "Jet2_mass_S2jet4",
    "Jet3_pt_S2jet4", "Jet3_eta_S2jet4", "Jet3_phi_S2jet4", "Jet3_mass_S2jet4",
    "Jet4_pt_S2jet4", "Jet4_eta_S2jet4", "Jet4_phi_S2jet4", "Jet4_mass_S2jet4",

    "njet_S2jet5", "nbjet_S2jet5", "nmuon_S2jet5", "nelectron_S2jet5", "nlepton_S2jet5", "MET_met_S2jet5", "MET_eta_S2jet5", "MET_phi_S2jet5",
    "Jet1_pt_S2jet5", "Jet1_eta_S2jet5", "Jet1_phi_S2jet5", "Jet1_mass_S2jet5",
    "Jet2_pt_S2jet5", "Jet2_eta_S2jet5", "Jet2_phi_S2jet5", "Jet2_mass_S2jet5",
    "Jet3_pt_S2jet5", "Jet3_eta_S2jet5", "Jet3_phi_S2jet5", "Jet3_mass_S2jet5",
    "Jet4_pt_S2jet5", "Jet4_eta_S2jet5", "Jet4_phi_S2jet5", "Jet4_mass_S2jet5",
    "Jet5_pt_S2jet5", "Jet5_eta_S2jet5", "Jet5_phi_S2jet5", "Jet5_mass_S2jet5",

    "njet_S2jet6", "nbjet_S2jet6", "nmuon_S2jet6", "nelectron_S2jet6", "nlepton_S2jet6", "MET_met_S2jet6", "MET_eta_S2jet6", "MET_phi_S2jet6",
    "Jet1_pt_S2jet6", "Jet1_eta_S2jet6", "Jet1_phi_S2jet6", "Jet1_mass_S2jet6",
    "Jet2_pt_S2jet6", "Jet2_eta_S2jet6", "Jet2_phi_S2jet6", "Jet2_mass_S2jet6",
    "Jet3_pt_S2jet6", "Jet3_eta_S2jet6", "Jet3_phi_S2jet6", "Jet3_mass_S2jet6",
    "Jet4_pt_S2jet6", "Jet4_eta_S2jet6", "Jet4_phi_S2jet6", "Jet4_mass_S2jet6",
    "Jet5_pt_S2jet6", "Jet5_eta_S2jet6", "Jet5_phi_S2jet6", "Jet5_mass_S2jet6",
    "Jet6_pt_S2jet6", "Jet6_eta_S2jet6", "Jet6_phi_S2jet6", "Jet6_mass_S2jet6",

    "njet_S3bjet1", "nbjet_S3bjet1", "nmuon_S3bjet1", "nelectron_S3bjet1", "nlepton_S3bjet1", "MET_met_S3bjet1", "MET_eta_S3bjet1", "MET_phi_S3bjet1",
    "Jet1_pt_S3bjet1", "Jet1_eta_S3bjet1", "Jet1_phi_S3bjet1", "Jet1_mass_S3bjet1",
    "Jet2_pt_S3bjet1", "Jet2_eta_S3bjet1", "Jet2_phi_S3bjet1", "Jet2_mass_S3bjet1",
    "Jet3_pt_S3bjet1", "Jet3_eta_S3bjet1", "Jet3_phi_S3bjet1", "Jet3_mass_S3bjet1",
    "Jet4_pt_S3bjet1", "Jet4_eta_S3bjet1", "Jet4_phi_S3bjet1", "Jet4_mass_S3bjet1",
    "bJet1_pt_S3bjet1", "bJet1_eta_S3bjet1", "bJet1_phi_S3bjet1", "bJet1_mass_S3bjet1",

    "njet_S3bjet2", "nbjet_S3bjet2", "nmuon_S3bjet2", "nelectron_S3bjet2", "nlepton_S3bjet2", "MET_met_S3bjet2", "MET_eta_S3bjet2", "MET_phi_S3bjet2",
    "Jet1_pt_S3bjet2", "Jet1_eta_S3bjet2", "Jet1_phi_S3bjet2", "Jet1_mass_S3bjet2",
    "Jet2_pt_S3bjet2", "Jet2_eta_S3bjet2", "Jet2_phi_S3bjet2", "Jet2_mass_S3bjet2",
    "Jet3_pt_S3bjet2", "Jet3_eta_S3bjet2", "Jet3_phi_S3bjet2", "Jet3_mass_S3bjet2",
    "Jet4_pt_S3bjet2", "Jet4_eta_S3bjet2", "Jet4_phi_S3bjet2", "Jet4_mass_S3bjet2",
    "bJet1_pt_S3bjet2", "bJet1_eta_S3bjet2", "bJet1_phi_S3bjet2", "bJet1_mass_S3bjet2",
    "bJet2_pt_S3bjet2", "bJet2_eta_S3bjet2", "bJet2_phi_S3bjet2", "bJet2_mass_S3bjet2",

    "njet_S3bjet3", "nbjet_S3bjet3", "nmuon_S3bjet3", "nelectron_S3bjet3", "nlepton_S3bjet3", "MET_met_S3bjet3", "MET_eta_S3bjet3", "MET_phi_S3bjet3",
    "Jet1_pt_S3bjet3", "Jet1_eta_S3bjet3", "Jet1_phi_S3bjet3", "Jet1_mass_S3bjet3",
    "Jet2_pt_S3bjet3", "Jet2_eta_S3bjet3", "Jet2_phi_S3bjet3", "Jet2_mass_S3bjet3",
    "Jet3_pt_S3bjet3", "Jet3_eta_S3bjet3", "Jet3_phi_S3bjet3", "Jet3_mass_S3bjet3",
    "Jet4_pt_S3bjet3", "Jet4_eta_S3bjet3", "Jet4_phi_S3bjet3", "Jet4_mass_S3bjet3",
    "bJet1_pt_S3bjet3", "bJet1_eta_S3bjet3", "bJet1_phi_S3bjet3", "bJet1_mass_S3bjet3",
    "bJet2_pt_S3bjet3", "bJet2_eta_S3bjet3", "bJet2_phi_S3bjet3", "bJet2_mass_S3bjet3",
    "bJet3_pt_S3bjet3", "bJet3_eta_S3bjet3", "bJet3_phi_S3bjet3", "bJet3_mass_S3bjet3",

    "njet_S3bjet4", "nbjet_S3bjet4", "nmuon_S3bjet4", "nelectron_S3bjet4", "nlepton_S3bjet4", "MET_met_S3bjet4", "MET_eta_S3bjet4", "MET_phi_S3bjet4",
    "Jet1_pt_S3bjet4", "Jet1_eta_S3bjet4", "Jet1_phi_S3bjet4", "Jet1_mass_S3bjet4",
    "Jet2_pt_S3bjet4", "Jet2_eta_S3bjet4", "Jet2_phi_S3bjet4", "Jet2_mass_S3bjet4",
    "Jet3_pt_S3bjet4", "Jet3_eta_S3bjet4", "Jet3_phi_S3bjet4", "Jet3_mass_S3bjet4",
    "Jet4_pt_S3bjet4", "Jet4_eta_S3bjet4", "Jet4_phi_S3bjet4", "Jet4_mass_S3bjet4",
    "bJet1_pt_S3bjet4", "bJet1_eta_S3bjet4", "bJet1_phi_S3bjet4", "bJet1_mass_S3bjet4",
    "bJet2_pt_S3bjet4", "bJet2_eta_S3bjet4", "bJet2_phi_S3bjet4", "bJet2_mass_S3bjet4",
    "bJet3_pt_S3bjet4", "bJet3_eta_S3bjet4", "bJet3_phi_S3bjet4", "bJet3_mass_S3bjet4",
    "bJet4_pt_S3bjet4", "bJet4_eta_S3bjet4", "bJet4_phi_S3bjet4", "bJet4_mass_S3bjet4",
]

drawhisto(hists, {"tthbb_"+lep:tthbb, "ttbb_"+lep:ttbb}, "")
