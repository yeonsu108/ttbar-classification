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
outdir = indir+"/plots_"+lep+"/"
if not os.path.exists(outdir):
    os.makedirs(outdir)
tthbb  = ROOT.TFile(indir + "/hist/" + "hist_tthbb_"+lep+".root")
ttbb   = ROOT.TFile(indir + "/hist/" + "hist_ttbb_"+lep+".root")
colors = [807, 862] # kOrange: 800, kAzure: 860

def drawhisto(hists, files, flag="_S0"):
    canvas = ROOT.TCanvas("c", "c", 800, 600)
    for hname in hists:
        hist_dict = {}
        legend = ROOT.TLegend(0.75, 0.9, 0.9, 0.8)
        ymax, color = 0, 0
        for fname, f in files.items():
            h = f.Get(hname)
            if ymax < h.GetMaximum(): ymax = h.GetMaximum()
            h.GetXaxis().SetTitle(h.GetTitle())
            h.GetYaxis().SetTitle("Normalized entries")
            h.GetYaxis().SetTitleOffset(1.5)
            h.SetLineColor(colors[color])
            color+=1
            h.SetLineWidth(2)
            legend.AddEntry(h, fname.split("_")[0], "l")
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

if lep == "mu": lep = "Muon"
else: lep = "Electron"
## Histogram Features  
hists_lep = [
#    lep+"_pt_S1", lep+"_eta_S1", lep+"_phi_S1", lep+"_e_S1",
#    lep+"_pt_S2jet1", lep+"_eta_S2jet1", lep+"_phi_S2jet1", lep+"_e_S2jet1",
#    lep+"_pt_S2jet2", lep+"_eta_S2jet2", lep+"_phi_S2jet2", lep+"_e_S2jet2",
#    lep+"_pt_S2jet3", lep+"_eta_S2jet3", lep+"_phi_S2jet3", lep+"_e_S2jet3",
#    lep+"_pt_S2jet4", lep+"_eta_S2jet4", lep+"_phi_S2jet4", lep+"_e_S2jet4",
#    lep+"_pt_S2jet5", lep+"_eta_S2jet5", lep+"_phi_S2jet5", lep+"_e_S2jet5",
#    lep+"_pt_S2jet6", lep+"_eta_S2jet6", lep+"_phi_S2jet6", lep+"_e_S2jet6",
#    lep+"_pt_S3bjet1", lep+"_eta_S3bjet1", lep+"_phi_S3bjet1", lep+"_e_S3bjet1",
#    lep+"_pt_S3bjet2", lep+"_eta_S3bjet2", lep+"_phi_S3bjet2", lep+"_e_S3bjet2",
#    lep+"_pt_S3bjet3", lep+"_eta_S3bjet3", lep+"_phi_S3bjet3", lep+"_e_S3bjet3",
    lep+"_pt_S3bjet4", lep+"_eta_S3bjet4", lep+"_phi_S3bjet4", lep+"_e_S3bjet4",
]
hists = hists_lep+[
#    "njet_S0", "nbjet_S0", "nmuon_S0", "nelectron_S0", "nlepton_S0", "MET_met_S0", "MET_eta_S0", "MET_phi_S0",
#
#    "njet_S1", "nbjet_S1", "nmuon_S1", "nelectron_S1", "nlepton_S1", "MET_met_S1", "MET_eta_S1", "MET_phi_S1",
#
#    "njet_S2jet1", "nbjet_S2jet1", "nmuon_S2jet1", "nelectron_S2jet1", "nlepton_S2jet1", "MET_met_S2jet1", "MET_eta_S2jet1", "MET_phi_S2jet1",
#    "Jet1_pt_S2jet1", "Jet1_eta_S2jet1", "Jet1_phi_S2jet1", "Jet1_mass_S2jet1",
#
#    "njet_S2jet2", "nbjet_S2jet2", "nmuon_S2jet2", "nelectron_S2jet2", "nlepton_S2jet2", "MET_met_S2jet2", "MET_eta_S2jet2", "MET_phi_S2jet2",
#    "Jet1_pt_S2jet2", "Jet1_eta_S2jet2", "Jet1_phi_S2jet2", "Jet1_mass_S2jet2",
#    "Jet2_pt_S2jet2", "Jet2_eta_S2jet2", "Jet2_phi_S2jet2", "Jet2_mass_S2jet2",
#    "seljet1_idx_S2jet2", "seljet2_idx_S2jet2",
#    "mindR_jjPt_S2jet2", "mindR_jjEta_S2jet2", "mindR_jjPhi_S2jet2", "mindR_jjMass_S2jet2", "mindR_jjdR_S2jet2",
#    "chi2jet1_idx_S2jet2", "chi2jet2_idx_S2jet2",
#    "chi2_jjPt_S2jet2", "chi2_jjEta_S2jet2", "chi2_jjPhi_S2jet2", "chi2_jjMass_S2jet2", "chi2_jjdR_S2jet2",
#
#    "njet_S2jet3", "nbjet_S2jet3", "nmuon_S2jet3", "nelectron_S2jet3", "nlepton_S2jet3", "MET_met_S2jet3", "MET_eta_S2jet3", "MET_phi_S2jet3",
#    "Jet1_pt_S2jet3", "Jet1_eta_S2jet3", "Jet1_phi_S2jet3", "Jet1_mass_S2jet3",
#    "Jet2_pt_S2jet3", "Jet2_eta_S2jet3", "Jet2_phi_S2jet3", "Jet2_mass_S2jet3",
#    "Jet3_pt_S2jet3", "Jet3_eta_S2jet3", "Jet3_phi_S2jet3", "Jet3_mass_S2jet3",
#    "seljet1_idx_S2jet3", "seljet2_idx_S2jet3",
#    "mindR_jjPt_S2jet3", "mindR_jjEta_S2jet3", "mindR_jjPhi_S2jet3", "mindR_jjMass_S2jet3", "mindR_jjdR_S2jet3",
#    "chi2jet1_idx_S2jet3", "chi2jet2_idx_S2jet3",
#    "chi2_jjPt_S2jet3", "chi2_jjEta_S2jet3", "chi2_jjPhi_S2jet3", "chi2_jjMass_S2jet3", "chi2_jjdR_S2jet3",
#
#    "njet_S2jet4", "nbjet_S2jet4", "nmuon_S2jet4", "nelectron_S2jet4", "nlepton_S2jet4", "MET_met_S2jet4", "MET_eta_S2jet4", "MET_phi_S2jet4",
#    "Jet1_pt_S2jet4", "Jet1_eta_S2jet4", "Jet1_phi_S2jet4", "Jet1_mass_S2jet4",
#    "Jet2_pt_S2jet4", "Jet2_eta_S2jet4", "Jet2_phi_S2jet4", "Jet2_mass_S2jet4",
#    "Jet3_pt_S2jet4", "Jet3_eta_S2jet4", "Jet3_phi_S2jet4", "Jet3_mass_S2jet4",
#    "Jet4_pt_S2jet4", "Jet4_eta_S2jet4", "Jet4_phi_S2jet4", "Jet4_mass_S2jet4",
#    "seljet1_idx_S2jet4", "seljet2_idx_S2jet4",
#    "mindR_jjPt_S2jet4", "mindR_jjEta_S2jet4", "mindR_jjPhi_S2jet4", "mindR_jjMass_S2jet4", "mindR_jjdR_S2jet4",
#    "chi2jet1_idx_S2jet4", "chi2jet2_idx_S2jet4",
#    "chi2_jjPt_S2jet4", "chi2_jjEta_S2jet4", "chi2_jjPhi_S2jet4", "chi2_jjMass_S2jet4", "chi2_jjdR_S2jet4",
#
#    "njet_S2jet5", "nbjet_S2jet5", "nmuon_S2jet5", "nelectron_S2jet5", "nlepton_S2jet5", "MET_met_S2jet5", "MET_eta_S2jet5", "MET_phi_S2jet5",
#    "Jet1_pt_S2jet5", "Jet1_eta_S2jet5", "Jet1_phi_S2jet5", "Jet1_mass_S2jet5",
#    "Jet2_pt_S2jet5", "Jet2_eta_S2jet5", "Jet2_phi_S2jet5", "Jet2_mass_S2jet5",
#    "Jet3_pt_S2jet5", "Jet3_eta_S2jet5", "Jet3_phi_S2jet5", "Jet3_mass_S2jet5",
#    "Jet4_pt_S2jet5", "Jet4_eta_S2jet5", "Jet4_phi_S2jet5", "Jet4_mass_S2jet5",
#    "Jet5_pt_S2jet5", "Jet5_eta_S2jet5", "Jet5_phi_S2jet5", "Jet5_mass_S2jet5",
#    "nonbJet1_pt_S2jet5", "nonbJet1_eta_S2jet5", "nonbJet1_phi_S2jet5", "nonbJet1_mass_S2jet5",
#    "seljet1_idx_S2jet5", "seljet2_idx_S2jet5",
#    "mindR_jjPt_S2jet5", "mindR_jjEta_S2jet5", "mindR_jjPhi_S2jet5", "mindR_jjMass_S2jet5", "mindR_jjdR_S2jet5",
#    "chi2jet1_idx_S2jet5", "chi2jet2_idx_S2jet5",
#    "chi2_jjPt_S2jet5", "chi2_jjEta_S2jet5", "chi2_jjPhi_S2jet5", "chi2_jjMass_S2jet5", "chi2_jjdR_S2jet5",
#
#    "njet_S2jet6", "nbjet_S2jet6", "nmuon_S2jet6", "nelectron_S2jet6", "nlepton_S2jet6", "MET_met_S2jet6", "MET_eta_S2jet6", "MET_phi_S2jet6",
#    "Jet1_pt_S2jet6", "Jet1_eta_S2jet6", "Jet1_phi_S2jet6", "Jet1_mass_S2jet6",
#    "Jet2_pt_S2jet6", "Jet2_eta_S2jet6", "Jet2_phi_S2jet6", "Jet2_mass_S2jet6",
#    "Jet3_pt_S2jet6", "Jet3_eta_S2jet6", "Jet3_phi_S2jet6", "Jet3_mass_S2jet6",
#    "Jet4_pt_S2jet6", "Jet4_eta_S2jet6", "Jet4_phi_S2jet6", "Jet4_mass_S2jet6",
#    "Jet5_pt_S2jet6", "Jet5_eta_S2jet6", "Jet5_phi_S2jet6", "Jet5_mass_S2jet6",
#    "Jet6_pt_S2jet6", "Jet6_eta_S2jet6", "Jet6_phi_S2jet6", "Jet6_mass_S2jet6",
#    "nonbJet1_pt_S2jet6", "nonbJet1_eta_S2jet6", "nonbJet1_phi_S2jet6", "nonbJet1_mass_S2jet6",
#    "nonbJet2_pt_S2jet6", "nonbJet2_eta_S2jet6", "nonbJet2_phi_S2jet6", "nonbJet2_mass_S2jet6",
#    "seljet1_idx_S2jet6", "seljet2_idx_S2jet6",
#    "mindR_jjPt_S2jet6", "mindR_jjEta_S2jet6", "mindR_jjPhi_S2jet6", "mindR_jjMass_S2jet6", "mindR_jjdR_S2jet6",
#    "selnonbjet1_idx_S2jet6", "selnonbjet2_idx_S2jet6",
#    "mindR_nnPt_S2jet6", "mindR_nnEta_S2jet6", "mindR_nnPhi_S2jet6", "mindR_nnMass_S2jet6", "mindR_nndR_S2jet6",
#    "chi2jet1_idx_S2jet6", "chi2jet2_idx_S2jet6",
#    "chi2_jjPt_S2jet6", "chi2_jjEta_S2jet6", "chi2_jjPhi_S2jet6", "chi2_jjMass_S2jet6", "chi2_jjdR_S2jet6",
#
#    "njet_S3bjet1", "nbjet_S3bjet1", "nmuon_S3bjet1", "nelectron_S3bjet1", "nlepton_S3bjet1", "MET_met_S3bjet1", "MET_eta_S3bjet1", "MET_phi_S3bjet1",
#    "Jet1_pt_S3bjet1", "Jet1_eta_S3bjet1", "Jet1_phi_S3bjet1", "Jet1_mass_S3bjet1",
#    "Jet2_pt_S3bjet1", "Jet2_eta_S3bjet1", "Jet2_phi_S3bjet1", "Jet2_mass_S3bjet1",
#    "Jet3_pt_S3bjet1", "Jet3_eta_S3bjet1", "Jet3_phi_S3bjet1", "Jet3_mass_S3bjet1",
#    "Jet4_pt_S3bjet1", "Jet4_eta_S3bjet1", "Jet4_phi_S3bjet1", "Jet4_mass_S3bjet1",
#    "Jet5_pt_S3bjet1", "Jet5_eta_S3bjet1", "Jet5_phi_S3bjet1", "Jet5_mass_S3bjet1",
#    "Jet6_pt_S3bjet1", "Jet6_eta_S3bjet1", "Jet6_phi_S3bjet1", "Jet6_mass_S3bjet1",
#    "nonbJet1_pt_S3bjet1", "nonbJet1_eta_S3bjet1", "nonbJet1_phi_S3bjet1", "nonbJet1_mass_S3bjet1",
#    "nonbJet2_pt_S3bjet1", "nonbJet2_eta_S3bjet1", "nonbJet2_phi_S3bjet1", "nonbJet2_mass_S3bjet1",
#    "bJet1_pt_S3bjet1", "bJet1_eta_S3bjet1", "bJet1_phi_S3bjet1", "bJet1_mass_S3bjet1",
#    "seljet1_idx_S3bjet1", "seljet2_idx_S3bjet1",
#    "mindR_jjPt_S3bjet1", "mindR_jjEta_S3bjet1", "mindR_jjPhi_S3bjet1", "mindR_jjMass_S3bjet1", "mindR_jjdR_S3bjet1",
#    "selnonbjet1_idx_S3bjet1", "selnonbjet2_idx_S3bjet1",
#    "mindR_nnPt_S3bjet1", "mindR_nnEta_S3bjet1", "mindR_nnPhi_S3bjet1", "mindR_nnMass_S3bjet1", "mindR_nndR_S3bjet1",
#    "chi2jet1_idx_S3bjet1", "chi2jet2_idx_S3bjet1",
#    "chi2_jjPt_S3bjet1", "chi2_jjEta_S3bjet1", "chi2_jjPhi_S3bjet1", "chi2_jjMass_S3bjet1", "chi2_jjdR_S3bjet1",
#
#    "njet_S3bjet2", "nbjet_S3bjet2", "nmuon_S3bjet2", "nelectron_S3bjet2", "nlepton_S3bjet2", "MET_met_S3bjet2", "MET_eta_S3bjet2", "MET_phi_S3bjet2",
#    "Jet1_pt_S3bjet2", "Jet1_eta_S3bjet2", "Jet1_phi_S3bjet2", "Jet1_mass_S3bjet2",
#    "Jet2_pt_S3bjet2", "Jet2_eta_S3bjet2", "Jet2_phi_S3bjet2", "Jet2_mass_S3bjet2",
#    "Jet3_pt_S3bjet2", "Jet3_eta_S3bjet2", "Jet3_phi_S3bjet2", "Jet3_mass_S3bjet2",
#    "Jet4_pt_S3bjet2", "Jet4_eta_S3bjet2", "Jet4_phi_S3bjet2", "Jet4_mass_S3bjet2",
#    "Jet5_pt_S3bjet2", "Jet5_eta_S3bjet2", "Jet5_phi_S3bjet2", "Jet5_mass_S3bjet2",
#    "Jet6_pt_S3bjet2", "Jet6_eta_S3bjet2", "Jet6_phi_S3bjet2", "Jet6_mass_S3bjet2",
#    "nonbJet1_pt_S3bjet2", "nonbJet1_eta_S3bjet2", "nonbJet1_phi_S3bjet2", "nonbJet1_mass_S3bjet2",
#    "nonbJet2_pt_S3bjet2", "nonbJet2_eta_S3bjet2", "nonbJet2_phi_S3bjet2", "nonbJet2_mass_S3bjet2",
#    "bJet1_pt_S3bjet2", "bJet1_eta_S3bjet2", "bJet1_phi_S3bjet2", "bJet1_mass_S3bjet2",
#    "bJet2_pt_S3bjet2", "bJet2_eta_S3bjet2", "bJet2_phi_S3bjet2", "bJet2_mass_S3bjet2",
#    "seljet1_idx_S3bjet2", "seljet2_idx_S3bjet2",
#    "mindR_jjPt_S3bjet2", "mindR_jjEta_S3bjet2", "mindR_jjPhi_S3bjet2", "mindR_jjMass_S3bjet2", "mindR_jjdR_S3bjet2",
#    "selnonbjet1_idx_S3bjet2", "selnonbjet2_idx_S3bjet2",
#    "mindR_nnPt_S3bjet2", "mindR_nnEta_S3bjet2", "mindR_nnPhi_S3bjet2", "mindR_nnMass_S3bjet2", "mindR_nndR_S3bjet2",
#    "selbjet1_idx_S3bjet2", "selbjet2_idx_S3bjet2",
#    "mindR_bbPt_S3bjet2", "mindR_bbEta_S3bjet2", "mindR_bbPhi_S3bjet2", "mindR_bbMass_S3bjet2", "mindR_bbdR_S3bjet2",
#    "chi2jet1_idx_S3bjet2", "chi2jet2_idx_S3bjet2",
#    "chi2_jjPt_S3bjet2", "chi2_jjEta_S3bjet2", "chi2_jjPhi_S3bjet2", "chi2_jjMass_S3bjet2", "chi2_jjdR_S3bjet2",
#    "chi2bjet1_idx_S3bjet2", "chi2bjet2_idx_S3bjet2",
#    "chi2_bbPt_S3bjet2", "chi2_bbEta_S3bjet2", "chi2_bbPhi_S3bjet2", "chi2_bbMass_S3bjet2", "chi2_bbdR_S3bjet2",
#
#    "njet_S3bjet3", "nbjet_S3bjet3", "nmuon_S3bjet3", "nelectron_S3bjet3", "nlepton_S3bjet3", "MET_met_S3bjet3", "MET_eta_S3bjet3", "MET_phi_S3bjet3",
#    "Jet1_pt_S3bjet3", "Jet1_eta_S3bjet3", "Jet1_phi_S3bjet3", "Jet1_mass_S3bjet3",
#    "Jet2_pt_S3bjet3", "Jet2_eta_S3bjet3", "Jet2_phi_S3bjet3", "Jet2_mass_S3bjet3",
#    "Jet3_pt_S3bjet3", "Jet3_eta_S3bjet3", "Jet3_phi_S3bjet3", "Jet3_mass_S3bjet3",
#    "Jet4_pt_S3bjet3", "Jet4_eta_S3bjet3", "Jet4_phi_S3bjet3", "Jet4_mass_S3bjet3",
#    "Jet5_pt_S3bjet3", "Jet5_eta_S3bjet3", "Jet5_phi_S3bjet3", "Jet5_mass_S3bjet3",
#    "Jet6_pt_S3bjet3", "Jet6_eta_S3bjet3", "Jet6_phi_S3bjet3", "Jet6_mass_S3bjet3",
#    "nonbJet1_pt_S3bjet3", "nonbJet1_eta_S3bjet3", "nonbJet1_phi_S3bjet3", "nonbJet1_mass_S3bjet3",
#    "nonbJet2_pt_S3bjet3", "nonbJet2_eta_S3bjet3", "nonbJet2_phi_S3bjet3", "nonbJet2_mass_S3bjet3",
#    "bJet1_pt_S3bjet3", "bJet1_eta_S3bjet3", "bJet1_phi_S3bjet3", "bJet1_mass_S3bjet3",
#    "bJet2_pt_S3bjet3", "bJet2_eta_S3bjet3", "bJet2_phi_S3bjet3", "bJet2_mass_S3bjet3",
#    "bJet3_pt_S3bjet3", "bJet3_eta_S3bjet3", "bJet3_phi_S3bjet3", "bJet3_mass_S3bjet3",
#    "seljet1_idx_S3bjet3", "seljet2_idx_S3bjet3",
#    "mindR_jjPt_S3bjet3", "mindR_jjEta_S3bjet3", "mindR_jjPhi_S3bjet3", "mindR_jjMass_S3bjet3", "mindR_jjdR_S3bjet3",
#    "selnonbjet1_idx_S3bjet3", "selnonbjet2_idx_S3bjet3",
#    "mindR_nnPt_S3bjet3", "mindR_nnEta_S3bjet3", "mindR_nnPhi_S3bjet3", "mindR_nnMass_S3bjet3", "mindR_nndR_S3bjet3",
#    "selbjet1_idx_S3bjet3", "selbjet2_idx_S3bjet3",
#    "mindR_bbPt_S3bjet3", "mindR_bbEta_S3bjet3", "mindR_bbPhi_S3bjet3", "mindR_bbMass_S3bjet3", "mindR_bbdR_S3bjet3",
#    "chi2jet1_idx_S3bjet3", "chi2jet2_idx_S3bjet3",
#    "chi2_jjPt_S3bjet3", "chi2_jjEta_S3bjet3", "chi2_jjPhi_S3bjet3", "chi2_jjMass_S3bjet3", "chi2_jjdR_S3bjet3",
#    "chi2bjet1_idx_S3bjet3", "chi2bjet2_idx_S3bjet3",
#    "chi2_bbPt_S3bjet3", "chi2_bbEta_S3bjet3", "chi2_bbPhi_S3bjet3", "chi2_bbMass_S3bjet3", "chi2_bbdR_S3bjet3",

    "njet_S3bjet4", "nbjet_S3bjet4", "nmuon_S3bjet4", "nelectron_S3bjet4", "nlepton_S3bjet4", "MET_met_S3bjet4", "MET_eta_S3bjet4", "MET_phi_S3bjet4",
    "Jet1_pt_S3bjet4", "Jet1_eta_S3bjet4", "Jet1_phi_S3bjet4", "Jet1_mass_S3bjet4",
    "Jet2_pt_S3bjet4", "Jet2_eta_S3bjet4", "Jet2_phi_S3bjet4", "Jet2_mass_S3bjet4",
    "Jet3_pt_S3bjet4", "Jet3_eta_S3bjet4", "Jet3_phi_S3bjet4", "Jet3_mass_S3bjet4",
    "Jet4_pt_S3bjet4", "Jet4_eta_S3bjet4", "Jet4_phi_S3bjet4", "Jet4_mass_S3bjet4",
    "Jet5_pt_S3bjet4", "Jet5_eta_S3bjet4", "Jet5_phi_S3bjet4", "Jet5_mass_S3bjet4",
    "Jet6_pt_S3bjet4", "Jet6_eta_S3bjet4", "Jet6_phi_S3bjet4", "Jet6_mass_S3bjet4",
    "nonbJet1_pt_S3bjet4", "nonbJet1_eta_S3bjet4", "nonbJet1_phi_S3bjet4", "nonbJet1_mass_S3bjet4",
    "nonbJet2_pt_S3bjet4", "nonbJet2_eta_S3bjet4", "nonbJet2_phi_S3bjet4", "nonbJet2_mass_S3bjet4",
    "bJet1_pt_S3bjet4", "bJet1_eta_S3bjet4", "bJet1_phi_S3bjet4", "bJet1_mass_S3bjet4",
    "bJet2_pt_S3bjet4", "bJet2_eta_S3bjet4", "bJet2_phi_S3bjet4", "bJet2_mass_S3bjet4",
    "bJet3_pt_S3bjet4", "bJet3_eta_S3bjet4", "bJet3_phi_S3bjet4", "bJet3_mass_S3bjet4",
    "bJet4_pt_S3bjet4", "bJet4_eta_S3bjet4", "bJet4_phi_S3bjet4", "bJet4_mass_S3bjet4",
    "seljet1_idx_S3bjet4", "seljet2_idx_S3bjet4",
    "mindR_jjPt_S3bjet4", "mindR_jjEta_S3bjet4", "mindR_jjPhi_S3bjet4", "mindR_jjMass_S3bjet4", "mindR_jjdR_S3bjet4",
    "selnonbjet1_idx_S3bjet4", "selnonbjet2_idx_S3bjet4",
    "mindR_nnPt_S3bjet4", "mindR_nnEta_S3bjet4", "mindR_nnPhi_S3bjet4", "mindR_nnMass_S3bjet4", "mindR_nndR_S3bjet4",
    "selbjet1_idx_S3bjet4", "selbjet2_idx_S3bjet4",
    "mindR_bbPt_S3bjet4", "mindR_bbEta_S3bjet4", "mindR_bbPhi_S3bjet4", "mindR_bbMass_S3bjet4", "mindR_bbdR_S3bjet4",
    "chi2jet1_idx_S3bjet4", "chi2jet2_idx_S3bjet4",
    "chi2_jjPt_S3bjet4", "chi2_jjEta_S3bjet4", "chi2_jjPhi_S3bjet4", "chi2_jjMass_S3bjet4", "chi2_jjdR_S3bjet4",
    "chi2bjet1_idx_S3bjet4", "chi2bjet2_idx_S3bjet4",
    "chi2_bbPt_S3bjet4", "chi2_bbEta_S3bjet4", "chi2_bbPhi_S3bjet4", "chi2_bbMass_S3bjet4", "chi2_bbdR_S3bjet4",
]

drawhisto(hists, {"tthbb_"+lep:tthbb, "ttbb_"+lep:ttbb}, "")
