import ROOT
import numpy as np
import os,sys
ROOT.gStyle.SetOptStat(0)

# RDF
TreeName = "dnn_input"

# Modify!! 
indir = sys.argv[1]
outdir = indir+"plots/"
if not os.path.exists(outdir): os.makedirs(outdir)
tthbb  = ROOT.RDataFrame(TreeName, indir + "tthbb*")
ttbb   = ROOT.RDataFrame(TreeName, indir + "ttbb*")
ttcc   = ROOT.RDataFrame(TreeName, indir + "ttcc*")
ttjj   = ROOT.RDataFrame(TreeName, indir + "ttjj*")
dfs = {"tthbb": tthbb, "ttbb": ttbb, "ttcc": ttcc, "ttjj": ttjj}

def drawHisto(hists, dfs, flag="_S0"):
    canvas = ROOT.TCanvas("c", "c", 400, 400)
    for hist_name in hists:
        hist_dict = {}
        legend = ROOT.TLegend(0.75, 0.9, 0.9, 0.80)
        hist_title = hist_name.replace("_", " ")

        ymax, color = 0, 1
        for df_name, df in dfs.items():
            nbin, xmin, xmax = 20, 0, 200
            print (hist_name)
            _xmax = df.Max(hist_name).GetValue()
            if _xmax < 100: xmax = 100
            if _xmax < 20: nbin, xmax = int(_xmax+2), int(_xmax+2)
            if _xmax < 5: nbin, xmin, xmax = 20, -4, 4
            h = df.Histo1D(ROOT.RDF.TH1DModel(hist_name, hist_title, nbin, xmin, xmax), hist_name)
            if ymax < h.GetMaximum(): ymax = h.GetMaximum()
            h.GetXaxis().SetTitle(hist_title)
            h.GetYaxis().SetTitle("Normalized entries")
            h.GetYaxis().SetTitleOffset(1.5)
            h.SetLineColor(color)
            color+=1
            h.SetLineWidth(2)
            legend.AddEntry(h.GetValue(), df_name, "l")
            hist_dict[hist_name + "_" + df_name] = h

        first = True
        for _tmp, h in hist_dict.items():
            h.SetMaximum(ymax * 1.4)
            if first:
                h.DrawNormalized("hist")
                #h.Draw("hist")
                first = False
            else:
                h.DrawNormalized("same")
                #h.Draw("same")
        legend.Draw()
        canvas.Print(outdir + hist_name + flag + ".pdf")
        canvas.Clear()

## Histogram Features  
hists1 = [
    'njets', 'nbjets', 'nElectron', 'nMuon', 'nLepton',
    'bjet1_pt', 'bjet1_eta', 'bjet1_phi', 'bjet1_e', 'bjet2_pt', 'bjet2_eta', 'bjet2_phi', 'bjet2_e',
    'Jet_pt1', 'Jet_eta1', 'Jet_phi1', 'Jet_e1', 'Jet_pt2', 'Jet_eta2', 'Jet_phi2', 'Jet_e2',
    'Jet_pt3', 'Jet_eta3', 'Jet_phi3', 'Jet_e3', 'Jet_pt4', 'Jet_eta4', 'Jet_phi4', 'Jet_e4',
    'Lepton_pt', 'Lepton_eta', 'Lepton_phi', 'Lepton_e', 'MET_px', 'MET_py', 'MET_met',
    'selbjet1_pt', 'selbjet1_eta', 'selbjet1_phi', 'selbjet1_e', 'selbjet2_pt', 'selbjet2_eta', 'selbjet2_phi', 'selbjet2_e',

    'bbdR',   'bbdEta',   'bbdPhi',   'bbPt',   'bbEta',   'bbPhi',   'bbMass',   'bbHt',   'bbMt',
    'nub1dR', 'nub1dEta', 'nub1dPhi', 'nub1Pt', 'nub1Eta', 'nub1Phi', 'nub1Mass', 'nub1Ht', 'nub1Mt',
    'nub2dR', 'nub2dEta', 'nub2dPhi', 'nub2Pt', 'nub2Eta', 'nub2Phi', 'nub2Mass', 'nub2Ht', 'nub2Mt',
    'nubbdR', 'nubbdEta', 'nubbdPhi', 'nubbPt', 'nubbEta', 'nubbPhi', 'nubbMass', 'nubbHt', 'nubbMt',
    'lb1dR',  'lb1dEta',  'lb1dPhi',  'lb1Pt',  'lb1Eta',  'lb1Phi',  'lb1Mass',  'lb1Ht',  'lb1Mt',
    'lb2dR',  'lb2dEta',  'lb2dPhi',  'lb2Pt',  'lb2Eta',  'lb2Phi',  'lb2Mass',  'lb2Ht',  'lb2Mt',
    'lbbdR',  'lbbdEta',  'lbbdPhi',  'lbbPt',  'lbbEta',  'lbbPhi',  'lbbMass',  'lbbHt',  'lbbMt',
    'Wjb1dR', 'Wjb1dEta', 'Wjb1dPhi', 'Wjb1Pt', 'Wjb1Eta', 'Wjb1Phi', 'Wjb1Mass', 'Wjb1Ht', 'Wjb1Mt',
    'Wjb2dR', 'Wjb2dEta', 'Wjb2dPhi', 'Wjb2Pt', 'Wjb2Eta', 'Wjb2Phi', 'Wjb2Mass', 'Wjb2Ht', 'Wjb2Mt',
    'Wlb1dR', 'Wlb1dEta', 'Wlb1dPhi', 'Wlb1Pt', 'Wlb1Eta', 'Wlb1Phi', 'Wlb1Mass', 'Wlb1Ht', 'Wlb1Mt',
    'Wlb2dR', 'Wlb2dEta', 'Wlb2dPhi', 'Wlb2Pt', 'Wlb2Eta', 'Wlb2Phi', 'Wlb2Mass', 'Wlb2Ht', 'Wlb2Mt',
]

drawHisto(hists1, dfs, "")


