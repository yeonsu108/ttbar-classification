
//how to use: root -l -b -q 'ana_ttbar_df.C("/filepath/infile.root", "./filepath/outfile.root")'

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#endif
#include <vector>
#include <algorithm>

#include "TTree.h"
#include "TFile.h"

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"

#include "utility.h"

void ana_ttbar_df(std::string infile, std::string outfile="./processed/out.root"){
    gSystem->Load("libDelphes");

    std::cout << infile << std::endl;
    std::cout << outfile << std::endl;
    auto treename = "Delphes";

    auto _df = ROOT::RDataFrame(treename, infile);


    //GenLevel selection
    auto df1 = _df.Define("GenAddQuark_bi", ::SelectAddQuark, {"Particle.PID", "Particle.M1", "Particle.M2", "Particle.D1", "Particle.D2"})
                  .Define("GenAddbQuark_bi", "abs(Particle.PID) == 5 && GenAddQuark_bi == 1")
                  
                  .Define("GenAddJet_bi", ::dRMatching, {"GenAddQuark_bi", "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.Mass", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass"})
                  .Define("GenAddbJet_bi", ::dRMatching, {"GenAddbQuark_bi", "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.Mass", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass"})

                  .Define("nGenAddQuark", "Sum(GenAddQuark_bi)")
                  .Define("nGenAddbQuark", "Sum(GenAddbQuark_bi)")
                  .Define("nGenAddJet", "Sum(GenAddJet_bi)")
                  .Define("nGenAddbJet", "Sum(GenAddbJet_bi)")

                  .Filter("nGenAddbJet >= 2");
                  
    //object selection
    auto df2 = df1.Define("goodJet", "Jet.PT>0 && abs(Jet.Eta)<2.8")
                  .Define("goodElectron", "Electron.PT>0 && abs(Electron.Eta)<2.8")
                  .Define("goodMuon", "Muon.PT>0 && abs(Muon.Eta)<2.8")

                  .Define("Jet_pt", "Jet.PT[goodJet]")
                  .Define("Jet_eta", "Jet.Eta[goodJet]")
                  .Define("Jet_phi", "Jet.Phi[goodJet]")
                  .Define("Jet_mass", "Jet.Mass[goodJet]")
                  .Define("Jet_btag", "Jet.BTag[goodJet]")
                  .Define("njet", "Sum(goodJet)")

                  .Define("bJet_pt", "Jet_pt[Jet_btag]")
                  .Define("bJet_eta", "Jet_eta[Jet_btag]")
                  .Define("bJet_phi", "Jet_phi[Jet_btag]")
                  .Define("bJet_mass", "Jet_mass[Jet_btag]")
                  .Define("nbjet", "Sum(Jet_btag)")

                  .Define("Muon_pt", "Muon.PT[goodMuon]")
                  .Define("Muon_eta", "Muon.Eta[goodMuon]")
                  .Define("Muon_phi", "Muon.Phi[goodMuon]")
                  .Define("Muon_e", ::GetE, {"Muon_pt", "Muon_eta", "Muon_phi"})
                  .Define("nmuon", "Sum(goodMuon)")

                  .Define("Electron_pt", "Electron.PT[goodElectron]")
                  .Define("Electron_eta", "Electron.Eta[goodElectron]")
                  .Define("Electron_phi", "Electron.Phi[goodElectron]")
                  .Define("Electron_e", ::GetE, {"Electron_pt", "Electron_eta", "Electron_phi"})
                  .Define("nelectron", "Sum(goodElectron)")
                  .Define("nlepton", "Muon_size+Electron_size")
                  
                  .Define("MET_met", "MissingET.MET")
                  .Define("MET_eta", "MissingET.Eta")
                  .Define("MET_phi", "MissingET.Phi");

    auto df = df2;

    std::initializer_list<std::string> variables = {//"Event", "Jet", "Muon", "Electron",
                      "nGenAddQuark", "nGenAddbQuark", "nGenAddJet", "nGenAddbJet",
                      //"GenAddQuark_bi", "GenAddbQuark_bi", "GenAddJet_bi", "GenAddbJet_bi",

                      "goodJet", "goodElectron", "goodMuon",
                      "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_btag", "njet",
                      "bJet_pt", "bJet_eta", "bJet_phi", "bJet_mass", "nbjet",
                      "Muon_pt", "Muon_eta", "Muon_phi", "Muon_e", "nmuon",
                      "Electron_pt", "Electron_eta", "Electron_phi", "Electron_e", "nelectron",
                      "nlepton",

                      "MET_met", "MET_eta", "MET_phi",
    };
    df.Snapshot(treename, outfile, variables);
    std::cout << "done" << std::endl;

    //df.Snapshot<TClonesArray, TClonesArray, TClonesArray, TClonesArray>("outputTree", "out.root", variables, ROOT::RDF::RSnapshotOptions("RECreate", ROOT::kZLIB, 1, 0, 99, false));
    //df.Snapshot<TClonesArray, TClonesArray, TClonesArray, TClonesArray>("outputTree", "out.root", {"Event", "Electron", "Muon", "Jet"}, ROOT::RDF::RSnapshotOptions("RECreate", ROOT::kZLIB, 1, 0, 99, false));
}
