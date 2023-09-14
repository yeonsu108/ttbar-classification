
// how to use: root -l -b -q 'genlevelstudy.C("tthbb", "./outputdir/")'

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

void genlevel_tthbb(std::string channel, std::string infile, std::string outdir="./genlevel/"){
    gSystem->Load("libDelphes");

    std::cout << infile << std::endl;
    std::cout << outdir << std::endl;
    auto treename = "Delphes";

    auto _df = ROOT::RDataFrame(treename, infile);


    //Define PID and some useful variables
    auto df1 = _df.Define("T_pid", "int(6)").Define("H_pid", "int(25)").Define("g_pid", "int(21)").Define("W_pid", "int(24)")
                  .Define("b_pid", "int(5)").Define("c_pid", "int(4)").Define("Elec_pid", "int(11)").Define("Mu_pid", "int(13)")
                  .Define("int0", "int(0)").Define("int1", "int(1)").Define("float0", "float(0)")
                  .Define("drmax1", "float(0.15)").Define("drmax2", "float(0.4)");


    //Check if the sample has been well-crafted
    //for tthbb, find final particles
    auto df2 = df1.Define("isLast", ::isLast, {"Particle.PID", "Particle.D1", "Particle.D2"})
                  .Define("Top", "abs(Particle.PID) == 6 && isLast")
                  .Define("nTop", "Sum(Top)")
                  .Define("W", "abs(Particle.PID) == 24 && isLast")
                  .Define("nW", "Sum(W)")
                  .Define("Higgs", "abs(Particle.PID) == 25 && isLast")
                  .Define("nHiggs", "Sum(Higgs)")
                  .Define("GenbQuark", "abs(Particle.PID) == 5 && isLast")
                  .Define("nGenbQuark", "Sum(GenbQuark)")
                  .Define("FinalGenPart_idx", ::FinalGenPart_idx, {"Particle.PID", "Particle.M1", "Particle.M2", "Particle.D1", "Particle.D2", "Top", "Higgs"})
                  .Define("GenPart_bFromTop1_idx","FinalGenPart_idx[0]")
                  .Define("GenPart_lepFromTop1_idx","FinalGenPart_idx[1]")
                  .Define("GenPart_bFromTop2_idx","FinalGenPart_idx[2]")
                  .Define("GenPart_q1FromTop2_idx","FinalGenPart_idx[3]")
                  .Define("GenPart_q2FromTop2_idx","FinalGenPart_idx[4]")
                  .Define("GenPart_b1FromHiggs_idx","FinalGenPart_idx[5]")
                  .Define("GenPart_b2FromHiggs_idx","FinalGenPart_idx[6]")
                  .Define("GenPart_LepTop_idx","FinalGenPart_idx[7]")
                  .Define("GenPart_HadTop_idx","FinalGenPart_idx[8]")

                  .Define("GenbFromTop1_idx", ::dRMatching_idx, {"GenPart_bFromTop1_idx", "drmax2", "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.Mass", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass"})
                  .Define("GenbFromTop2_idx", ::dRMatching_idx, {"GenPart_bFromTop2_idx", "drmax2", "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.Mass", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass"})
                  .Define("Genq1FromTop2_idx", ::dRMatching_idx, {"GenPart_q1FromTop2_idx", "drmax2", "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.Mass", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass"})
                  .Define("Genq2FromTop2_idx", ::dRMatching_idx, {"GenPart_q2FromTop2_idx", "drmax2", "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.Mass", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass"})
                  .Define("Genb1FromHiggs_idx", ::dRMatching_idx, {"GenPart_b1FromHiggs_idx", "drmax2", "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.Mass", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass"})
                  .Define("Genb2FromHiggs_idx", ::dRMatching_idx, {"GenPart_b2FromHiggs_idx", "drmax2", "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.Mass", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass"})

                  .Define("bFromTop1_idx", ::dRMatching_idx, {"GenbFromTop1_idx", "drmax2", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass", "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass"})
                  .Define("bFromTop2_idx", ::dRMatching_idx, {"GenbFromTop2_idx", "drmax2", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass", "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass"})
                  .Define("q1FromTop2_idx", ::dRMatching_idx, {"Genq1FromTop2_idx", "drmax2", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass", "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass"})
                  .Define("q2FromTop2_idx", ::dRMatching_idx, {"Genq2FromTop2_idx", "drmax2", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass", "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass"})
                  .Define("b1FromHiggs_idx", ::dRMatching_idx, {"Genb1FromHiggs_idx", "drmax2", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass", "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass"})
                  .Define("b2FromHiggs_idx", ::dRMatching_idx, {"Genb2FromHiggs_idx", "drmax2", "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass", "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass"})


                  .Define("muonFromTop1_idx", ::dRMatching_idx, {"GenPart_lepFromTop1_idx", "drmax1", "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.Mass", "Muon.PT", "Muon.Eta", "Muon.Phi", "Muon.PT"})
                  .Define("elecFromTop1_idx", ::dRMatching_idx, {"GenPart_lepFromTop1_idx", "drmax1", "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.Mass", "Electron.PT", "Electron.Eta", "Electron.Phi", "Muon.PT"});


    //Define 4Vecs for final particles
    auto df3 = df2.Filter("GenPart_bFromTop1_idx>=0 && GenPart_lepFromTop1_idx>=0 && GenPart_bFromTop2_idx>=0 && GenPart_q1FromTop2_idx>=0 && GenPart_q2FromTop2_idx>=0 && GenPart_b1FromHiggs_idx>=0 && GenPart_b2FromHiggs_idx>=0")
                  .Define("GenPart_bFromTop1_pt",  "Particle.PT[GenPart_bFromTop1_idx]")
                  .Define("GenPart_lepFromTop1_pt","Particle.PT[GenPart_lepFromTop1_idx]")
                  .Define("GenPart_bFromTop2_pt",  "Particle.PT[GenPart_bFromTop2_idx]")
                  .Define("GenPart_q1FromTop2_pt", "Particle.PT[GenPart_q1FromTop2_idx]")
                  .Define("GenPart_q2FromTop2_pt", "Particle.PT[GenPart_q2FromTop2_idx]")
                  .Define("GenPart_b1FromHiggs_pt","Particle.PT[GenPart_b1FromHiggs_idx]")
                  .Define("GenPart_b2FromHiggs_pt","Particle.PT[GenPart_b2FromHiggs_idx]")
                  
                  .Define("GenPart_bFromTop1_eta",  "Particle.Eta[GenPart_bFromTop1_idx]")
                  .Define("GenPart_lepFromTop1_eta","Particle.Eta[GenPart_lepFromTop1_idx]")
                  .Define("GenPart_bFromTop2_eta",  "Particle.Eta[GenPart_bFromTop2_idx]")
                  .Define("GenPart_q1FromTop2_eta", "Particle.Eta[GenPart_q1FromTop2_idx]")
                  .Define("GenPart_q2FromTop2_eta", "Particle.Eta[GenPart_q2FromTop2_idx]")
                  .Define("GenPart_b1FromHiggs_eta","Particle.Eta[GenPart_b1FromHiggs_idx]")
                  .Define("GenPart_b2FromHiggs_eta","Particle.Eta[GenPart_b2FromHiggs_idx]")

                  .Define("GenPart_bFromTop1_phi",  "Particle.Phi[GenPart_bFromTop1_idx]")
                  .Define("GenPart_lepFromTop1_phi","Particle.Phi[GenPart_lepFromTop1_idx]")
                  .Define("GenPart_bFromTop2_phi",  "Particle.Phi[GenPart_bFromTop2_idx]")
                  .Define("GenPart_q1FromTop2_phi", "Particle.Phi[GenPart_q1FromTop2_idx]")
                  .Define("GenPart_q2FromTop2_phi", "Particle.Phi[GenPart_q2FromTop2_idx]")
                  .Define("GenPart_b1FromHiggs_phi","Particle.Phi[GenPart_b1FromHiggs_idx]")
                  .Define("GenPart_b2FromHiggs_phi","Particle.Phi[GenPart_b2FromHiggs_idx]")

                  .Define("GenPart_bFromTop1_mass",  "Particle.Mass[GenPart_bFromTop1_idx]")
                  .Define("GenPart_lepFromTop1_mass","Particle.Mass[GenPart_lepFromTop1_idx]")
                  .Define("GenPart_bFromTop2_mass",  "Particle.Mass[GenPart_bFromTop2_idx]")
                  .Define("GenPart_q1FromTop2_mass", "Particle.Mass[GenPart_q1FromTop2_idx]")
                  .Define("GenPart_q2FromTop2_mass", "Particle.Mass[GenPart_q2FromTop2_idx]")
                  .Define("GenPart_b1FromHiggs_mass","Particle.Mass[GenPart_b1FromHiggs_idx]")
                  .Define("GenPart_b2FromHiggs_mass","Particle.Mass[GenPart_b2FromHiggs_idx]");
                  

    //Define 4Vecs for final GenJets
    auto df4 = df3.Filter("GenbFromTop1_idx>=0 && GenbFromTop2_idx>=0 && Genq1FromTop2_idx>=0 && Genq2FromTop2_idx>=0 && Genb1FromHiggs_idx>=0 && Genb2FromHiggs_idx>=0")
                  .Define("GenbFromTop1_pt",  "GenJet.PT[GenbFromTop1_idx]")
                  .Define("GenbFromTop2_pt",  "GenJet.PT[GenbFromTop2_idx]")
                  .Define("Genq1FromTop2_pt", "GenJet.PT[Genq1FromTop2_idx]")
                  .Define("Genq2FromTop2_pt", "GenJet.PT[Genq2FromTop2_idx]")
                  .Define("Genb1FromHiggs_pt","GenJet.PT[Genb1FromHiggs_idx]")
                  .Define("Genb2FromHiggs_pt","GenJet.PT[Genb2FromHiggs_idx]")
                  
                  .Define("GenbFromTop1_eta",  "GenJet.Eta[GenbFromTop1_idx]")
                  .Define("GenbFromTop2_eta",  "GenJet.Eta[GenbFromTop2_idx]")
                  .Define("Genq1FromTop2_eta", "GenJet.Eta[Genq1FromTop2_idx]")
                  .Define("Genq2FromTop2_eta", "GenJet.Eta[Genq2FromTop2_idx]")
                  .Define("Genb1FromHiggs_eta","GenJet.Eta[Genb1FromHiggs_idx]")
                  .Define("Genb2FromHiggs_eta","GenJet.Eta[Genb2FromHiggs_idx]")

                  .Define("GenbFromTop1_phi",  "GenJet.Phi[GenbFromTop1_idx]")
                  .Define("GenbFromTop2_phi",  "GenJet.Phi[GenbFromTop2_idx]")
                  .Define("Genq1FromTop2_phi", "GenJet.Phi[Genq1FromTop2_idx]")
                  .Define("Genq2FromTop2_phi", "GenJet.Phi[Genq2FromTop2_idx]")
                  .Define("Genb1FromHiggs_phi","GenJet.Phi[Genb1FromHiggs_idx]")
                  .Define("Genb2FromHiggs_phi","GenJet.Phi[Genb2FromHiggs_idx]")

                  .Define("GenbFromTop1_mass",  "GenJet.Mass[GenbFromTop1_idx]")
                  .Define("GenbFromTop2_mass",  "GenJet.Mass[GenbFromTop2_idx]")
                  .Define("Genq1FromTop2_mass", "GenJet.Mass[Genq1FromTop2_idx]")
                  .Define("Genq2FromTop2_mass", "GenJet.Mass[Genq2FromTop2_idx]")
                  .Define("Genb1FromHiggs_mass","GenJet.Mass[Genb1FromHiggs_idx]")
                  .Define("Genb2FromHiggs_mass","GenJet.Mass[Genb2FromHiggs_idx]");

    //Define 4Vecs for final objects on reco level
    auto df5 = df4.Filter("bFromTop1_idx>=0 && GenbFromTop2_idx>=0 && Genq1FromTop2_idx>=0 && Genq2FromTop2_idx>=0 && Genb1FromHiggs_idx>=0 && Genb2FromHiggs_idx>=0")

                  .Define("bFromTop1_pt",  "Jet.PT[bFromTop1_idx]")
                  .Define("bFromTop2_pt",  "Jet.PT[bFromTop2_idx]")
                  .Define("q1FromTop2_pt", "Jet.PT[q1FromTop2_idx]")
                  .Define("q2FromTop2_pt", "Jet.PT[q2FromTop2_idx]")
                  .Define("b1FromHiggs_pt","Jet.PT[b1FromHiggs_idx]")
                  .Define("b2FromHiggs_pt","Jet.PT[b2FromHiggs_idx]")
                  
                  .Define("bFromTop1_eta",  "Jet.Eta[bFromTop1_idx]")
                  .Define("bFromTop2_eta",  "Jet.Eta[bFromTop2_idx]")
                  .Define("q1FromTop2_eta", "Jet.Eta[q1FromTop2_idx]")
                  .Define("q2FromTop2_eta", "Jet.Eta[q2FromTop2_idx]")
                  .Define("b1FromHiggs_eta","Jet.Eta[b1FromHiggs_idx]")
                  .Define("b2FromHiggs_eta","Jet.Eta[b2FromHiggs_idx]")

                  .Define("bFromTop1_phi",  "Jet.Phi[bFromTop1_idx]")
                  .Define("bFromTop2_phi",  "Jet.Phi[bFromTop2_idx]")
                  .Define("q1FromTop2_phi", "Jet.Phi[q1FromTop2_idx]")
                  .Define("q2FromTop2_phi", "Jet.Phi[q2FromTop2_idx]")
                  .Define("b1FromHiggs_phi","Jet.Phi[b1FromHiggs_idx]")
                  .Define("b2FromHiggs_phi","Jet.Phi[b2FromHiggs_idx]")

                  .Define("bFromTop1_mass",  "Jet.Mass[bFromTop1_idx]")
                  .Define("bFromTop2_mass",  "Jet.Mass[bFromTop2_idx]")
                  .Define("q1FromTop2_mass", "Jet.Mass[q1FromTop2_idx]")
                  .Define("q2FromTop2_mass", "Jet.Mass[q2FromTop2_idx]")
                  .Define("b1FromHiggs_mass","Jet.Mass[b1FromHiggs_idx]")
                  .Define("b2FromHiggs_mass","Jet.Mass[b2FromHiggs_idx]");

    auto df61= df5.Filter("muonFromTop1_idx >= 0")
                  .Define("lepFromTop1_pt","Muon.PT[muonFromTop1_idx]")
                  .Define("lepFromTop1_eta","Muon.Eta[muonFromTop1_idx]")
                  .Define("lepFromTop1_phi","Muon.Phi[muonFromTop1_idx]");

    auto df62= df5.Filter("elecFromTop1_idx >= 0")
                  .Define("lepFromTop1_pt","Electron.PT[elecFromTop1_idx]")
                  .Define("lepFromTop1_eta","Electron.Eta[elecFromTop1_idx]")
                  .Define("lepFromTop1_phi","Electron.Phi[elecFromTop1_idx]");

    std::initializer_list<std::string> variables = {
        "nTop", "nW", "nHiggs", "nGenbQuark", 
        "GenPart_bFromTop1_pt", "GenPart_lepFromTop1_pt", "GenPart_bFromTop2_pt", "GenPart_q1FromTop2_pt", "GenPart_q2FromTop2_pt", "GenPart_b1FromHiggs_pt", "GenPart_b2FromHiggs_pt",
        "GenPart_bFromTop1_eta", "GenPart_lepFromTop1_eta", "GenPart_bFromTop2_eta", "GenPart_q1FromTop2_eta", "GenPart_q2FromTop2_eta", "GenPart_b1FromHiggs_eta", "GenPart_b2FromHiggs_eta",
        "GenPart_bFromTop1_phi", "GenPart_lepFromTop1_phi", "GenPart_bFromTop2_phi", "GenPart_q1FromTop2_phi", "GenPart_q2FromTop2_phi", "GenPart_b1FromHiggs_phi", "GenPart_b2FromHiggs_phi",
        "GenPart_bFromTop1_mass", "GenPart_lepFromTop1_mass", "GenPart_bFromTop2_mass", "GenPart_q1FromTop2_mass", "GenPart_q2FromTop2_mass", "GenPart_b1FromHiggs_mass", "GenPart_b2FromHiggs_mass",
        "GenbFromTop1_pt", "GenbFromTop2_pt", "Genq1FromTop2_pt", "Genq2FromTop2_pt", "Genb1FromHiggs_pt", "Genb2FromHiggs_pt",
        "GenbFromTop1_eta", "GenbFromTop2_eta", "Genq1FromTop2_eta", "Genq2FromTop2_eta", "Genb1FromHiggs_eta", "Genb2FromHiggs_eta",
        "GenbFromTop1_phi", "GenbFromTop2_phi", "Genq1FromTop2_phi", "Genq2FromTop2_phi", "Genb1FromHiggs_phi", "Genb2FromHiggs_phi",
        "GenbFromTop1_mass", "GenbFromTop2_mass", "Genq1FromTop2_mass", "Genq2FromTop2_mass", "Genb1FromHiggs_mass", "Genb2FromHiggs_mass",
        "bFromTop1_pt", "lepFromTop1_pt", "bFromTop2_pt", "q1FromTop2_pt", "q2FromTop2_pt", "b1FromHiggs_pt", "b2FromHiggs_pt",
        "bFromTop1_eta", "lepFromTop1_eta", "bFromTop2_eta", "q1FromTop2_eta", "q2FromTop2_eta", "b1FromHiggs_eta", "b2FromHiggs_eta",
        "bFromTop1_phi", "lepFromTop1_phi", "bFromTop2_phi", "q1FromTop2_phi", "q2FromTop2_phi", "b1FromHiggs_phi", "b2FromHiggs_phi",
        "bFromTop1_mass", "bFromTop2_mass", "q1FromTop2_mass", "q2FromTop2_mass", "b1FromHiggs_mass", "b2FromHiggs_mass",
    };
    df61.Snapshot(treename, outdir+channel+"_mu.root", variables);
    df62.Snapshot(treename, outdir+channel+"_elec.root", variables);
    std::cout << "done" << std::endl;


    //df.Snapshot<TClonesArray, TClonesArray, TClonesArray, TClonesArray>("outputTree", "out.root", variables, ROOT::RDF::RSnapshotOptions("RECreate", ROOT::kZLIB, 1, 0, 99, false));
    //df.Snapshot<TClonesArray, TClonesArray, TClonesArray, TClonesArray>("outputTree", "out.root", {"Event", "Electron", "Muon", "Jet"}, ROOT::RDF::RSnapshotOptions("RECreate", ROOT::kZLIB, 1, 0, 99, false));
}
