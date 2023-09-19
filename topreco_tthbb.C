
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

void topreco_tthbb(std::string infile, std::string outfile="./topreco/out.root"){
    gSystem->Load("libDelphes");

    std::cout << infile << std::endl;
    std::cout << outfile << std::endl;
    auto treename = "Delphes";

    auto _df = ROOT::RDataFrame(treename, infile);

    auto df1 = _df.Filter("b1FromHiggs_pt>0 && b2FromHiggs_pt>0")
                  .Define("GenPart_Higgs", ::HiggsReco, {"GenPart_b1FromHiggs_pt","GenPart_b1FromHiggs_eta","GenPart_b1FromHiggs_phi","GenPart_b1FromHiggs_mass","GenPart_b2FromHiggs_pt","GenPart_b2FromHiggs_eta","GenPart_b2FromHiggs_phi","GenPart_b2FromHiggs_mass"})
                  .Define("GenPart_Higgs_pt",   "GenPart_Higgs[0]")
                  .Define("GenPart_Higgs_eta",  "GenPart_Higgs[1]")
                  .Define("GenPart_Higgs_phi",  "GenPart_Higgs[2]")
                  .Define("GenPart_Higgs_mass", "GenPart_Higgs[3]")
                  .Define("GenPart_Higgs_dR",   "GenPart_Higgs[4]")
                  .Define("GenHiggs", ::HiggsReco, {"Genb1FromHiggs_pt","Genb1FromHiggs_eta","Genb1FromHiggs_phi","Genb1FromHiggs_mass","Genb2FromHiggs_pt","Genb2FromHiggs_eta","Genb2FromHiggs_phi","Genb2FromHiggs_mass"})
                  .Define("GenHiggs_pt",   "GenHiggs[0]")
                  .Define("GenHiggs_eta",  "GenHiggs[1]")
                  .Define("GenHiggs_phi",  "GenHiggs[2]")
                  .Define("GenHiggs_mass", "GenHiggs[3]")
                  .Define("GenHiggs_dR",   "GenHiggs[4]")
                  .Define("Higgs", ::HiggsReco, {"b1FromHiggs_pt","b1FromHiggs_eta","b1FromHiggs_phi","b1FromHiggs_mass","b2FromHiggs_pt","b2FromHiggs_eta","b2FromHiggs_phi","b2FromHiggs_mass"})
                  .Define("Higgs_pt",   "Higgs[0]")
                  .Define("Higgs_eta",  "Higgs[1]")
                  .Define("Higgs_phi",  "Higgs[2]")
                  .Define("Higgs_mass", "Higgs[3]")
                  .Define("Higgs_dR",   "Higgs[4]");

    auto df2 = df1.Filter("bFromTop2_pt*q1FromTop2_pt*q2FromTop2_pt>0")
                  .Define("GenPart_HadTop", ::HadTopReco, {"GenPart_bFromTop2_pt", "GenPart_bFromTop2_eta", "GenPart_bFromTop2_phi", "GenPart_bFromTop2_mass", "GenPart_q1FromTop2_pt", "GenPart_q1FromTop2_eta", "GenPart_q1FromTop2_phi", "GenPart_q1FromTop2_mass", "GenPart_q2FromTop2_pt", "GenPart_q2FromTop2_eta", "GenPart_q2FromTop2_phi", "GenPart_q2FromTop2_mass"}) 
                  .Define("GenPart_Had_W_pt", "GenPart_HadTop[0]")
                  .Define("GenPart_Had_W_eta", "GenPart_HadTop[1]")
                  .Define("GenPart_Had_W_phi", "GenPart_HadTop[2]")
                  .Define("GenPart_Had_W_mass", "GenPart_HadTop[3]")
                  .Define("GenPart_Had_W_dR", "GenPart_HadTop[4]")
                  .Define("GenPart_HadTop_pt", "GenPart_HadTop[5]")
                  .Define("GenPart_HadTop_eta", "GenPart_HadTop[6]")
                  .Define("GenPart_HadTop_phi", "GenPart_HadTop[7]")
                  .Define("GenPart_HadTop_mass", "GenPart_HadTop[8]")
                  .Define("GenPart_HadTop_dR", "GenPart_HadTop[9]")
                  .Define("GenHadTop", ::HadTopReco, {"GenbFromTop2_pt", "GenbFromTop2_eta", "GenbFromTop2_phi", "GenbFromTop2_mass", "Genq1FromTop2_pt", "Genq1FromTop2_eta", "Genq1FromTop2_phi", "Genq1FromTop2_mass", "Genq2FromTop2_pt", "Genq2FromTop2_eta", "Genq2FromTop2_phi", "Genq2FromTop2_mass"}) 
                  .Define("GenHad_W_pt", "GenHadTop[0]")
                  .Define("GenHad_W_eta", "GenHadTop[1]")
                  .Define("GenHad_W_phi", "GenHadTop[2]")
                  .Define("GenHad_W_mass", "GenHadTop[3]")
                  .Define("GenHad_W_dR", "GenHadTop[4]")
                  .Define("GenHadTop_pt", "GenHadTop[5]")
                  .Define("GenHadTop_eta", "GenHadTop[6]")
                  .Define("GenHadTop_phi", "GenHadTop[7]")
                  .Define("GenHadTop_mass", "GenHadTop[8]")
                  .Define("GenHadTop_dR", "GenHadTop[9]")
                  .Define("HadTop", ::HadTopReco, {"bFromTop2_pt", "bFromTop2_eta", "bFromTop2_phi", "bFromTop2_mass", "q1FromTop2_pt", "q1FromTop2_eta", "q1FromTop2_phi", "q1FromTop2_mass", "q2FromTop2_pt", "q2FromTop2_eta", "q2FromTop2_phi", "q2FromTop2_mass"}) 
                  .Define("Had_W_pt", "HadTop[0]")
                  .Define("Had_W_eta", "HadTop[1]")
                  .Define("Had_W_phi", "HadTop[2]")
                  .Define("Had_W_mass", "HadTop[3]")
                  .Define("Had_W_dR", "HadTop[4]")
                  .Define("HadTop_pt", "HadTop[5]")
                  .Define("HadTop_eta", "HadTop[6]")
                  .Define("HadTop_phi", "HadTop[7]")
                  .Define("HadTop_mass", "HadTop[8]")
                  .Define("HadTop_dR", "HadTop[9]");

    std::initializer_list<std::string> variables = {
        "GenPart_Higgs_pt", "GenPart_Higgs_eta", "GenPart_Higgs_phi", "GenPart_Higgs_mass", "GenPart_Higgs_dR", 
        "GenHiggs_pt", "GenHiggs_eta", "GenHiggs_phi", "GenHiggs_mass", "GenHiggs_dR", 
        "Higgs_pt", "Higgs_eta", "Higgs_phi", "Higgs_mass", "Higgs_dR", 
        "GenPart_Had_W_pt", "GenPart_Had_W_eta", "GenPart_Had_W_phi", "GenPart_Had_W_mass", "GenPart_Had_W_dR", 
        "GenHad_W_pt", "GenHad_W_eta", "GenHad_W_phi", "GenHad_W_mass", "GenHad_W_dR", 
        "Had_W_pt", "Had_W_eta", "Had_W_phi", "Had_W_mass", "Had_W_dR", 
        "GenPart_HadTop_pt", "GenPart_HadTop_eta", "GenPart_HadTop_phi", "GenPart_HadTop_mass", "GenPart_HadTop_dR", 
        "GenHadTop_pt", "GenHadTop_eta", "GenHadTop_phi", "GenHadTop_mass", "GenHadTop_dR", 
        "HadTop_pt", "HadTop_eta", "HadTop_phi", "HadTop_mass", "HadTop_dR", 
    };

    df2.Snapshot(treename, outfile, variables);
    std::cout << "done" << std::endl;


    //df.Snapshot<TClonesArray, TClonesArray, TClonesArray, TClonesArray>("outputTree", "out.root", variables, ROOT::RDF::RSnapshotOptions("RECreate", ROOT::kZLIB, 1, 0, 99, false));
    //df.Snapshot<TClonesArray, TClonesArray, TClonesArray, TClonesArray>("outputTree", "out.root", {"Event", "Electron", "Muon", "Jet"}, ROOT::RDF::RSnapshotOptions("RECreate", ROOT::kZLIB, 1, 0, 99, false));
}
