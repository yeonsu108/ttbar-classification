/*
root -l -b -q ana.C'("step_1.root", "step_1_plots.root")'
*/

//------------------------------------------------------------------------------
#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#endif
#include <vector>
#include <algorithm>

bool isFromTop(const GenParticle * p, const TClonesArray * branchParticle, int motherPID = 6){
  float m1 = p->M1; float m2 = p->M2;
  GenParticle * mother;
  if( m1 < 0 && m2 < 0) return false;

  if( ( m1 >= 0 && m2 < 0 ) || ( m1 == m2 ) ){
      mother = (GenParticle *) branchParticle->At(m1);
  }else if( m1 < 0 && m2 >= 0){
      mother = (GenParticle *) branchParticle->At(m2);
  }else{
      GenParticle * mother1 = (GenParticle *) branchParticle->At(m1);
      GenParticle * mother2 = (GenParticle *) branchParticle->At(m2);
      //cout << " mother1:" << mother1->PID << " mother2:" << mother2->PID;
      if( abs(mother1->PID) == motherPID || abs(mother2->PID) == motherPID ) return true;
      else return (isFromTop(mother1, branchParticle, motherPID) || isFromTop(mother2, branchParticle, motherPID));
  }
  //cout << " mother:" << mother->PID;
  if( abs(mother->PID) == motherPID ) return true;
  else return isFromTop(mother, branchParticle, motherPID);
}

bool compareJetPt(const Jet * jet1, const Jet * jet2){
  return jet1->P4().Pt() > jet2->P4().Pt();
} 

void duplication( vector<Jet*>& a, vector<Jet*>& b){
  for( int i = 0; i < a.size(); i++){
    for( int j = 0; j < b.size(); j++){
      if( a[i]->P4().Pt() == b[j]->P4().Pt() ) a.erase( a.begin() + i );
    }
  }
}

void ana_ttbar(const char *inputFile, const char *outputFile, int jcut=4, int bcut=2, int no_file=1){
  //jcut is for the number of jets cut and bcut is for the number of b-jets cut
  gSystem->Load("libDelphes");
 
  // Create chain of root trees
  TChain chain("Delphes");
  chain.Add(inputFile);
 
  cout<<"Single Lepton Channel"<<endl;
  cout<<"Filename: "<< inputFile << endl;

  //Output
  TFile *fout = TFile::Open(outputFile,"RECREATE");
  fout->cd();
 
  //DNN variables
  int category;
  int njets, nbjets, ncjets, nMuon, nElectron, nLepton;
  int Jet1_btag, Jet2_btag, Jet3_btag, Jet4_btag;
  float bjet1_pt, bjet1_eta, bjet1_phi, bjet1_e;
  float bjet2_pt, bjet2_eta, bjet2_phi, bjet2_e;
 
  float Jet_pt1, Jet_eta1, Jet_phi1, Jet_e1;
  float Jet_pt2, Jet_eta2, Jet_phi2, Jet_e2;
  float Jet_pt3, Jet_eta3, Jet_phi3, Jet_e3;
  float Jet_pt4, Jet_eta4, Jet_phi4, Jet_e4;
  float Lepton_pt, Lepton_eta, Lepton_phi, Lepton_e;
  float MET_eta, MET_phi, MET_px, MET_py, MET_met, HT;

  float selbjet1_pt, selbjet1_eta, selbjet1_phi, selbjet1_e;
  float selbjet2_pt, selbjet2_eta, selbjet2_phi, selbjet2_e;

  float bbdR, bbdEta, bbdPhi, bbPt, bbEta, bbPhi, bbMass, bbHt, bbMt;
  float nubbdR, nubbdEta, nubbdPhi, nubbPt, nubbEta, nubbPhi, nubbMass, nubbHt, nubbMt;
  float lbbdR, lbbdEta, lbbdPhi, lbbPt, lbbEta, lbbPhi, lbbMass, lbbHt, lbbMt;
  float Wjb1dR, Wjb1dEta, Wjb1dPhi, Wjb1Pt, Wjb1Eta, Wjb1Phi, Wjb1Mass, Wjb1Ht, Wjb1Mt;
  float Wjb2dR, Wjb2dEta, Wjb2dPhi, Wjb2Pt, Wjb2Eta, Wjb2Phi, Wjb2Mass, Wjb2Ht, Wjb2Mt;
  float Wlb1dR, Wlb1dEta, Wlb1dPhi, Wlb1Pt, Wlb1Eta, Wlb1Phi, Wlb1Mass, Wlb1Ht, Wlb1Mt;
  float Wlb2dR, Wlb2dEta, Wlb2dPhi, Wlb2Pt, Wlb2Eta, Wlb2Phi, Wlb2Mass, Wlb2Ht, Wlb2Mt;
  float nub1dR, nub1dEta, nub1dPhi, nub1Pt, nub1Eta, nub1Phi, nub1Mass, nub1Ht, nub1Mt;
  float nub2dR, nub2dEta, nub2dPhi, nub2Pt, nub2Eta, nub2Phi, nub2Mass, nub2Ht, nub2Mt;
  float lb1dR, lb1dEta, lb1dPhi, lb1Pt, lb1Eta, lb1Phi, lb1Mass, lb1Ht, lb1Mt;
  float lb2dR, lb2dEta, lb2dPhi, lb2Pt, lb2Eta, lb2Phi, lb2Mass, lb2Ht, lb2Mt;
 
 
  // Selected Events (Cut Flow)
  int s1 = 0; int s2 = 0; int s3 = 0; int s4 = 0;
  int ttbbs2 = 0; int ttbbs3 = 0; int ttbbs4 = 0;
  int ttbjs2 = 0; int ttbjs3 = 0; int ttbjs4 = 0;
  int ttccs2 = 0; int ttccs3 = 0; int ttccs4 = 0;
  int ttlfs2 = 0; int ttlfs3 = 0; int ttlfs4 = 0;
 
  int nttbb = 0; int nttbj = 0; int nttcc = 0; int nttlf = 0;
 
  //Tree for Deep learning input 
  TTree * dnn_tree = new TTree( "dnn_input", "tree for dnn");
  dnn_tree->Branch("category",&category,"category/i");
  dnn_tree->Branch("njets",&njets,"njets/i");
  dnn_tree->Branch("nbjets",&nbjets,"nbjets/i"); 
  dnn_tree->Branch("ncjets",&ncjets,"ncjets/i"); 
  dnn_tree->Branch("nElectron",&nElectron,"nElectron/s");
  dnn_tree->Branch("nMuon",&nMuon,"nMuon/s");
  dnn_tree->Branch("nLepton",&nLepton,"nLepton/s");
 
  dnn_tree->Branch("bjet1_pt",&bjet1_pt,"bjet1_pt/f");    dnn_tree->Branch("bjet2_pt",&bjet2_pt,"bjet2_pt/f");
  dnn_tree->Branch("bjet1_eta",&bjet1_eta,"bjet1_eta/f"); dnn_tree->Branch("bjet2_eta",&bjet2_eta,"bjet2_eta/f");
  dnn_tree->Branch("bjet1_phi",&bjet1_phi,"bjet1_phi/f"); dnn_tree->Branch("bjet2_phi",&bjet2_phi,"bjet2_phi/f");
  dnn_tree->Branch("bjet1_e",&bjet2_e,"bjet1_e/f");       dnn_tree->Branch("bjet2_e",&bjet2_e,"bjet2_e/f");
 
  dnn_tree->Branch("Jet1_pt",&Jet_pt1,"Jet1_pt/f");       dnn_tree->Branch("Jet3_pt",&Jet_pt3,"Jet3_pt/f");
  dnn_tree->Branch("Jet1_eta",&Jet_eta1,"Jet1_eta/f");    dnn_tree->Branch("Jet3_eta",&Jet_eta3,"Jet3_eta/f");
  dnn_tree->Branch("Jet1_phi",&Jet_phi1,"Jet1_phi/f");    dnn_tree->Branch("Jet3_phi",&Jet_phi3,"Jet3_phi/f");
  dnn_tree->Branch("Jet1_e",&Jet_e1,"Jet1_e/f");          dnn_tree->Branch("Jet3_e",&Jet_e3,"Jet3_e/f");
  dnn_tree->Branch("Jet1_btag",&Jet1_btag,"Jet1_btag/i"); dnn_tree->Branch("Jet3_btag",&Jet3_btag,"Jet3_btag/i");
  dnn_tree->Branch("Jet2_pt",&Jet_pt2,"Jet2_pt/f");       dnn_tree->Branch("Jet4_pt",&Jet_pt4,"Jet4_pt/f");
  dnn_tree->Branch("Jet2_eta",&Jet_eta2,"Jet2_eta/f");    dnn_tree->Branch("Jet4_eta",&Jet_eta4,"Jet4_eta/f");
  dnn_tree->Branch("Jet2_phi",&Jet_phi2,"Jet2_phi/f");    dnn_tree->Branch("Jet4_phi",&Jet_phi4,"Jet4_phi/f");
  dnn_tree->Branch("Jet2_e",&Jet_e2,"Jet2_e/f");          dnn_tree->Branch("Jet4_e",&Jet_e4,"Jet4_e/f");
  dnn_tree->Branch("Jet2_btag",&Jet2_btag,"Jet2_btag/i"); dnn_tree->Branch("Jet4_btag",&Jet4_btag,"Jet4_btag/i");
 
  dnn_tree->Branch("Lepton_pt",&Lepton_pt,"Lepton_pt/f");
  dnn_tree->Branch("Lepton_eta",&Lepton_eta,"Lepton_eta/f");
  dnn_tree->Branch("Lepton_phi",&Lepton_phi,"Lepton_phi/f");
  dnn_tree->Branch("Lepton_e",&Lepton_e,"Lepton_e/f");
  dnn_tree->Branch("MET_eta",&MET_eta,"MET_eta/f");
  dnn_tree->Branch("MET_phi",&MET_phi,"MET_phi/f");
  dnn_tree->Branch("MET_px",&MET_px,"MET_px/f");
  dnn_tree->Branch("MET_py",&MET_py,"MET_py/f");
  dnn_tree->Branch("MET_met",&MET_met,"MET_met/f");
  dnn_tree->Branch("HT",&HT,"HT/f");
  
  dnn_tree->Branch("selbjet1_pt" ,&selbjet1_pt ,"selbjet1_pt/f");
  dnn_tree->Branch("selbjet1_eta",&selbjet1_eta,"selbjet1_eta/f");
  dnn_tree->Branch("selbjet1_phi",&selbjet1_phi,"selbjet1_phi/f");
  dnn_tree->Branch("selbjet1_e"  ,&selbjet1_e  ,"selbjet1_e/f");
  dnn_tree->Branch("selbjet2_pt" ,&selbjet2_pt ,"selbjet2_pt/f");
  dnn_tree->Branch("selbjet2_eta",&selbjet2_eta,"selbjet2_eta/f");
  dnn_tree->Branch("selbjet2_phi",&selbjet2_phi,"selbjet2_phi/f");
  dnn_tree->Branch("selbjet2_e"  ,&selbjet2_e  ,"selbjet2_e/f");
 
  dnn_tree->Branch("bbdR",&bbdR,"bbdR/f");                dnn_tree->Branch("nubbdR",&nubbdR,"nubbdR/f");
  dnn_tree->Branch("bbdEta",&bbdEta,"bbdEta/f");          dnn_tree->Branch("nubbdEta",&nubbdEta,"nubbdEta/f");
  dnn_tree->Branch("bbdPhi",&bbdPhi,"bbdPhi/f");          dnn_tree->Branch("nubbdPhi",&nubbdPhi,"nubbdPhi/f");
  dnn_tree->Branch("bbPt",&bbPt,"bbPt/f");                dnn_tree->Branch("nubbPt",&nubbPt,"nubbPt/f");
  dnn_tree->Branch("bbEta",&bbEta,"bbEta/f");             dnn_tree->Branch("nubbEta",&nubbEta,"nubbEta/f");
  dnn_tree->Branch("bbPhi",&bbPhi,"bbPhi/f");             dnn_tree->Branch("nubbPhi",&nubbPhi,"nubbPhi/f");
  dnn_tree->Branch("bbMass",&bbMass,"bbMass/f");          dnn_tree->Branch("nubbMass",&nubbMass,"nubbMass/f");
  dnn_tree->Branch("bbHt",&bbHt,"bbHt/f");                dnn_tree->Branch("nubbHt",&nubbHt,"nubbHt/f");
  dnn_tree->Branch("bbMt",&bbMt,"bbMt/f");                dnn_tree->Branch("nubbMt",&nubbMt,"nubbMt/f");

  dnn_tree->Branch("nub1dR",&nub1dR,"nub1dR/f");          dnn_tree->Branch("nub2dR",&nub2dR,"nub2dR/f");
  dnn_tree->Branch("nub1dEta",&nub1dEta,"nub1dEta/f");    dnn_tree->Branch("nub2dEta",&nub2dEta,"nub2dEta/f");
  dnn_tree->Branch("nub1dPhi",&nub1dPhi,"nub1dPhi/f");    dnn_tree->Branch("nub2dPhi",&nub2dPhi,"nub2dPhi/f");
  dnn_tree->Branch("nub1Pt",&nub1Pt,"nub1Pt/f");          dnn_tree->Branch("nub2Pt",&nub2Pt,"nub2Pt/f");
  dnn_tree->Branch("nub1Eta",&nub1Eta,"nub1Eta/f");       dnn_tree->Branch("nub2Eta",&nub2Eta,"nub2Eta/f");
  dnn_tree->Branch("nub1Phi",&nub1Phi,"nub1Phi/f");       dnn_tree->Branch("nub2Phi",&nub2Phi,"nub2Phi/f");
  dnn_tree->Branch("nub1Mass",&nub1Mass,"nub1Mass/f");    dnn_tree->Branch("nub2Mass",&nub2Mass,"nub2Mass/f");
  dnn_tree->Branch("nub1Ht",&nub1Ht,"nub1Ht/f");          dnn_tree->Branch("nub2Ht",&nub2Ht,"nub2Ht/f");
  dnn_tree->Branch("nub1Mt",&nub1Mt,"nub1Mt/f");          dnn_tree->Branch("nub2Mt",&nub2Mt,"nub2Mt/f");
 
  dnn_tree->Branch("lbbdR",&lbbdR,"lbbdR/f");
  dnn_tree->Branch("lbbdEta",&lbbdEta,"lbbdEta/f");
  dnn_tree->Branch("lbbdPhi",&lbbdPhi,"lbbdPhi/f");
  dnn_tree->Branch("lbbPt",&lbbPt,"lbbPt/f");
  dnn_tree->Branch("lbbEta",&lbbEta,"lbbEta/f");
  dnn_tree->Branch("lbbPhi",&lbbPhi,"lbbPhi/f");
  dnn_tree->Branch("lbbMass",&lbbMass,"lbbMass/f");
  dnn_tree->Branch("lbbHt",&lbbHt,"lbbHt/f");
  dnn_tree->Branch("lbbMt",&lbbMt,"lbbMt/f");
 
  dnn_tree->Branch("lb1dR",&lb1dR,"lb1dR/f");             dnn_tree->Branch("lb2dR",&lb2dR,"lb2dR/f");
  dnn_tree->Branch("lb1dEta",&lb1dEta,"lb1dEta/f");       dnn_tree->Branch("lb2dEta",&lb2dEta,"lb2dEta/f");
  dnn_tree->Branch("lb1dPhi",&lb1dPhi,"lb1dPhi/f");       dnn_tree->Branch("lb2dPhi",&lb2dPhi,"lb2dPhi/f");
  dnn_tree->Branch("lb1Pt",&lb1Pt,"lb1Pt/f");             dnn_tree->Branch("lb2Pt",&lb2Pt,"lb2Pt/f");
  dnn_tree->Branch("lb1Eta",&lb1Eta,"lb1Eta/f");          dnn_tree->Branch("lb2Eta",&lb2Eta,"lb2Eta/f");
  dnn_tree->Branch("lb1Phi",&lb1Phi,"lb1Phi/f");          dnn_tree->Branch("lb2Phi",&lb2Phi,"lb2Phi/f");
  dnn_tree->Branch("lb1Mass",&lb1Mass,"lb1Mass/f");       dnn_tree->Branch("lb2Mass",&lb2Mass,"lb2Mass/f");
  dnn_tree->Branch("lb1Ht",&lb1Ht,"lb1Ht/f");             dnn_tree->Branch("lb2Ht",&lb2Ht,"lb2Ht/f");
  dnn_tree->Branch("lb1Mt",&lb1Mt,"lb1Mt/f");             dnn_tree->Branch("lb2Mt",&lb2Mt,"lb2Mt/f");

  dnn_tree->Branch("Wjb1dR",&Wjb1dR,"Wjb1dR/f");          dnn_tree->Branch("Wjb2dR",&Wjb2dR,"Wjb2dR/f");
  dnn_tree->Branch("Wjb1dEta",&Wjb1dEta,"Wjb1dEta/f");    dnn_tree->Branch("Wjb2dEta",&Wjb2dEta,"Wjb2dEta/f");
  dnn_tree->Branch("Wjb1dPhi",&Wjb1dPhi,"Wjb1dPhi/f");    dnn_tree->Branch("Wjb2dPhi",&Wjb2dPhi,"Wjb2dPhi/f");
  dnn_tree->Branch("Wjb1Pt",&Wjb1Pt,"Wjb1Pt/f");          dnn_tree->Branch("Wjb2Pt",&Wjb2Pt,"Wjb2Pt/f");
  dnn_tree->Branch("Wjb1Eta",&Wjb1Eta,"Wjb1Eta/f");       dnn_tree->Branch("Wjb2Eta",&Wjb2Eta,"Wjb2Eta/f");
  dnn_tree->Branch("Wjb1Phi",&Wjb1Phi,"Wjb1Phi/f");       dnn_tree->Branch("Wjb2Phi",&Wjb2Phi,"Wjb2Phi/f");
  dnn_tree->Branch("Wjb1Mass",&Wjb1Mass,"Wjb1Mass/f");    dnn_tree->Branch("Wjb2Mass",&Wjb2Mass,"Wjb2Mass/f");
  dnn_tree->Branch("Wjb1Ht",&Wjb1Ht,"Wjb1Ht/f");          dnn_tree->Branch("Wjb2Ht",&Wjb2Ht,"Wjb2Ht/f");
  dnn_tree->Branch("Wjb1Mt",&Wjb1Mt,"Wjb1Mt/f");          dnn_tree->Branch("Wjb2Mt",&Wjb2Mt,"Wjb2Mt/f");

  dnn_tree->Branch("Wlb1dR",&Wlb1dR,"Wlb1dR/f");          dnn_tree->Branch("Wlb2dR",&Wlb2dR,"Wlb2dR/f");
  dnn_tree->Branch("Wlb1dEta",&Wlb1dEta,"Wlb1dEta/f");    dnn_tree->Branch("Wlb2dEta",&Wlb2dEta,"Wlb2dEta/f");
  dnn_tree->Branch("Wlb1dPhi",&Wlb1dPhi,"Wlb1dPhi/f");    dnn_tree->Branch("Wlb2dPhi",&Wlb2dPhi,"Wlb2dPhi/f");
  dnn_tree->Branch("Wlb1Pt",&Wlb1Pt,"Wlb1Pt/f");          dnn_tree->Branch("Wlb2Pt",&Wlb2Pt,"Wlb2Pt/f");
  dnn_tree->Branch("Wlb1Eta",&Wlb1Eta,"Wlb1Eta/f");       dnn_tree->Branch("Wlb2Eta",&Wlb2Eta,"Wlb2Eta/f");
  dnn_tree->Branch("Wlb1Phi",&Wlb1Phi,"Wlb1Phi/f");       dnn_tree->Branch("Wlb2Phi",&Wlb2Phi,"Wlb2Phi/f");
  dnn_tree->Branch("Wlb1Mass",&Wlb1Mass,"Wlb1Mass/f");    dnn_tree->Branch("Wlb2Mass",&Wlb2Mass,"Wlb2Mass/f");
  dnn_tree->Branch("Wlb1Ht",&Wlb1Ht,"Wlb1Ht/f");          dnn_tree->Branch("Wlb2Ht",&Wlb2Ht,"Wlb2Ht/f");
  dnn_tree->Branch("Wlb1Mt",&Wlb1Mt,"Wlb1Mt/f");          dnn_tree->Branch("Wlb2Mt",&Wlb2Mt,"Wlb2Mt/f");
 
  // Book histograms
  TH1 *hist_selection = new TH1F("h_selection","Selection",5,0.0,5.0);
 
  // Create object of class ExRootTreeReader
  ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);
  Long64_t numberOfEntries = treeReader->GetEntries();
 
  // Get pointers to branches used in this analysis
  TClonesArray *branchGenJet  = treeReader->UseBranch("GenJet");
  TClonesArray *branchJet  = treeReader->UseBranch("Jet");
  TClonesArray *branchParticle  = treeReader->UseBranch("Particle");
  TClonesArray *branchElectron = treeReader->UseBranch("Electron");
  TClonesArray *branchMuon = treeReader->UseBranch("Muon");
  TClonesArray *branchMissingET = treeReader->UseBranch("MissingET");
 
 
  Int_t numberOfSelectedEvents = 0;
 
  vector<Jet *> bJets; vector<Jet *> Jets;
  vector<Electron *> Electrons; vector<Muon *> Muons;
 
  Jet *jet;
  Electron *electron;
  Muon *muon;
  MissingET *met;
 
  int entry, i;
 
  // Loop over all events
  cout << "total # of events: " << numberOfEntries << endl;
  for(entry = 0; entry < numberOfEntries; ++entry) {
    if(entry%1000 == 0 && entry < 10000) cout << "event number: " << entry << endl;
    else if(entry%10000 == 0) cout<< "event number: " << entry << endl;

    // Load selected branches with data from specified event
    treeReader->ReadEntry(entry);
    
    // Initialization
    Jets.clear(); Electrons.clear(); Muons.clear();  bJets.clear();
    
    // dnn variables
    category = -1;
    selbjet1_pt = -9999; selbjet1_eta = -9999; selbjet1_phi = -9999; selbjet1_e = -9999;
    selbjet2_pt = -9999; selbjet2_eta = -9999; selbjet2_phi = -9999; selbjet2_e = -9999;

    bbdR = -9999; bbdEta = -9999; bbdPhi = -9999;
    bbPt = -9999; bbEta = -9999; bbPhi = -9999; bbMass = -9999;
    bbHt = -9999; bbMt = -9999;
 
    nubbdR = -9999; nubbdEta = -9999; nubbdPhi = -9999;
    nubbPt = -9999; nubbEta = -9999; nubbPhi = -9999; nubbMass = -9999;
    nubbHt = -9999; nubbMt = -9999;

    nub1dR = -9999; nub1dEta = -9999; nub1dPhi = -9999;
    nub1Pt = -9999; nub1Eta = -9999; nub1Phi = -9999; nub1Mass = -9999;
    nub1Ht = -9999; nub1Mt = -9999;

    nub2dR = -9999; nub2dEta = -9999; nub2dPhi = -9999;
    nub2Pt = -9999; nub2Eta = -9999; nub2Phi = -9999; nub2Mass = -9999;
    nub2Ht = -9999; nub2Mt = -9999;

    lbbdR = -9999; lbbdEta = -9999; lbbdPhi = -9999;
    lbbPt = -9999; lbbEta = -9999; lbbPhi = -9999; lbbMass = -9999;
    lbbHt = -9999; lbbMt = -9999;

    lb1dR = -9999; lb1dEta = -9999; lb1dPhi = -9999;
    lb1Pt = -9999; lb1Eta = -9999; lb1Phi = -9999; lb1Mass = -9999;
    lb1Ht = -9999; lb1Mt = -9999;
    
    lb2dR = -9999; lb2dEta = -9999; lb2dPhi = -9999;
    lb2Pt = -9999; lb2Eta = -9999; lb2Phi = -9999; lb2Mass = -9999;
    lb2Ht = -9999; lb2Mt = -9999;
 
    Wjb1dR = -9999; Wjb1dEta = -9999; Wjb1dPhi = -9999;
    Wjb1Pt = -9999; Wjb1Eta = -9999; Wjb1Phi = -9999; Wjb1Mass = -9999;
    Wjb1Ht = -9999; Wjb1Mt = -9999;
    
    Wjb2dR = -9999; Wjb2dEta = -9999; Wjb2dPhi = -9999;
    Wjb2Pt = -9999; Wjb2Eta = -9999; Wjb2Phi = -9999; Wjb2Mass = -9999;
    Wjb2Ht = -9999; Wjb2Mt = -9999;
 
    Wlb1dR = -9999; Wlb1dEta = -9999; Wlb1dPhi = -9999;
    Wlb1Pt = -9999; Wlb1Eta = -9999; Wlb1Phi = -9999; Wlb1Mass = -9999;
    Wlb1Ht = -9999; Wlb1Mt = -9999;
    
    Wlb2dR = -9999; Wlb2dEta = -9999; Wlb2dPhi = -9999;
    Wlb2Pt = -9999; Wlb2Eta = -9999; Wlb2Phi = -9999; Wlb2Mass = -9999;
    Wlb2Ht = -9999; Wlb2Mt = -9999;

    // tree variables // for mindR
    Jet_pt1 = -9999; Jet_eta1 = -9999; Jet_phi1 = -9999; Jet_e1 = -9999; Jet1_btag = -9999;
    Jet_pt2 = -9999; Jet_eta2 = -9999; Jet_phi2 = -9999; Jet_e2 = -9999; Jet2_btag = -9999;
    Jet_pt3 = -9999; Jet_eta3 = -9999; Jet_phi3 = -9999; Jet_e3 = -9999; Jet3_btag = -9999;
    Jet_pt4 = -9999; Jet_eta4 = -9999; Jet_phi4 = -9999; Jet_e4 = -9999; Jet4_btag = -9999;
    Lepton_pt = -9999; Lepton_eta = -9999; Lepton_phi = -9999; Lepton_e = -9999;
    bjet1_pt = -9999; bjet1_eta = -9999; bjet1_phi = -9999; bjet1_e = -9999;
    bjet2_pt = -9999; bjet2_eta = -9999; bjet2_phi = -9999; bjet2_e = -9999;
    MET_eta = -9999; MET_phi = -9999; MET_px = -9999; MET_py = -9999; MET_met = -9999, HT = -9999;
 
    //GenParticle Selection (S1)
    int ntop = 0;
    vector<GenParticle*> GenAddQuarks; vector<GenParticle*> GenParticleFromW; vector<GenParticle*> GenParticleFromTop;
    vector<Jet*> GenAddJets; vector<Jet*> GenAddbJets; vector<Jet*> GenAddcJets; vector<Jet*> GenAddlfJets;
 

    // Find Additional Quarks (not from Top)
    for(i = 0; i < branchParticle->GetEntriesFast(); ++i){
      GenParticle *genP = (GenParticle*) branchParticle->At(i);
      if( abs(genP->PID) < 1 || abs(genP->PID) > 6 ) continue;
      GenParticle *dauP1 = (GenParticle *) branchParticle->At(genP->D1);
      GenParticle *dauP2 = (GenParticle *) branchParticle->At(genP->D2);
      if( abs(dauP1->PID) == abs(genP->PID) || abs(dauP2->PID) == abs(genP->PID)) continue; // if particle is not last, continue 
      if( abs(genP->PID) == 6 ) ntop++;
      else if( isFromTop(genP, branchParticle, 24) ) GenParticleFromW.push_back(genP); 
      else if( isFromTop(genP, branchParticle, 6) ) GenParticleFromTop.push_back(genP);
      else GenAddQuarks.push_back(genP);
    } 

    // Match GenAdditionalQuark and GenJet
    // And classify its PID
    for( i = 0; i < GenAddQuarks.size(); i++ ){
      int genP_PID = abs(GenAddQuarks[i]->PID);
      Jet * matchedJet; float mindR = 9999;
      for( int j = 0 ; j < branchGenJet->GetEntriesFast() ; j++ ){
        Jet * genjet = (Jet*) branchGenJet->At(j);
        if( genjet->P4().Pt() < 20 || genjet->P4().Eta() > 3 ) continue; // (gen level) object selection
        if( GenAddQuarks[i]->P4().DeltaR( genjet->P4() ) < mindR ) {
            mindR = GenAddQuarks[i]->P4().DeltaR( genjet->P4() );
            matchedJet = genjet;
        }
      }
      if( mindR > 0.5 ) continue;
      GenAddJets.push_back(matchedJet);
      if( genP_PID == 5 ) GenAddbJets.push_back(matchedJet);
      else if( genP_PID == 4 ) GenAddcJets.push_back(matchedJet);
      else if( genP_PID >= 1 && genP_PID <= 3 ) GenAddlfJets.push_back(matchedJet);
    }
    
    //sort by PT and remove duplicates
    sort(GenAddJets.begin(), GenAddJets.end(), compareJetPt);
    GenAddJets.erase(unique(GenAddJets.begin(), GenAddJets.end()), GenAddJets.end());
    
    sort(GenAddbJets.begin(), GenAddbJets.end(), compareJetPt);
    GenAddbJets.erase(unique(GenAddbJets.begin(), GenAddbJets.end()), GenAddbJets.end());
 
    sort(GenAddcJets.begin(), GenAddcJets.end(), compareJetPt);
    GenAddcJets.erase(unique(GenAddcJets.begin(), GenAddcJets.end()), GenAddcJets.end());
 
    sort(GenAddlfJets.begin(), GenAddlfJets.end(), compareJetPt);
    GenAddlfJets.erase(unique(GenAddlfJets.begin(), GenAddlfJets.end()), GenAddlfJets.end());
 
    duplication(GenAddcJets, GenAddbJets);
    duplication(GenAddlfJets, GenAddbJets);
    duplication(GenAddlfJets, GenAddcJets);

    bool isttbb=false; bool isttbj=false; bool isttcc=false; bool isttlf=false;
    if( GenAddJets.size() < 2 ) continue; // s1. ttjj event selection
    if( GenAddbJets.size() > 1)      { isttbb=true; category = 1; nttbb++; }
    else if( GenAddbJets.size() > 0) { isttbj=true; category = 2; nttbj++; }
    else if( GenAddcJets.size() > 1) { isttcc=true; category = 3; nttcc++; }
    else                             { isttlf=true; category = 4; nttlf++; }
    s1++;
 
    //Lepton Selection (S2)
    //Electron Selection
    nElectron = 0;
    for(i = 0; i < branchElectron->GetEntries(); ++i){
      electron = (Electron*) branchElectron->At(i);
      if((electron->PT < 20) || (fabs(electron->Eta) > 3)) continue;
      nElectron++;
      Electrons.push_back(electron);
    }
    //Muon Selection
    nMuon = 0;
    for(i = 0; i < branchMuon->GetEntries(); ++i){
      muon = (Muon*) branchMuon->At(i);
      if((muon->PT < 30) || (fabs(muon->Eta) > 2.8)) continue;
      nMuon++;
      Muons.push_back(muon);
    }
    nLepton = nElectron + nMuon;
    
    //TODO Lepton selection nLepton >= 1 (or == 1)
    if (nLepton != 1) continue;   // S2
    s2++;
    if (nLepton == 1 && isttbb) ttbbs2++;
    if (nLepton == 1 && isttbj) ttbjs2++;
    if (nLepton == 1 && isttcc) ttccs2++;
    if (nLepton == 1 && isttlf) ttlfs2++;
 
    // Jet and b-tag Selections (S3)
    njets = 0; nbjets = 0; ncjets = 0; HT = 0;
    for(i = 0; i < branchJet->GetEntriesFast(); ++i){
      jet = (Jet*) branchJet->At(i);
      if( (jet->PT < 30) || (fabs(jet->Eta) > 3) ) continue;
      njets++;
      HT = HT + jet->PT;
      Jets.push_back(jet);
      if( jet->BTag%2 == 1 ) { nbjets++; bJets.push_back(jet); }
      else if( jet->BTag == 2 ) { ncjets++; }
    }
    njets = Jets.size(); nbjets = bJets.size();
 
    if ( njets < jcut ) continue;   // S3
    s3++;
    if ( njets >= jcut && isttbb) ttbbs3++;
    if ( njets >= jcut && isttbj) ttbjs3++;
    if ( njets >= jcut && isttcc) ttccs3++;
    if ( njets >= jcut && isttlf) ttlfs3++;
 
 
    if ( nbjets < bcut ) continue;   // S4
    s4++;
    if ( nbjets >= bcut && isttbb) ttbbs4++;
    if ( nbjets >= bcut && isttbj) ttbjs4++;
    if ( nbjets >= bcut && isttcc) ttccs4++;
    if ( nbjets >= bcut && isttlf) ttlfs4++;
 
    //MET
    TLorentzVector nu;
    if( branchMissingET->GetEntriesFast() > 0){
      met = (MissingET*) branchMissingET->At(0);
      MET_px = met->MET * cos( met->Phi );
      MET_py = met->MET * sin( met->Phi );
      MET_met = met->MET; MET_eta = met->Eta; MET_phi = met->Phi;
      //cout << "px " << MET_px <<" py "<< MET_py<<" MET "<<met->MET<<endl;
      nu.SetPxPyPzE( MET_px, MET_py, 0, met->MET );
    }
 
    //Lepton 4-vector ( only for lep+jet )
    TLorentzVector lep;
    if ( nElectron == 1 && nMuon == 0 )  lep = Electrons[0]->P4();
    else if ( nMuon == 1 && nElectron == 0 ) lep = Muons[0]->P4();
    TLorentzVector Wl = lep + nu;
 
    // Select two bjets with minimum dR and fill the dnn ntuples
    TLorentzVector RecoAddJets[2];
    float mbb = 999; float dRbb = 999;
    for(int b1 = 0; b1 < bJets.size()-1; b1++){
      for(int b2 = b1+1; b2 < bJets.size(); b2++){
        float tmp_mbb = (bJets[b1]->P4() + bJets[b2]->P4()).M();
        float tmp_dRbb = bJets[b1]->P4().DeltaR(bJets[b2]->P4());
        //if(abs(tmp_mbb-125.25)<abs(mbb-125.25)) {
        if(tmp_dRbb < dRbb) {
           dRbb = tmp_dRbb;
           mbb = tmp_mbb; 
           RecoAddJets[0] = bJets[b1]->P4();
           RecoAddJets[1] = bJets[b2]->P4();
        }
      }
    }

    //Jet Combination
    TLorentzVector Wj;
    float tmpWjM = 9999;
    for(int j1 = 0; j1 < njets-1; j1++) {
      TLorentzVector jet1 = Jets[j1]->P4();
      if(jet1.Pt()==RecoAddJets[0].Pt() || jet1.Pt()==RecoAddJets[1].Pt()) continue;
      for(int j2 = j1+1; j2 < njets; j2++) {
        TLorentzVector jet2 = Jets[j2]->P4();
        if(jet2.Pt()==RecoAddJets[0].Pt() || jet2.Pt()==RecoAddJets[1].Pt()) continue;

        TLorentzVector tmpWj = jet1 + jet2;
        if( abs(tmpWj.M() - 80.377 ) < tmpWjM ) {
          Wj = tmpWj;
          tmpWjM = tmpWj.M() - 80.377;
        }
      }
    }//Jet combi
 
    // Fill the tree ntuples
    Jet_pt1   = Jets[0]->P4().Pt();                     Jet_pt3   = Jets[2]->P4().Pt();
    Jet_eta1  = Jets[0]->P4().Eta();                    Jet_eta3  = Jets[2]->P4().Eta();
    Jet_phi1  = Jets[0]->P4().Phi();                    Jet_phi3  = Jets[2]->P4().Phi();
    Jet_e1    = Jets[0]->P4().E();                      Jet_e3    = Jets[2]->P4().E();
    Jet1_btag = Jets[0]->BTag;                          Jet3_btag = Jets[2]->BTag;
    Jet_pt2   = Jets[1]->P4().Pt();                     Jet_pt4   = Jets[3]->P4().Pt();
    Jet_eta2  = Jets[1]->P4().Eta();                    Jet_eta4  = Jets[3]->P4().Eta();
    Jet_phi2  = Jets[1]->P4().Phi();                    Jet_phi4  = Jets[3]->P4().Phi();
    Jet_e2    = Jets[1]->P4().E();                      Jet_e4    = Jets[3]->P4().E();
    Jet2_btag = Jets[1]->BTag;                          Jet4_btag = Jets[3]->BTag;

    Lepton_pt  = lep.Pt();
    Lepton_eta = lep.Eta();
    Lepton_phi = lep.Phi();
    Lepton_e   = lep.E();
    
    bjet1_pt  = bJets[0]->P4().Pt();                    bjet2_pt  = bJets[1]->P4().Pt();
    bjet1_eta = bJets[0]->P4().Eta();                   bjet2_eta = bJets[1]->P4().Eta();
    bjet1_phi = bJets[0]->P4().Phi();                   bjet2_phi = bJets[1]->P4().Phi();
    bjet1_e   = bJets[0]->P4().E();                     bjet2_e   = bJets[1]->P4().E();


    //selected bjet 1 and 2
    selbjet1_pt  = RecoAddJets[0].Pt();                 selbjet2_pt  = RecoAddJets[1].Pt();
    selbjet1_eta = RecoAddJets[0].Eta();                selbjet2_eta = RecoAddJets[1].Eta();
    selbjet1_phi = RecoAddJets[0].Phi();                selbjet2_phi = RecoAddJets[1].Phi();
    selbjet1_e   = RecoAddJets[0].E();                  selbjet2_e   = RecoAddJets[1].E();

    bbdR   = RecoAddJets[0].DeltaR(RecoAddJets[1]);
    bbdEta = abs(RecoAddJets[0].Eta()-RecoAddJets[1].Eta());
    bbdPhi = RecoAddJets[0].DeltaPhi(RecoAddJets[1]);
    bbPt   = (RecoAddJets[0]+RecoAddJets[1]).Pt();
    bbEta  = (RecoAddJets[0]+RecoAddJets[1]).Eta();
    bbPhi  = (RecoAddJets[0]+RecoAddJets[1]).Phi();
    bbMass = (RecoAddJets[0]+RecoAddJets[1]).M();
    bbHt   = RecoAddJets[0].Pt()+RecoAddJets[1].Pt();
    bbMt   = (RecoAddJets[0]+RecoAddJets[1]).Mt();
 
    nubbdR   = (RecoAddJets[0]+RecoAddJets[1]).DeltaR(nu);
    nubbdEta = abs((RecoAddJets[0]+RecoAddJets[1]).Eta()-nu.Eta());
    nubbdPhi = (RecoAddJets[0]+RecoAddJets[1]).DeltaPhi(nu);
    nubbPt   = (RecoAddJets[0]+RecoAddJets[1]+nu).Pt();
    nubbEta  = (RecoAddJets[0]+RecoAddJets[1]+nu).Eta();
    nubbPhi  = (RecoAddJets[0]+RecoAddJets[1]+nu).Phi();
    nubbMass = (RecoAddJets[0]+RecoAddJets[1]+nu).M();
    nubbHt   = (RecoAddJets[0]+RecoAddJets[1]).Pt()+nu.Pt();
    nubbMt   = (RecoAddJets[0]+RecoAddJets[1]+nu).Mt();
 
    nub1dR   = RecoAddJets[0].DeltaR(nu);               nub2dR   = RecoAddJets[1].DeltaR(nu);
    nub1dEta = abs(RecoAddJets[0].Eta()-nu.Eta());      nub2dEta = abs(RecoAddJets[1].Eta()-nu.Eta());
    nub1dPhi = RecoAddJets[0].DeltaPhi(nu);             nub2dPhi = RecoAddJets[1].DeltaPhi(nu);
    nub1Pt   = (RecoAddJets[0]+nu).Pt();                nub2Pt   = (RecoAddJets[1]+nu).Pt();
    nub1Eta  = (RecoAddJets[0]+nu).Eta();               nub2Eta  = (RecoAddJets[1]+nu).Eta();
    nub1Phi  = (RecoAddJets[0]+nu).Phi();               nub2Phi  = (RecoAddJets[1]+nu).Phi();
    nub1Mass = (RecoAddJets[0]+nu).M();                 nub2Mass = (RecoAddJets[1]+nu).M();
    nub1Ht   = RecoAddJets[0].Pt()+nu.Pt();             nub2Ht   = RecoAddJets[1].Pt()+nu.Pt();
    nub1Mt   = (RecoAddJets[0]+nu).Mt();                nub2Mt   = (RecoAddJets[1]+nu).Mt();
 
    lbbdR   = (RecoAddJets[0]+RecoAddJets[1]).DeltaR(lep);
    lbbdEta = abs((RecoAddJets[0]+RecoAddJets[1]).Eta()-lep.Eta());
    lbbdPhi = (RecoAddJets[0]+RecoAddJets[1]).DeltaPhi(lep);
    lbbPt   = (RecoAddJets[0]+RecoAddJets[1]+lep).Pt();
    lbbEta  = (RecoAddJets[0]+RecoAddJets[1]+lep).Eta();
    lbbPhi  = (RecoAddJets[0]+RecoAddJets[1]+lep).Phi();
    lbbMass = (RecoAddJets[0]+RecoAddJets[1]+lep).M();
    lbbHt   = (RecoAddJets[0]+RecoAddJets[1]).Pt()+lep.Pt();
    lbbMt   = (RecoAddJets[0]+RecoAddJets[1]+lep).Mt();
 
    lb1dR   = RecoAddJets[0].DeltaR(lep);               lb2dR   = RecoAddJets[1].DeltaR(lep);
    lb1dEta = abs(RecoAddJets[0].Eta()-lep.Eta());      lb2dEta = abs(RecoAddJets[1].Eta()-lep.Eta());
    lb1dPhi = RecoAddJets[0].DeltaPhi(lep);             lb2dPhi = RecoAddJets[1].DeltaPhi(lep);
    lb1Pt   = (RecoAddJets[0]+lep).Pt();                lb2Pt   = (RecoAddJets[1]+lep).Pt();
    lb1Eta  = (RecoAddJets[0]+lep).Eta();               lb2Eta  = (RecoAddJets[1]+lep).Eta();
    lb1Phi  = (RecoAddJets[0]+lep).Phi();               lb2Phi  = (RecoAddJets[1]+lep).Phi();
    lb1Mass = (RecoAddJets[0]+lep).M();                 lb2Mass = (RecoAddJets[1]+lep).M();
    lb1Ht   = RecoAddJets[0].Pt()+lep.Pt();             lb2Ht   = RecoAddJets[1].Pt()+lep.Pt();
    lb1Mt   = (RecoAddJets[0]+lep).Mt();                lb2Mt   = (RecoAddJets[1]+lep).Mt();

    Wjb1dR   = RecoAddJets[0].DeltaR(Wj);               Wjb2dR   = RecoAddJets[1].DeltaR(Wj);
    Wjb1dEta = abs(RecoAddJets[0].Eta()-Wj.Eta());      Wjb2dEta = abs(RecoAddJets[1].Eta()-Wj.Eta());
    Wjb1dPhi = RecoAddJets[0].DeltaPhi(Wj);             Wjb2dPhi = RecoAddJets[1].DeltaPhi(Wj);
    Wjb1Pt   = (RecoAddJets[0]+Wj).Pt();                Wjb2Pt   = (RecoAddJets[1]+Wj).Pt();
    Wjb1Eta  = (RecoAddJets[0]+Wj).Eta();               Wjb2Eta  = (RecoAddJets[1]+Wj).Eta();
    Wjb1Phi  = (RecoAddJets[0]+Wj).Phi();               Wjb2Phi  = (RecoAddJets[1]+Wj).Phi();
    Wjb1Mass = (RecoAddJets[0]+Wj).M();                 Wjb2Mass = (RecoAddJets[1]+Wj).M();
    Wjb1Ht   = RecoAddJets[0].Pt()+Wj.Pt();             Wjb2Ht   = RecoAddJets[1].Pt()+Wj.Pt();
    Wjb1Mt   = (RecoAddJets[0]+Wj).Mt();                Wjb2Mt   = (RecoAddJets[1]+Wj).Mt();

    Wlb1dR   = RecoAddJets[0].DeltaR(Wl);               Wlb2dR   = RecoAddJets[1].DeltaR(Wl);
    Wlb1dEta = abs(RecoAddJets[0].Eta()-Wl.Eta());      Wlb2dEta = abs(RecoAddJets[1].Eta()-Wl.Eta());
    Wlb1dPhi = RecoAddJets[0].DeltaPhi(Wl);             Wlb2dPhi = RecoAddJets[1].DeltaPhi(Wl);
    Wlb1Pt   = (RecoAddJets[0]+Wl).Pt();                Wlb2Pt   = (RecoAddJets[1]+Wl).Pt();
    Wlb1Eta  = (RecoAddJets[0]+Wl).Eta();               Wlb2Eta  = (RecoAddJets[1]+Wl).Eta();
    Wlb1Phi  = (RecoAddJets[0]+Wl).Phi();               Wlb2Phi  = (RecoAddJets[1]+Wl).Phi();
    Wlb1Mass = (RecoAddJets[0]+Wl).M();                 Wlb2Mass = (RecoAddJets[1]+Wl).M();
    Wlb1Ht   = RecoAddJets[0].Pt()+Wl.Pt();             Wlb2Ht   = RecoAddJets[1].Pt()+Wl.Pt();
    Wlb1Mt   = (RecoAddJets[0]+Wl).Mt();                Wlb2Mt   = (RecoAddJets[1]+Wl).Mt();
 
    dnn_tree->Fill();
 
    ++numberOfSelectedEvents;
 
 
  }
  hist_selection->SetBinContent(1,numberOfEntries);
  hist_selection->SetBinContent(2,s1);
  hist_selection->SetBinContent(3,s2);
  hist_selection->SetBinContent(4,s3);
  hist_selection->SetBinContent(5,s4);
 
  cout<<"Event Info : jet >= "<<jcut<<" bjet >= "<<bcut<<endl;
  cout << "Total number of selected events = " << numberOfSelectedEvents << endl;
  float accept1 = (float) s1 / (float) entry; 
  float accept2 = (float) s2 / (float) entry;
  float accept3 = (float) s3 / (float) entry;
  float accept4 = (float) s4 / (float) entry;
  cout << "Entries "<<numberOfEntries<<endl;
  cout << "Acceptance1 (S1/Entry) = "<<accept1<<" ( "<<s1<<" )"<<endl;
  cout << "Acceptance2 (S2/Entry) = "<<accept2<<" ( "<<s2<<" )"<<endl;
  cout << "Acceptance3 (S3/Entry) = "<<accept3<<" ( "<<s3<<" )"<<endl;
  cout << "Acceptance4 (S4/Entry) = "<<accept4<<" ( "<<s4<<" )"<<endl;
 
  cout << "category1 ttbb(category1/Entry) = "<<nttbb/(float) s1<<" ( "<<nttbb<<" )"<<endl;
  cout << "Acceptance2 (ttbbS2/nttbb) = "<<ttbbs2/(float)nttbb<<" ( "<<ttbbs2<<" )"<<endl;
  cout << "Acceptance3 (ttbbS3/nttbb) = "<<ttbbs3/(float)nttbb<<" ( "<<ttbbs3<<" )"<<endl;
  cout << "Acceptance4 (ttbbS4/nttbb) = "<<ttbbs4/(float)nttbb<<" ( "<<ttbbs4<<" )"<<endl;
 
  cout << "category2 ttbj(category2/Entry) = "<<nttbj/(float) s1<<" ( "<<nttbj<<" )"<<endl;
  cout << "Acceptance2 (ttbjS2/nttbj) = "<<ttbjs2/(float)nttbj<<" ( "<<ttbjs2<<" )"<<endl;
  cout << "Acceptance3 (ttbjS3/nttbj) = "<<ttbjs3/(float)nttbj<<" ( "<<ttbjs3<<" )"<<endl;
  cout << "Acceptance4 (ttbjS4/nttbj) = "<<ttbjs4/(float)nttbj<<" ( "<<ttbjs4<<" )"<<endl;
 
  cout << "category3 ttcc(category3/Entry) = "<<nttcc/(float) s1<<" ( "<<nttcc<<" )"<<endl;
  cout << "Acceptance2 (ttccS2/nttcc) = "<<ttccs2/(float)nttcc<<" ( "<<ttccs2<<" )"<<endl;
  cout << "Acceptance3 (ttccS3/nttcc) = "<<ttccs3/(float)nttcc<<" ( "<<ttccs3<<" )"<<endl;
  cout << "Acceptance4 (ttccS4/nttcc) = "<<ttccs4/(float)nttcc<<" ( "<<ttccs4<<" )"<<endl;
 
  cout << "category4 ttlf(category4/Entry) = "<<nttlf/(float) s1<<" ( "<<nttlf<<" )"<<endl;
  cout << "Acceptance2 (ttlfS2/nttlf) = "<<ttlfs2/(float)nttlf<<" ( "<<ttlfs2<<" )"<<endl;
  cout << "Acceptance3 (ttlfS3/nttlf) = "<<ttlfs3/(float)nttlf<<" ( "<<ttlfs3<<" )"<<endl;
  cout << "Acceptance4 (ttlfS4/nttlf) = "<<ttlfs4/(float)nttlf<<" ( "<<ttlfs4<<" )"<<endl;
 
 
  fout->Write();
  fout->Close();
}
