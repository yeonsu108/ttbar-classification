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
  double m1 = p->M1; double m2 = p->M2;
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

void ana_ttbar(const char *inputFile, const char *outputFile, int jcut, int bcut, int no_file=1){
  //jcut is for the number of jets cut and bcut is for the number of b-jets cut
  gSystem->Load("libDelphes");
 
  // Create chain of root trees
  TChain chain("Delphes");
  chain.Add(inputFile);
 
  TString filename = inputFile;
  
  // Switch for single lepton or dilepton
  bool isdilepton = false;
  if( filename.Contains("di") == true ){
    isdilepton = true;
    cout<<"Dilepton Channel"<<endl;
  }
  else cout<<"Single Lepton Channel"<<endl;

  //Output
  TFile *fout = TFile::Open(outputFile,"RECREATE");
  fout->cd();
 
  //DNN variables
  int signal, event, category;
  int njets, nbjets;
  double addbjet1_pt, addbjet1_eta, addbjet1_phi, addbjet1_e;
  double addbjet2_pt, addbjet2_eta, addbjet2_phi, addbjet2_e;
  double selbjet1_pt, selbjet1_eta, selbjet1_phi, selbjet1_e;
  double selbjet2_pt, selbjet2_eta, selbjet2_phi, selbjet2_e;
  double bbdR, bbdEta, bbdPhi, bbPt, bbEta, bbPhi, bbMass, bbHt, bbMt;
  double nubbdR, nubbdEta, nubbdPhi, nubbPt, nubbEta, nubbPhi, nubbMass, nubbHt, nubbMt;
  double lbbdR, lbbdEta, lbbdPhi, lbbPt, lbbEta, lbbPhi, lbbMass, lbbHt, lbbMt;
  double Wjb1dR, Wjb1dEta, Wjb1dPhi, Wjb1Pt, Wjb1Eta, Wjb1Phi, Wjb1Mass, Wjb1Ht, Wjb1Mt;
  double Wjb2dR, Wjb2dEta, Wjb2dPhi, Wjb2Pt, Wjb2Eta, Wjb2Phi, Wjb2Mass, Wjb2Ht, Wjb2Mt;
  double nub1dR, nub1dEta, nub1dPhi, nub1Pt, nub1Eta, nub1Phi, nub1Mass, nub1Ht, nub1Mt;
  double nub2dR, nub2dEta, nub2dPhi, nub2Pt, nub2Eta, nub2Phi, nub2Mass, nub2Ht, nub2Mt;
  double lb1dR, lb1dEta, lb1dPhi, lb1Pt, lb1Eta, lb1Phi, lb1Mass, lb1Ht, lb1Mt;
  double lb2dR, lb2dEta, lb2dPhi, lb2Pt, lb2Eta, lb2Phi, lb2Mass, lb2Ht, lb2Mt;
 
  //KLFitter variables
  unsigned char lepton_is_e, lepton_is_mu;
  vector<float> jet_pt; vector<float> jet_eta; vector<float> jet_phi; vector<float> jet_e;
  vector<float> jet_btag_weight; vector<char> jet_has_btag;
 
  //Tree variables ( minimum deltaR )
  double bjet1_pt, bjet1_eta, bjet1_phi, bjet1_e;
  double bjet2_pt, bjet2_eta, bjet2_phi, bjet2_e;
  double bjet3_pt, bjet3_eta, bjet3_phi, bjet3_e;
  double bjet4_pt, bjet4_eta, bjet4_phi, bjet4_e;
 
  unsigned short nMuon, nElectron, nLepton;
  double Jet_pt1, Jet_eta1, Jet_phi1, Jet_e1;
  double Jet_pt2, Jet_eta2, Jet_phi2, Jet_e2;
  double Jet_pt3, Jet_eta3, Jet_phi3, Jet_e3;
  double Jet_pt4, Jet_eta4, Jet_phi4, Jet_e4;
  double Electron1_pt, Electron1_eta, Electron1_phi, Electron1_e;
  double Electron2_pt, Electron2_eta, Electron2_phi, Electron2_e;
  double Muon1_pt, Muon1_eta, Muon1_phi, Muon1_e;
  double Muon2_pt, Muon2_eta, Muon2_phi, Muon2_e;
  double Lepton_pt, Lepton_eta, Lepton_phi, Lepton_e;
  double MET_px, MET_py;
  int MATCHED;
  
  // Selected Events (Cut Flow)
  int s1 = 0; int s2 = 0; int s3 = 0; int s4 = 0;
  int ttbbs2 = 0; int ttbbs3 = 0; int ttbbs4 = 0;
  int ttbjs2 = 0; int ttbjs3 = 0; int ttbjs4 = 0;
  int ttccs2 = 0; int ttccs3 = 0; int ttccs4 = 0;
  int ttlfs2 = 0; int ttlfs3 = 0; int ttlfs4 = 0;
 
  int nttbb = 0; int nttbj = 0; int nttcc = 0; int nttlf = 0;
 
  //Tree for Deep learning input 
  TTree * dnn_tree = new TTree( "dnn_input", "tree for dnn");
  dnn_tree->Branch("signal",&signal,"signal/i");
  dnn_tree->Branch("event",&event,"event/i");
  dnn_tree->Branch("category",&category,"category/i");
  dnn_tree->Branch("njets",&njets,"njets/i");
  dnn_tree->Branch("nbjets",&nbjets,"nbjets/i"); 
  dnn_tree->Branch("addbjet1_pt",&addbjet1_pt,"addbjet1_pt/d");
  dnn_tree->Branch("addbjet1_eta",&addbjet1_eta,"addbjet1_eta/d");
  dnn_tree->Branch("addbjet1_phi",&addbjet1_phi,"addbjet1_phi/d");
  dnn_tree->Branch("addbjet1_e",&addbjet1_e,"addbjet1_e/d");
  dnn_tree->Branch("addbjet2_pt",&addbjet2_pt,"addbjet2_pt/d");
  dnn_tree->Branch("addbjet2_eta",&addbjet2_eta,"addbjet2_eta/d");
  dnn_tree->Branch("addbjet2_phi",&addbjet2_phi,"addbjet2_phi/d");
  dnn_tree->Branch("addbjet2_e",&addbjet2_e,"addbjet2_e/d");
  
  dnn_tree->Branch("selbjet1_pt" ,&selbjet1_pt ,"selbjet1_pt/d");
  dnn_tree->Branch("selbjet1_eta",&selbjet1_eta,"selbjet1_eta/d");
  dnn_tree->Branch("selbjet1_phi",&selbjet1_phi,"selbjet1_phi/d");
  dnn_tree->Branch("selbjet1_e"  ,&selbjet1_e  ,"selbjet1_e/d");
  dnn_tree->Branch("selbjet2_pt" ,&selbjet2_pt ,"selbjet2_pt/d");
  dnn_tree->Branch("selbjet2_eta",&selbjet2_eta,"selbjet2_eta/d");
  dnn_tree->Branch("selbjet2_phi",&selbjet2_phi,"selbjet2_phi/d");
  dnn_tree->Branch("selbjet2_e"  ,&selbjet2_e  ,"selbjet2_e/d");
 
  dnn_tree->Branch("bbdR",&bbdR,"bbdR/d");
  dnn_tree->Branch("bbdEta",&bbdEta,"bbdEta/d");
  dnn_tree->Branch("bbdPhi",&bbdPhi,"bbdPhi/d");
  dnn_tree->Branch("bbPt",&bbPt,"bbPt/d");
  dnn_tree->Branch("bbEta",&bbEta,"bbEta/d");
  dnn_tree->Branch("bbPhi",&bbPhi,"bbPhi/d");
  dnn_tree->Branch("bbMass",&bbMass,"bbMass/d");
  dnn_tree->Branch("bbHt",&bbHt,"bbHt/d");
  dnn_tree->Branch("bbMt",&bbMt,"bbMt/d");
 
  dnn_tree->Branch("nubbdR",&nubbdR,"nubbdR/d");
  dnn_tree->Branch("nubbdEta",&nubbdEta,"nubbdEta/d");
  dnn_tree->Branch("nubbdPhi",&nubbdPhi,"nubbdPhi/d");
  dnn_tree->Branch("nubbPt",&nubbPt,"nubbPt/d");
  dnn_tree->Branch("nubbEta",&nubbEta,"nubbEta/d");
  dnn_tree->Branch("nubbPhi",&nubbPhi,"nubbPhi/d");
  dnn_tree->Branch("nubbMass",&nubbMass,"nubbMass/d");
  dnn_tree->Branch("nubbHt",&nubbHt,"nubbHt/d");
  dnn_tree->Branch("nubbMt",&nubbMt,"nubbMt/d");
 
  dnn_tree->Branch("nub1dR",&nub1dR,"nub1dR/d");
  dnn_tree->Branch("nub1dEta",&nub1dEta,"nub1dEta/d");
  dnn_tree->Branch("nub1dPhi",&nub1dPhi,"nub1dPhi/d");
  dnn_tree->Branch("nub1Pt",&nub1Pt,"nub1Pt/d");
  dnn_tree->Branch("nub1Eta",&nub1Eta,"nub1Eta/d");
  dnn_tree->Branch("nub1Phi",&nub1Phi,"nub1Phi/d");
  dnn_tree->Branch("nub1Mass",&nub1Mass,"nub1Mass/d");
  dnn_tree->Branch("nub1Ht",&nub1Ht,"nub1Ht/d");
  dnn_tree->Branch("nub1Mt",&nub1Mt,"nub1Mt/d");
  dnn_tree->Branch("nub2dR",&nub2dR,"nub2dR/d");
  dnn_tree->Branch("nub2dEta",&nub2dEta,"nub2dEta/d");
  dnn_tree->Branch("nub2dPhi",&nub2dPhi,"nub2dPhi/d");
  dnn_tree->Branch("nub2Pt",&nub2Pt,"nub2Pt/d");
  dnn_tree->Branch("nub2Eta",&nub2Eta,"nub2Eta/d");
  dnn_tree->Branch("nub2Phi",&nub2Phi,"nub2Phi/d");
  dnn_tree->Branch("nub2Mass",&nub2Mass,"nub2Mass/d");
  dnn_tree->Branch("nub2Ht",&nub2Ht,"nub2Ht/d");
  dnn_tree->Branch("nub2Mt",&nub2Mt,"nub2Mt/d");
 
  dnn_tree->Branch("lbbdR",&lbbdR,"lbbdR/d");
  dnn_tree->Branch("lbbdEta",&lbbdEta,"lbbdEta/d");
  dnn_tree->Branch("lbbdPhi",&lbbdPhi,"lbbdPhi/d");
  dnn_tree->Branch("lbbPt",&lbbPt,"lbbPt/d");
  dnn_tree->Branch("lbbEta",&lbbEta,"lbbEta/d");
  dnn_tree->Branch("lbbPhi",&lbbPhi,"lbbPhi/d");
  dnn_tree->Branch("lbbMass",&lbbMass,"lbbMass/d");
  dnn_tree->Branch("lbbHt",&lbbHt,"lbbHt/d");
  dnn_tree->Branch("lbbMt",&lbbMt,"lbbMt/d");
 
  dnn_tree->Branch("lb1dR",&lb1dR,"lb1dR/d");
  dnn_tree->Branch("lb1dEta",&lb1dEta,"lb1dEta/d");
  dnn_tree->Branch("lb1dPhi",&lb1dPhi,"lb1dPhi/d");
  dnn_tree->Branch("lb1Pt",&lb1Pt,"lb1Pt/d");
  dnn_tree->Branch("lb1Eta",&lb1Eta,"lb1Eta/d");
  dnn_tree->Branch("lb1Phi",&lb1Phi,"lb1Phi/d");
  dnn_tree->Branch("lb1Mass",&lb1Mass,"lb1Mass/d");
  dnn_tree->Branch("lb1Ht",&lb1Ht,"lb1Ht/d");
  dnn_tree->Branch("lb1Mt",&lb1Mt,"lb1Mt/d");
  dnn_tree->Branch("lb2dR",&lb2dR,"lb2dR/d");
  dnn_tree->Branch("lb2dEta",&lb2dEta,"lb2dEta/d");
  dnn_tree->Branch("lb2dPhi",&lb2dPhi,"lb2dPhi/d");
  dnn_tree->Branch("lb2Pt",&lb2Pt,"lb2Pt/d");
  dnn_tree->Branch("lb2Eta",&lb2Eta,"lb2Eta/d");
  dnn_tree->Branch("lb2Phi",&lb2Phi,"lb2Phi/d");
  dnn_tree->Branch("lb2Mass",&lb2Mass,"lb2Mass/d");
  dnn_tree->Branch("lb2Ht",&lb2Ht,"lb2Ht/d");
  dnn_tree->Branch("lb2Mt",&lb2Mt,"lb2Mt/d");
 
  dnn_tree->Branch("Wjb1dR",&Wjb1dR,"Wjb1dR/d");
  dnn_tree->Branch("Wjb1dEta",&Wjb1dEta,"Wjb1dEta/d");
  dnn_tree->Branch("Wjb1dPhi",&Wjb1dPhi,"Wjb1dPhi/d");
  dnn_tree->Branch("Wjb1Pt",&Wjb1Pt,"Wjb1Pt/d");
  dnn_tree->Branch("Wjb1Eta",&Wjb1Eta,"Wjb1Eta/d");
  dnn_tree->Branch("Wjb1Phi",&Wjb1Phi,"Wjb1Phi/d");
  dnn_tree->Branch("Wjb1Mass",&Wjb1Mass,"Wjb1Mass/d");
  dnn_tree->Branch("Wjb1Ht",&Wjb1Ht,"Wjb1Ht/d");
  dnn_tree->Branch("Wjb1Mt",&Wjb1Mt,"Wjb1Mt/d");
  dnn_tree->Branch("Wjb2dR",&Wjb2dR,"Wjb2dR/d");
  dnn_tree->Branch("Wjb2dEta",&Wjb2dEta,"Wjb2dEta/d");
  dnn_tree->Branch("Wjb2dPhi",&Wjb2dPhi,"Wjb2dPhi/d");
  dnn_tree->Branch("Wjb2Pt",&Wjb2Pt,"Wjb2Pt/d");
  dnn_tree->Branch("Wjb2Eta",&Wjb2Eta,"Wjb2Eta/d");
  dnn_tree->Branch("Wjb2Phi",&Wjb2Phi,"Wjb2Phi/d");
  dnn_tree->Branch("Wjb2Mass",&Wjb2Mass,"Wjb2Mass/d");
  dnn_tree->Branch("Wjb2Ht",&Wjb2Ht,"Wjb2Ht/d");
  dnn_tree->Branch("Wjb2Mt",&Wjb2Mt,"Wjb2Mt/d");
 
  //Tree for minimum dR analysis
  TTree * tree = new TTree( "tree", "tree for ttbb");
  tree->Branch("nbjets",&nbjets,"nbjets/s");
  dnn_tree->Branch("bjet1_pt",&bjet1_pt,"bjet1_pt/d");
  dnn_tree->Branch("bjet1_eta",&bjet1_eta,"bjet1_eta/d");
  dnn_tree->Branch("bjet1_phi",&bjet1_phi,"bjet1_phi/d");
  dnn_tree->Branch("bjet1_e",&bjet2_e,"bjet1_e/d");
  dnn_tree->Branch("bjet2_pt",&bjet2_pt,"bjet2_pt/d");
  dnn_tree->Branch("bjet2_eta",&bjet2_eta,"bjet2_eta/d");
  dnn_tree->Branch("bjet2_phi",&bjet2_phi,"bjet2_phi/d");
  dnn_tree->Branch("bjet2_e",&bjet2_e,"bjet2_e/d");
  tree->Branch("bjet3_pt",&bjet3_pt,"bjet3_pt/d");
  tree->Branch("bjet3_eta",&bjet3_eta,"bjet3_eta/d");
  tree->Branch("bjet3_phi",&bjet3_phi,"bjet3_phi/d");
  tree->Branch("bjet3_e",&bjet3_e,"bjet3_e/d");
  tree->Branch("bjet4_pt",&bjet4_pt,"bjet4_pt/d");
  tree->Branch("bjet4_eta",&bjet4_eta,"bjet4_eta/d");
  tree->Branch("bjet4_phi",&bjet4_phi,"bjet4_phi/d");
  tree->Branch("bjet4_e",&bjet4_e,"bjet4_e/d");
 
  tree->Branch("njets",&njets,"njets/s");
  dnn_tree->Branch("Jet_pt1",&Jet_pt1,"Jet_pt1/d");
  dnn_tree->Branch("Jet_eta1",&Jet_eta1,"Jet_eta1/d");
  dnn_tree->Branch("Jet_phi1",&Jet_phi1,"Jet_phi1/d");
  dnn_tree->Branch("Jet_e1",&Jet_e1,"Jet_e1/d");
  dnn_tree->Branch("Jet_pt2",&Jet_pt2,"Jet_pt2/d");
  dnn_tree->Branch("Jet_eta2",&Jet_eta2,"Jet_eta2/d");
  dnn_tree->Branch("Jet_phi2",&Jet_phi2,"Jet_phi2/d");
  dnn_tree->Branch("Jet_e2",&Jet_e2,"Jet_e2/d");
  dnn_tree->Branch("Jet_pt3",&Jet_pt3,"Jet_pt3/d");
  dnn_tree->Branch("Jet_eta3",&Jet_eta3,"Jet_eta3/d");
  dnn_tree->Branch("Jet_phi3",&Jet_phi3,"Jet_phi3/d");
  dnn_tree->Branch("Jet_e3",&Jet_e3,"Jet_e3/d");
  dnn_tree->Branch("Jet_pt4",&Jet_pt4,"Jet_pt4/d");
  dnn_tree->Branch("Jet_eta4",&Jet_eta4,"Jet_eta4/d");
  dnn_tree->Branch("Jet_phi4",&Jet_phi4,"Jet_phi4/d");
  dnn_tree->Branch("Jet_e4",&Jet_e4,"Jet_e4/d");
 
  dnn_tree->Branch("nElectron",&nElectron,"nElectron/s");
  tree->Branch("Electron1_pt",&Electron1_pt,"Electron1_pt/d");
  tree->Branch("Electron1_eta",&Electron1_eta,"Electron1_eta/d");
  tree->Branch("Electron1_phi",&Electron1_phi,"Electron1_phi/d");
  tree->Branch("Electron1_e",&Electron1_e,"Electron1_e/d");
  tree->Branch("Electron2_pt",&Electron2_pt,"Electron2_pt/d");
  tree->Branch("Electron2_eta",&Electron2_eta,"Electron2_eta/d");
  tree->Branch("Electron2_phi",&Electron2_phi,"Electron2_phi/d");
  tree->Branch("Electron2_e",&Electron2_e,"Electron2_e/d");
  
  dnn_tree->Branch("nMuon",&nMuon,"nMuon/s");
  tree->Branch("Muon1_pt",&Muon1_pt,"Muon1_pt/d");
  tree->Branch("Muon1_eta",&Muon1_eta,"Muon1_eta/d");
  tree->Branch("Muon1_phi",&Muon1_phi,"Muon1_phi/d");
  tree->Branch("Muon1_e",&Muon1_e,"Muon1_e/d");
  tree->Branch("Muon2_pt",&Muon2_pt,"Muon2_pt/d");
  tree->Branch("Muon2_eta",&Muon2_eta,"Muon2_eta/d");
  tree->Branch("Muon2_phi",&Muon2_phi,"Muon2_phi/d");
  tree->Branch("Muon2_e",&Muon2_e,"Muon2_e/d");
 
  dnn_tree->Branch("MET_px",&MET_px,"MET_px/d");
  dnn_tree->Branch("MET_py",&MET_py,"MET_py/d");
  dnn_tree->Branch("nLepton",&nLepton,"nLepton/s");
  dnn_tree->Branch("Lepton_pt",&Lepton_pt,"Lepton_pt/d");
  dnn_tree->Branch("Lepton_eta",&Lepton_eta,"Lepton_eta/d");
  dnn_tree->Branch("Lepton_phi",&Lepton_phi,"Lepton_phi/d");
  dnn_tree->Branch("Lepton_e",&Lepton_e,"Lepton_e/d");
  tree->Branch("MATCHED",&MATCHED,"MATCHED/i");
  
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
 
  // Book histograms
  TH1 *histnjet = new TH1F("h_njet", "Number of jets", 14, 0.0, 14.0);
  TH1 *histnElectron = new TH1F("h_nElectron", "Number of electrons", 3, 0.0, 3.0);
  TH1 *histnMuon = new TH1F("h_nMuon", "Number of muons", 3, 0.0, 3.0);
  
  TH1 *histnbjet = new TH1F("h_nbjet", "Number of b-jets", 10, 0.0, 10.0);
  TH1 *histMbb = new TH1F("h_mbb", "M_{inv}(b, b)", 30, 0.0, 300.0);
  TH1 *histdRbb = new TH1F("h_dRbb", "dR(b, b)", 20, 0, 4.0);
 
  TH1 *hist_gennbjet = new TH1F("h_gennbjet", "Number of b-jets", 6, 0.0, 6.0);
  TH1 *hist_genMbb = new TH1F("h_genmbb", "M_{inv}(b, b)", 30, 0.0, 300.0);
  TH1 *hist_gendRbb = new TH1F("h_gendRbb", "dR(b, b)", 20, 0, 4.0);
 
  TH1 *hist_matchednbjet = new TH1F("h_matchednbjet", "Number of b-jets", 6, 0.0, 6.0);
  TH1 *hist_matchedMbb = new TH1F("h_matchedmbb", "M_{inv}(b, b)", 30, 0.0, 300.0);
  TH1 *hist_matcheddRbb = new TH1F("h_matcheddRbb", "dR(b, b)", 20, 0, 4.0);
 
  TH1 *hist_selection = new TH1F("h_selection","Selection",4,0.0,4.0);
 
  Int_t numberOfSelectedEvents = 0;
  Int_t numberOfMatchedEvents = 0;
 
  vector<Jet *> nobJets; vector<Jet *> bJets; vector<Jet *> Jets;
  vector<Electron *> Electrons; vector<Muon *> Muons;
 
  TLorentzVector p4[2];
  Jet *jet;
  Electron *electron;
  Muon *muon;
  MissingET *met;
 
  int entry, i;
 
  // Loop over all events
  for(entry = 0; entry < numberOfEntries; ++entry) {
    if(entry%1000 == 0 && entry < 10000) cout << "event number: " << entry << endl;
    else if(entry%10000 == 0) cout<< "event number: " << entry << endl;
    // Load selected branches with data from specified event
    treeReader->ReadEntry(entry);
    
    // Initialization
    Jets.clear();
    Electrons.clear();
    Muons.clear(); 
    bJets.clear();
    nobJets.clear();
    
    // dnn variables
    signal = -1; event = -1; category = -1;
    addbjet1_pt = -9999; addbjet1_eta = -9999; addbjet1_phi = -9999; addbjet1_e = -9999;
    addbjet2_pt = -9999; addbjet2_eta = -9999; addbjet2_phi = -9999; addbjet2_e = -9999;
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
 
    // tree variables // for mindR
    Jet_pt1 = -9999; Jet_eta1 = -9999; Jet_phi1 = -9999; Jet_e1 = -9999;
    Jet_pt2 = -9999; Jet_eta2 = -9999; Jet_phi2 = -9999; Jet_e2 = -9999;
    Jet_pt3 = -9999; Jet_eta3 = -9999; Jet_phi3 = -9999; Jet_e3 = -9999;
    Jet_pt4 = -9999; Jet_eta4 = -9999; Jet_phi4 = -9999; Jet_e4 = -9999;
    Electron1_pt = -9999; Electron1_eta = -9999; Electron1_phi = -9999; Electron1_e = -9999;
    Electron2_pt = -9999; Electron2_eta = -9999; Electron2_phi = -9999; Electron2_e = -9999;
    Muon1_pt = -9999; Muon1_eta = -9999; Muon1_phi = -9999; Muon1_e = -9999;
    Muon2_pt = -9999; Muon2_eta = -9999; Muon2_phi = -9999; Muon2_e = -9999;
    Lepton_pt = -9999; Lepton_eta = -9999; Lepton_phi = -9999; Lepton_e = -9999;
    bjet1_pt = -9999; bjet1_eta = -9999; bjet1_phi = -9999; bjet1_e = -9999;
    bjet2_pt = -9999; bjet2_eta = -9999; bjet2_phi = -9999; bjet2_e = -9999;
    bjet3_pt = -9999; bjet3_eta = -9999; bjet3_phi = -9999; bjet3_e = -9999;
    bjet4_pt = -9999; bjet4_eta = -9999; bjet4_phi = -9999; bjet4_e = -9999;
    MET_px = -9999; MET_py = -9999;
    MATCHED = -1;
 
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
      Jet * matchedJet; double mindR = 9999;
      for( int j = 0 ; j < branchGenJet->GetEntriesFast() ; j++ ){
        Jet * genjet = (Jet*) branchGenJet->At(j);
        if( genjet->P4().Pt() < 20 || genjet->P4().Eta() > 2.5 ) continue; // (gen level) object selection
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

    //Select 2 addbjets, addcjets, addlfjets in descending order of Pt
    TLorentzVector addjet[2]; TLorentzVector addbjet[2]; TLorentzVector addcjet[2]; TLorentzVector addlfjet[2];
    for( i = 0; i < 2; i++){
      if( GenAddJets.size() > i ) addjet[i] = GenAddJets[i]->P4();
      if( GenAddbJets.size() > i ) addbjet[i] = GenAddbJets[i]->P4();
      if( GenAddcJets.size() > i ) addcjet[i] = GenAddcJets[i]->P4();
      if( GenAddlfJets.size() > i ) addlfjet[i] = GenAddlfJets[i]->P4();
    }
 
    bool isttbb=false; bool isttbj=false; bool isttcc=false; bool isttlf=false;
    if( GenAddJets.size() < 2 ) continue; // s1. ttjj event selection
    if( GenAddbJets.size() > 1)      { isttbb=true; category = 0; nttbb++; }
    else if( GenAddbJets.size() > 0) { isttbj=true; category = 1; nttbj++; }
    else if( GenAddcJets.size() > 1) { isttcc=true; category = 2; nttcc++; }
    else                             { isttlf=true; category = 3; nttlf++; }
    s1++;
 
    //Lepton Selection (S2)
    //Electron Selection
    nElectron = 0;
    for(i = 0; i < branchElectron->GetEntries(); ++i){
      electron = (Electron*) branchElectron->At(i);
      if( !isdilepton && ((electron->PT < 30) || (fabs(electron->Eta) > 2.4)) ) continue;
      if( isdilepton && ((electron->PT < 20) || (fabs(electron->Eta) > 2.4)) ) continue;
      nElectron++;
      Electrons.push_back(electron);
    }
    //Muon Selection
    nMuon = 0;
    for(i = 0; i < branchMuon->GetEntries(); ++i){
      muon = (Muon*) branchMuon->At(i);
      if( !isdilepton && ((muon->PT < 30) || (fabs(muon->Eta) > 2.4)) ) continue;
      if( isdilepton && ((muon->PT < 30) || (fabs(muon->Eta) > 2.4)) ) continue;
      nMuon++;
      Muons.push_back(muon);
    }
    nLepton = nElectron + nMuon;
    
    //TODO Lepton selection nLepton >= 1 (or == 1)
    bool leptonSelection = false;
    if( isdilepton ) leptonSelection = (nLepton >= 2);
    else leptonSelection = (nLepton >= 1);
    if (!leptonSelection) continue;   // S2
    s2++;
    if (leptonSelection && isttbb) ttbbs2++;
    if (leptonSelection && isttbj) ttbjs2++;
    if (leptonSelection && isttcc) ttccs2++;
    if (leptonSelection && isttlf) ttlfs2++;
 
    // Jet and b-tag Selections (S3)
    njets = 0;
    nbjets = 0;
    for(i = 0; i < branchJet->GetEntriesFast(); ++i){
      jet = (Jet*) branchJet->At(i);
      if( (jet->PT < 30) || (fabs(jet->Eta) > 2.5) ) continue;
      njets++;
      Jets.push_back(jet);
      if( jet->BTag ) { nbjets++; bJets.push_back(jet); }
      else nobJets.push_back(jet);
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
 
    cout << "cut are all fine" << endl;
 
    //MET
    TLorentzVector nu;
    if( branchMissingET->GetEntriesFast() > 0){
      met = (MissingET*) branchMissingET->At(0);
      MET_px = met->MET * cos( met->Phi );
      MET_py = met->MET * sin( met->Phi );
      //cout << "px " << MET_px <<" py "<< MET_py<<" MET "<<met->MET<<endl;
      nu.SetPxPyPzE( MET_px, MET_py, 0, met->MET );
    }
 
    //Lepton 4-vector ( only for lep+jet )
    TLorentzVector lep;
    if ( nElectron == 1 && nMuon == 0 )  lep = Electrons[0]->P4();
    else if ( nMuon == 1 && nElectron == 0 ) lep = Muons[0]->P4();
    else if ( nMuon >= 1 && nElectron >= 1 ) {
        if ( Electrons[0]->P4().Pt() >= Muons[0]->P4().Pt() ) lep = Electrons[0]->P4();
        else lep = Muons[0]->P4();
    }
 
    // Fill the tree ntuples (minimum dR)
    Jet_pt1  = Jets[0]->P4().Pt();  //Jet_pt1  = nobJets[0]->P4().Pt();  
    Jet_eta1 = Jets[0]->P4().Eta(); //Jet_eta1 = nobJets[0]->P4().Eta();
    Jet_phi1 = Jets[0]->P4().Phi(); //Jet_phi1 = nobJets[0]->P4().Phi();
    Jet_e1   = Jets[0]->P4().E();   //Jet_e1   = nobJets[0]->P4().E();    
    Jet_pt2  = Jets[1]->P4().Pt();  //Jet_pt2  = nobJets[1]->P4().Pt();  
    Jet_eta2 = Jets[1]->P4().Eta(); //Jet_eta2 = nobJets[1]->P4().Eta();
    Jet_phi2 = Jets[1]->P4().Phi(); //Jet_phi2 = nobJets[1]->P4().Phi();
    Jet_e2   = Jets[1]->P4().E();   //Jet_e2   = nobJets[1]->P4().E();    
    Jet_pt3  = Jets[2]->P4().Pt();  //Jet_pt3  = nobJets[2]->P4().Pt();  
    Jet_eta3 = Jets[2]->P4().Eta(); //Jet_eta3 = nobJets[2]->P4().Eta();
    Jet_phi3 = Jets[2]->P4().Phi(); //Jet_phi3 = nobJets[2]->P4().Phi();
    Jet_e3   = Jets[2]->P4().E();   //Jet_e3   = nobJets[2]->P4().E();    
    Jet_pt4  = Jets[3]->P4().Pt();  //Jet_pt4  = nobJets[3]->P4().Pt();  
    Jet_eta4 = Jets[3]->P4().Eta(); //Jet_eta4 = nobJets[3]->P4().Eta();
    Jet_phi4 = Jets[3]->P4().Phi(); //Jet_phi4 = nobJets[3]->P4().Phi();
    Jet_e4   = Jets[3]->P4().E();   //Jet_e4   = nobJets[3]->P4().E();    
    Lepton_pt  = lep.Pt();
    Lepton_eta = lep.Eta();
    Lepton_phi = lep.Phi();
    Lepton_e   = lep.E();
    if( nElectron >= 1){
      Electron1_pt  = Electrons[0]->P4().Pt();
      Electron1_eta = Electrons[0]->P4().Eta();
      Electron1_phi = Electrons[0]->P4().Phi();
      Electron1_e   = Electrons[0]->P4().E();
    }
    if( nMuon >= 1 ){
      Muon1_pt = Muons[0]->P4().Pt();
      Muon1_eta = Muons[0]->P4().Eta();
      Muon1_phi = Muons[0]->P4().Phi();
      Muon1_e = Muons[0]->P4().E();
    }
    if(isdilepton){
      if(nElectron >= 2){
        Electron2_pt = Electrons[1]->P4().Pt();
        Electron2_eta = Electrons[1]->P4().Eta();
        Electron2_phi = Electrons[1]->P4().Phi();
        Electron2_e = Electrons[1]->P4().E();
      }
      if(nMuon >= 2){
        Muon2_pt = Muons[1]->P4().Pt();
        Muon2_eta = Muons[1]->P4().Eta();
        Muon2_phi = Muons[1]->P4().Phi();
        Muon2_e = Muons[1]->P4().E();
      }
    } 
    bjet1_pt  = bJets[0]->P4().Pt();
    bjet1_eta = bJets[0]->P4().Eta();
    bjet1_phi = bJets[0]->P4().Phi();
    bjet1_e   = bJets[0]->P4().E();
    if(bJets.size() >= 2){
      bjet2_pt  = bJets[1]->P4().Pt();
      bjet2_eta = bJets[1]->P4().Eta();
      bjet2_phi = bJets[1]->P4().Phi();
      bjet2_e   = bJets[1]->P4().E();
    }
    if(bJets.size() >=3){
      bjet3_pt  = bJets[2]->P4().Pt();
      bjet3_eta = bJets[2]->P4().Eta();
      bjet3_phi = bJets[2]->P4().Phi();
      bjet3_e   = bJets[2]->P4().E();
    }
    if(bJets.size() >=4){
      bjet4_pt  = bJets[3]->P4().Pt();
      bjet4_eta = bJets[3]->P4().Eta();
      bjet4_phi = bJets[3]->P4().Phi();
      bjet4_e   = bJets[3]->P4().E();
    }
    
    histnjet->Fill( njets );
    histnElectron->Fill( nElectron );
    histnMuon->Fill( nMuon );
 
    //Matched b-jets
    vector<Jet*> MatchedbJets;
    for( int j = 0; j < bJets.size(); j++){
      TLorentzVector recobjet = bJets[j]->P4();
      for(int k = 0 ; k < GenAddbJets.size(); k++){
        TLorentzVector genbjet = GenAddbJets[k]->P4();
        double dR = recobjet.DeltaR( genbjet );
        //cout <<" test dR = " << dR << endl;
        if( dR < 0.4 ) {
          MatchedbJets.push_back( bJets[j] ) ;
          //cout<<entry<<" "<<j<<k<<"MatchedbJet pt : "<< MatchedbJets[MatchedbJets.size()-1]->P4().Pt() << endl;
        }
      }
    }
    //for(int a=0; a < MatchedbJets.size(); a++) cout<<MatchedbJets[a]->P4().Pt()<<endl;
   
    histnbjet->Fill(bJets.size());
    hist_gennbjet->Fill(GenAddbJets.size());
    if( GenAddbJets.size() > 1){
      double gen_Mbb = ( GenAddbJets[0]->P4() + GenAddbJets[1]->P4() ).M();
      double gen_dRbb = GenAddbJets[0]->P4().DeltaR( GenAddbJets[1]->P4() );
      hist_genMbb->Fill( gen_Mbb );
      hist_gendRbb->Fill( gen_dRbb );
    }
    hist_matchednbjet->Fill( MatchedbJets.size() );
    if( MatchedbJets.size() > 1){
      double matched_mbb = ( MatchedbJets[0]->P4() + MatchedbJets[1]->P4() ).M();
      double matched_dRbb = MatchedbJets[0]->P4().DeltaR( MatchedbJets[1]->P4() ); 
      hist_matchedMbb->Fill(matched_mbb);
      hist_matcheddRbb->Fill(matched_dRbb);
    }

    //Jet Combination
    TLorentzVector Wj;
    TLorentzVector tmpWj;
    double tmpWjM = 9999;
    for(int j1 = 0; j1 < njets-1; j1++) {
      if(Jets[j1]->BTag) continue;
      TLorentzVector jet1 = Jets[j1]->P4();
      for(int j2 = j1+1; j2 < njets; j2++) {
        if(Jets[j2]->BTag) continue;
        TLorentzVector jet2 = Jets[j2]->P4();
 
        tmpWj = jet1 + jet2;
        if( abs(tmpWj.M() - 80.2 ) < tmpWjM ) {
          Wj = tmpWj;
          tmpWjM = tmpWj.M() - 80.2;
        }
      }
    }//Jet combi
 
    float mbb = 999;
    float dRbb = 999;
 
    // Select two bjets with minimum dR and fill the dnn ntuples
    TLorentzVector RecoAddJets[2];
    for(int b1 = 0; b1 < bJets.size()-1; b1++){
      for(int b2 = b1+1; b2 < bJets.size(); b2++){
        p4[0] = bJets[b1]->P4();
        p4[1] = bJets[b2]->P4();
 
        float tmp_mbb = ((p4[0]) + (p4[1])).M();
        float tmp_dRbb = p4[0].DeltaR(p4[1]);
        if(tmp_dRbb < dRbb) {
           dRbb = tmp_dRbb;
           mbb = tmp_mbb; 
           RecoAddJets[0] = p4[0];
           RecoAddJets[1] = p4[1];
        }
      }
    }

    //selected bjet 1 and 2
    selbjet1_pt  = RecoAddJets[0].Pt();
    selbjet1_eta = RecoAddJets[0].Eta();
    selbjet1_phi = RecoAddJets[0].Phi();
    selbjet1_e   = RecoAddJets[0].E();
    selbjet2_pt  = RecoAddJets[1].Pt();
    selbjet2_eta = RecoAddJets[1].Eta();
    selbjet2_phi = RecoAddJets[1].Phi();
    selbjet2_e   = RecoAddJets[1].E();
 
    addbjet1_pt  = addbjet[0].Pt();
    addbjet1_eta = addbjet[0].Eta();
    addbjet1_phi = addbjet[0].Phi();
    addbjet1_e   = addbjet[0].E();
    addbjet2_pt  = addbjet[1].Pt();
    addbjet2_eta = addbjet[1].Eta();
    addbjet2_phi = addbjet[1].Phi();
    addbjet2_e   = addbjet[1].E();
 
    bbdR   = dRbb;
    bbdEta = abs(RecoAddJets[0].Eta()-RecoAddJets[1].Eta());
    bbdPhi = RecoAddJets[0].DeltaPhi(RecoAddJets[1]);
    bbPt   = (RecoAddJets[0]+RecoAddJets[1]).Pt();
    bbEta  = (RecoAddJets[0]+RecoAddJets[1]).Eta();
    bbPhi  = (RecoAddJets[0]+RecoAddJets[1]).Phi();
    bbMass = mbb;
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
 
    nub1dR   = RecoAddJets[0].DeltaR(nu);
    nub1dEta = abs(RecoAddJets[0].Eta()-nu.Eta());
    nub1dPhi = RecoAddJets[0].DeltaPhi(nu);
    nub1Pt   = (RecoAddJets[0]+nu).Pt();
    nub1Eta  = (RecoAddJets[0]+nu).Eta();
    nub1Phi  = (RecoAddJets[0]+nu).Phi();
    nub1Mass = (RecoAddJets[0]+nu).M();
    nub1Ht   = RecoAddJets[0].Pt()+nu.Pt();
    nub1Mt   = (RecoAddJets[0]+nu).Mt();
 
    nub2dR   = RecoAddJets[1].DeltaR(nu);
    nub2dEta = abs(RecoAddJets[1].Eta()-nu.Eta());
    nub2dPhi = RecoAddJets[1].DeltaPhi(nu);
    nub2Pt   = (RecoAddJets[1]+nu).Pt();
    nub2Eta  = (RecoAddJets[1]+nu).Eta();
    nub2Phi  = (RecoAddJets[1]+nu).Phi();
    nub2Mass = (RecoAddJets[1]+nu).M();
    nub2Ht   = RecoAddJets[1].Pt()+nu.Pt();
    nub2Mt   = (RecoAddJets[1]+nu).Mt();
 
    lbbdR   = (RecoAddJets[0]+RecoAddJets[1]).DeltaR(lep);
    lbbdEta = abs((RecoAddJets[0]+RecoAddJets[1]).Eta()-lep.Eta());
    lbbdPhi = (RecoAddJets[0]+RecoAddJets[1]).DeltaPhi(lep);
    lbbPt   = (RecoAddJets[0]+RecoAddJets[1]+lep).Pt();
    lbbEta  = (RecoAddJets[0]+RecoAddJets[1]+lep).Eta();
    lbbPhi  = (RecoAddJets[0]+RecoAddJets[1]+lep).Phi();
    lbbMass = (RecoAddJets[0]+RecoAddJets[1]+lep).M();
    lbbHt   = (RecoAddJets[0]+RecoAddJets[1]).Pt()+lep.Pt();
    lbbMt   = (RecoAddJets[0]+RecoAddJets[1]+lep).Mt();
 
    lb1dR   = RecoAddJets[0].DeltaR(lep);
    lb1dEta = abs(RecoAddJets[0].Eta()-lep.Eta());
    lb1dPhi = RecoAddJets[0].DeltaPhi(lep);
    lb1Pt   = (RecoAddJets[0]+lep).Pt();
    lb1Eta  = (RecoAddJets[0]+lep).Eta();
    lb1Phi  = (RecoAddJets[0]+lep).Phi();
    lb1Mass = (RecoAddJets[0]+lep).M();
    lb1Ht   = RecoAddJets[0].Pt()+lep.Pt();
    lb1Mt   = (RecoAddJets[0]+lep).Mt();
    lb2dR   = RecoAddJets[1].DeltaR(lep);
    lb2dEta = abs(RecoAddJets[1].Eta()-lep.Eta());
    lb2dPhi = RecoAddJets[1].DeltaPhi(lep);
    lb2Pt   = (RecoAddJets[1]+lep).Pt();
    lb2Eta  = (RecoAddJets[1]+lep).Eta();
    lb2Phi  = (RecoAddJets[1]+lep).Phi();
    lb2Mass = (RecoAddJets[1]+lep).M();
    lb2Ht   = RecoAddJets[1].Pt()+lep.Pt();
    lb2Mt   = (RecoAddJets[1]+lep).Mt();
 
    Wjb1dR   = RecoAddJets[0].DeltaR(Wj);
    Wjb1dEta = abs(RecoAddJets[0].Eta()-Wj.Eta());
    Wjb1dPhi = RecoAddJets[0].DeltaPhi(Wj);
    Wjb1Pt   = (RecoAddJets[0]+Wj).Pt();
    Wjb1Eta  = (RecoAddJets[0]+Wj).Eta();
    Wjb1Phi  = (RecoAddJets[0]+Wj).Phi();
    Wjb1Mass = (RecoAddJets[0]+Wj).M();
    Wjb1Ht   = RecoAddJets[0].Pt()+Wj.Pt();
    Wjb1Mt   = (RecoAddJets[0]+Wj).Mt();
    Wjb2dR   = RecoAddJets[1].DeltaR(Wj);
    Wjb2dEta = abs(RecoAddJets[1].Eta()-Wj.Eta());
    Wjb2dPhi = RecoAddJets[1].DeltaPhi(Wj);
    Wjb2Pt   = (RecoAddJets[1]+Wj).Pt();
    Wjb2Eta  = (RecoAddJets[1]+Wj).Eta();
    Wjb2Phi  = (RecoAddJets[1]+Wj).Phi();
    Wjb2Mass = (RecoAddJets[1]+Wj).M();
    Wjb2Ht   = RecoAddJets[1].Pt()+Wj.Pt();
    Wjb2Mt   = (RecoAddJets[1]+Wj).Mt();
 
    dnn_tree->Fill();
    
 
    double matched_mbb = ( RecoAddJets[0] + RecoAddJets[1] ).M();
    double matched_dRbb = RecoAddJets[0].DeltaR( RecoAddJets[1] ); 
    hist_matchedMbb->Fill(matched_mbb);
    hist_matcheddRbb->Fill(matched_dRbb);
    
    hist_selection->SetBinContent(1,s1);
    hist_selection->SetBinContent(2,s2);
    hist_selection->SetBinContent(3,s3);
    hist_selection->SetBinContent(4,s4);
 
    bool matched = false;
    bool matched1 = ( RecoAddJets[0].DeltaR( addjet[0] ) < 0.4 ) && ( RecoAddJets[1].DeltaR( addjet[1] ) < 0.4 );
    bool matched2 = ( RecoAddJets[0].DeltaR( addjet[1] ) < 0.4 ) && ( RecoAddJets[1].DeltaR( addjet[0] ) < 0.4 );
    if ( matched1 || matched2 ) matched = true;
    //cout<<"nElectron = "<< nElectron <<" nMuon = "<< nMuon <<" njets = "<< njets <<" nbjets = "<< nbjets << endl;
 
    ++numberOfSelectedEvents;
    if(matched) {
      numberOfMatchedEvents++;
      MATCHED = 1;
    }
    else MATCHED = 0;
 
    histMbb->Fill(mbb);
    histdRbb->Fill(dRbb);
    tree->Fill();
 
  }
 
  cout<<"Event Info : jet >= "<<jcut<<" bjet >= "<<bcut<<endl;
  //cout << "Total number of events = " << numberOfSelectedEvents << endl;
  //cout << "Total number of matched events = " << numberOfMatchedEvents << endl;
  double eff = (double) numberOfMatchedEvents/ (double) numberOfSelectedEvents;
  cout << "Matching eff. = " << eff << " ( "<<numberOfMatchedEvents<<" / "<<numberOfSelectedEvents<<" )"<< endl;
  double accept1 = (double) s1 / (double) entry; 
  double accept2 = (double) s2 / (double) entry;
  double accept3 = (double) s3 / (double) entry;
  double accept4 = (double) s4 / (double) entry;
  cout << "Entries "<<numberOfEntries<<endl;
  cout << "Acceptance1 (S1/Entry) = "<<accept1<<" ( "<<s1<<" )"<<endl;
  cout << "Acceptance2 (S2/Entry) = "<<accept2<<" ( "<<s2<<" )"<<endl;
  cout << "Acceptance3 (S3/Entry) = "<<accept3<<" ( "<<s3<<" )"<<endl;
  cout << "Acceptance4 (S4/Entry) = "<<accept4<<" ( "<<s4<<" )"<<endl;
 
  cout << "category1 ttbb(category1/Entry) = "<<nttbb/(double) s1<<" ( "<<nttbb<<" )"<<endl;
  cout << "Acceptance2 (ttbbS2/nttbb) = "<<ttbbs2/(double)nttbb<<" ( "<<ttbbs2<<" )"<<endl;
  cout << "Acceptance3 (ttbbS3/nttbb) = "<<ttbbs3/(double)nttbb<<" ( "<<ttbbs3<<" )"<<endl;
  cout << "Acceptance4 (ttbbS4/nttbb) = "<<ttbbs4/(double)nttbb<<" ( "<<ttbbs4<<" )"<<endl;
 
  cout << "category2 ttbj(category2/Entry) = "<<nttbj/(double) s1<<" ( "<<nttbj<<" )"<<endl;
  cout << "Acceptance2 (ttbjS2/nttbj) = "<<ttbjs2/(double)nttbj<<" ( "<<ttbjs2<<" )"<<endl;
  cout << "Acceptance3 (ttbjS3/nttbj) = "<<ttbjs3/(double)nttbj<<" ( "<<ttbjs3<<" )"<<endl;
  cout << "Acceptance4 (ttbjS4/nttbj) = "<<ttbjs4/(double)nttbj<<" ( "<<ttbjs4<<" )"<<endl;
 
  cout << "category3 ttcc(category3/Entry) = "<<nttcc/(double) s1<<" ( "<<nttcc<<" )"<<endl;
  cout << "Acceptance2 (ttccS2/nttcc) = "<<ttccs2/(double)nttcc<<" ( "<<ttccs2<<" )"<<endl;
  cout << "Acceptance3 (ttccS3/nttcc) = "<<ttccs3/(double)nttcc<<" ( "<<ttccs3<<" )"<<endl;
  cout << "Acceptance4 (ttccS4/nttcc) = "<<ttccs4/(double)nttcc<<" ( "<<ttccs4<<" )"<<endl;
 
  cout << "category4 ttlf(category4/Entry) = "<<nttlf/(double) s1<<" ( "<<nttlf<<" )"<<endl;
  cout << "Acceptance2 (ttlfS2/nttlf) = "<<ttlfs2/(double)nttlf<<" ( "<<ttlfs2<<" )"<<endl;
  cout << "Acceptance3 (ttlfS3/nttlf) = "<<ttlfs3/(double)nttlf<<" ( "<<ttlfs3<<" )"<<endl;
  cout << "Acceptance4 (ttlfS4/nttlf) = "<<ttlfs4/(double)nttlf<<" ( "<<ttlfs4<<" )"<<endl;
 
 
  fout->Write();
  fout->Close();
}
