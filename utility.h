
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

using floats =  ROOT::VecOps::RVec<float>;
using floatsVec =  ROOT::VecOps::RVec<ROOT::VecOps::RVec<float>>;
using doubles =  ROOT::VecOps::RVec<double>;
using doublesVec =  ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>>;
using ints =  ROOT::VecOps::RVec<int>;
using bools = ROOT::VecOps::RVec<bool>;
using uchars = ROOT::VecOps::RVec<unsigned char>;
using strings = ROOT::VecOps::RVec<std::string>;

using FourVector = ROOT::Math::PtEtaPhiMVector;
using FourVectorVec = std::vector<FourVector>;
using FourVectorRVec = ROOT::VecOps::RVec<FourVector>;

bool isFromTop (ints pid, ints m1, ints m2, int idx, int motherPID=6){
    int mother = -1;
    if ( m1[idx] < 0 && m2[idx] < 0 ) return false;
    if ( (m1[idx] >= 0 && m2[idx] < 0) || (m1[idx] == m2[idx]) ) mother = m1[idx];
    else if ( m1[idx] < 0 && m2[idx] >= 0 ) mother = m2[idx];
    else{
        if ( abs(pid[m1[idx]]) == motherPID || abs(pid[m2[idx]]) == motherPID ) return true;
        else return (isFromTop(pid, m1, m2, m1[idx], motherPID) || isFromTop(pid, m1, m2, m2[idx], motherPID));
    }
    if ( abs(pid[mother]) == motherPID ) return true;
    else return isFromTop(pid, m1, m2, mother, motherPID);
} 

ints SelectAddQuark(ints pid, ints m1, ints m2, ints d1, ints d2){
    ints out;
    for (int i=0; i<int(pid.size()); i++){
        if (abs(pid[i]) < 1 || abs(pid[i]) > 6) out.push_back(0);
        else if (pid[i] == pid[d1[i]] || pid[i] == pid[d2[i]]) out.push_back(0); // particle should be last
        else if (isFromTop(pid, m1, m2, i)) out.push_back(0);
        else out.push_back(1);
    }
    return out;   
}

ints isLast (ints pid, ints d1, ints d2){
    ints out;
    for (int i=0; i<int(pid.size()); i++){
        if ((d1[i]>=0 && pid[i]==pid[d1[i]]) || (d2[i]>=0 && pid[i]==pid[d2[i]])) out.push_back(0);
        else out.push_back(1);
    }
    return out;
}

int findLastIdx(int idx, ints pid, ints d1, ints d2){
    while(true){
        if (d1[idx] < 0 && d2[idx] < 0) return idx;
        if (d1[idx] >= 0 && pid[d1[idx]] == pid[idx]) idx = d1[idx];
        else if(d2[idx] >= 0 && pid[d2[idx]] == pid[idx]) idx = d2[idx];
        else return idx;
    }

}

ints FinalGenPart_idx(ints pid, ints m1, ints m2, ints d1, ints d2, ints top, ints higgs){
    ints out;
    int top1=-1; int bFromTop1=-1; int lepFromTop1=-1;                  //leptonic top
    int top2=-1; int bFromTop2=-1; int q1FromTop2=-1; int q2FromTop2=-1; //hadronic top
    int b1FromHiggs=-1; int b2FromHiggs=-1;

    int b_idx, w_idx;
    //std::cout << "FinalPart Functions " << std::endl;

    for (int i=0; i<int(pid.size()); i++){
        if (top[i] != 0){
            if (abs(pid[d1[i]]) == 5 && abs(pid[d2[i]]) == 24){
               b_idx = d1[i]; w_idx = findLastIdx(d2[i], pid, d1, d2);
            }
            else if (abs(pid[d1[i]]) == 24 && abs(pid[d2[i]]) == 5){
               b_idx = d2[i]; w_idx = findLastIdx(d1[i], pid, d1, d2);
            }
            else std::cout << "d1 pid: " << pid[d1[i]] << " d2 pid: " << pid[d2[i]] << std::endl;
            if (abs(pid[d1[w_idx]]) < 10){  //hadronic top
                top2 = i;
                //std::cout << "b: " << b_idx << " q1: " << d1[w_idx] << " q2: " << d2[w_idx] << std::endl;
                //std::cout << "b: " << pid[b_idx] << " q1: " << pid[d1[w_idx]] << " q2: " << pid[d2[w_idx]] << std::endl;
                bFromTop2 = findLastIdx(b_idx, pid, d1, d2);
                q1FromTop2 = findLastIdx(d1[w_idx], pid, d1, d2);
                q2FromTop2 = findLastIdx(d2[w_idx], pid, d1, d2);
            } else{
                top1 = i;
                //std::cout << "b: " << b_idx << " pid: " << pid[b_idx];
                bFromTop1 = findLastIdx(b_idx, pid, d1, d2);
                if (abs(pid[d1[w_idx]]) % 2 == 1) {
                    lepFromTop1 = findLastIdx(d1[w_idx], pid, d1, d2);
                    //std::cout << " lep: " << d1[w_idx] << " pid: " << pid[d1[w_idx]] << std::endl;
                }
                else {
                    lepFromTop1 = findLastIdx(d2[w_idx], pid, d1, d2);
                    //std::cout << " lep: " << d2[w_idx] << " pid: " << pid[d2[w_idx]] << std::endl;
                }
            }
        }
        else if (higgs[i] != 0){
            b1FromHiggs = findLastIdx(d1[i], pid, d1, d2);
            b2FromHiggs = findLastIdx(d2[i], pid, d1, d2);
        }
    }
    if (bFromTop1 < 0 || lepFromTop1 < 0) std::cout << "Top 1 is not founded!!!!!!!!!!!!" << std::endl;
    if (bFromTop2 < 0 || q1FromTop2 < 0 || q2FromTop2 < 0) std::cout << "Top 2 is not founded!!!!!!!!!!!!" << std::endl;
    if (b1FromHiggs < 0 || b2FromHiggs < 0) std::cout << "Higgs is not founded!!!!!!!!!!!!" << std::endl;

    out.push_back(bFromTop1);
    out.push_back(lepFromTop1);
    out.push_back(bFromTop2);
    out.push_back(q1FromTop2);
    out.push_back(q2FromTop2);
    out.push_back(b1FromHiggs);
    out.push_back(b2FromHiggs);
    out.push_back(top1);
    out.push_back(top2);
    return out;
}


ints make_binary(ints idx, int size){
    ints out;
    for (int i=0; i<size; i++){
        int tag = 0;
        for (int j=0; j<idx.size(); j++){
            if (idx[j] == i) tag=1;
        }
        out.push_back(tag);
    }
    return out;
}

int dRMatching_idx(int idx, float drmax, floats pt1, floats eta1, floats phi1, floats m1, floats pt2, floats eta2, floats phi2, floats m2){
    auto tmp1 = TLorentzVector();
    auto tmp2 = TLorentzVector();
    if (idx < 0) return -1;
    tmp1.SetPtEtaPhiM(pt1[idx], eta1[idx], phi1[idx], m1[idx]);
    int matched_idx = -1; float mindR = drmax;
    for (int j=0; j<int(pt2.size()); j++){
        if (pt2[j] == m2[j]) m2[j]=0;
        tmp2.SetPtEtaPhiM(pt2[j], eta2[j], phi2[j], m2[j]);
        if (tmp1.DeltaR(tmp2) < mindR) {
            matched_idx = j;
            mindR = tmp1.DeltaR(tmp2);
        }
    }
    if (mindR > drmax) return -1;
    return matched_idx;
}
 

ints dRMatching(ints idx, floats pt1, floats eta1, floats phi1, floats m1, floats pt2, floats eta2, floats phi2, floats m2){
    ints out;
    auto tmp1 = TLorentzVector();
    auto tmp2 = TLorentzVector();
    for (int i=0; i<int(pt1.size()); i++){
        if (idx[i] == 0) continue;
        int matched_idx = dRMatching_idx(i, 0.4, pt1, eta1,phi1, m1, pt2, eta2, phi2, m2);
        if (matched_idx < 0) continue;
        out.push_back(matched_idx);
    }
    return make_binary(out, int(pt2.size()));
} 

floats GetE(floats pt, floats eta, floats phi){
    floats out;
    for (int i=0; i<int(pt.size()); i++){
        auto tmp = TLorentzVector();
        tmp.SetPtEtaPhiM(pt[i], eta[i], phi[i], 0);
        out.push_back(tmp.E());
    }
    return out;
}

floats HiggsReco(float pt1, float eta1, float phi1, float m1, float pt2, float eta2, float phi2, float m2){
    floats out;
    auto tmp1 = TLorentzVector(); tmp1.SetPtEtaPhiM(pt1, eta1, phi1, m1);
    auto tmp2 = TLorentzVector(); tmp2.SetPtEtaPhiM(pt2, eta2, phi2, m2);
    out.push_back((tmp1+tmp2).Pt());
    out.push_back((tmp1+tmp2).Eta());
    out.push_back((tmp1+tmp2).Phi());
    out.push_back((tmp1+tmp2).M());
    out.push_back(tmp1.DeltaR(tmp2));
    return out;
}

floats HadTopReco(float pt1, float eta1, float phi1, float m1, float pt2, float eta2, float phi2, float m2, float pt3, float eta3, float phi3, float m3){
    floats out;
    auto tmp1 = TLorentzVector(); tmp1.SetPtEtaPhiM(pt1, eta1, phi1, m1); //b
    auto tmp2 = TLorentzVector(); tmp2.SetPtEtaPhiM(pt2, eta2, phi2, m2); //q1 from W
    auto tmp3 = TLorentzVector(); tmp3.SetPtEtaPhiM(pt3, eta3, phi3, m3); //q2 from W

    // W boson
    out.push_back((tmp2+tmp3).Pt());
    out.push_back((tmp2+tmp3).Eta());
    out.push_back((tmp2+tmp3).Phi());
    out.push_back((tmp2+tmp3).M());
    out.push_back(tmp2.DeltaR(tmp3));

    // Hadronic Top
    out.push_back((tmp1+(tmp2+tmp3)).Pt());
    out.push_back((tmp1+(tmp2+tmp3)).Eta());
    out.push_back((tmp1+(tmp2+tmp3)).Phi());
    out.push_back((tmp1+(tmp2+tmp3)).M());
    out.push_back(tmp1.DeltaR(tmp2+tmp3));
    return out;
}
