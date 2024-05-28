# ttbar classification with SNN on Loihi neuromorphic chip

## Installation

   ```bash
   git clone https://github.com/yeonsu108/ttbar-classification.git

   cd ttbar-classification
   conda env create -f env.yaml

   conda activate 23snn
   ```

## Usage
1. ama_ttbar.C
  ```bash
  # input file should be delphes root file
  root -l -b -q 'ana_ttbar.C("/filepath/infile.root", "./filepath/outfile.root", njet, nbjet)' 
  ```
2. uproot_hist.py
  ```bash
  python uproot_hist.py /indir_path/
  ```
3. train.py
  ```bash
  python train.py /indir_path/
  ```
4. jupyter notebook
  ipynb files should be opened on jupyter notebook
  ```bash
  jupyter notebook
  ```
