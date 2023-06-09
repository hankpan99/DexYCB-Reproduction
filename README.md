# DexYCB-Reproduction
`view_sequence.py` `dex_ycb_toolkit` are copied from [dex-ycb-toolkit](https://github.com/NVlabs/dex-ycb-toolkit)

## Installation
### 1. Clone the repo with --recursive and install dependencies:
```
git clone --recursive https://github.com/hankpan99/DexYCB-Reproduction.git
cd DexYCB-reproduction
pip install -r requirements.txt
```
### 2. Install `PyTorch` from [pytorch.org](https://pytorch.org).
### 3. Install manopth
```
cd manopth
pip install -e .
cd ..
```
### 4. Download MANO models
* Download MANO models(mano_v1_2.zip) from [MANO](https://mano.is.tue.mpg.de).
* Unzip and copy the models folder into the `DexYCB-Reproduction/manopth/mano` folder.
## Usage
### 1. Optimize hand poses
```
python runner.py
```
### 2. Visualize the result
```
python view_sequence.py --name 20221001_171108
```
