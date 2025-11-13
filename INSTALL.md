# Installation

This repository is built in PyTorch 1.8.1 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/LINBOTANG123/CoNAD-CoilWise_NoiseAdaptive_Denoiser.git
cd CoNAD-CoilWise_NoiseAdaptive_Denoiser
```

2. Make conda environment
```
conda create -n pytorch113 python=3.7
conda activate pytorch113
```

3. Install dependencies
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips nibabel

```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```

