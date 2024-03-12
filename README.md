# Denoising Point Clouds in Latent Space via Graph Convolution and Invertible Neural Network

by Aihua Mao, Biao Yan and Ying He.

## Introduction

## Environment

First clone the code of this repo:

```bash
git clone --recursive https://github.com/yanbiao1/PD-LTS
```

### Manual configuration

### Additional configuration

If you want to train the network, you also need to build the kernel of PytorchEMD like followings:

```bash
cd metric/PytorchEMD/
python setup.py install --user
#cp build/lib.linux-x86_64-3.8/emd_cuda.cpython-38m-x86_64-linux-gnu.so .
cp build/lib.linux-x86_64-3.8/emd.cpython-38-x86_64-linux-gnu.so .
```

## Datasets

## Training & Denosing & Evaluation

## Citation

If this work is useful for your research, please consider citing:





