# Denoising Point Clouds in Latent Space via Graph Convolution and Invertible Neural Network

by Aihua Mao, Biao Yan and Ying He.

## Introduction

## Environment

First clone the code of this repo:

```bash
git clone --recursive https://github.com/yanbiao1/PD-LTS
```

### Manual configuration

The code is implemented with CUDA 11.5, Python 3.8, PyTorch 1.11.
Other require libraries:

- pytorch-lightning==1.5.3 (for training)
- [knn_cuda](https://github.com/unlimblue/KNN_CUDA)
- [point-cloud-utils](https://github.com/fwilliams/point-cloud-utils) (for evaluation)
- [torch-cluster](https://github.com/rusty1s/pytorch_cluster) (for denoising)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d) (for denoising)
- [kaolin](https://github.com/NVIDIAGameWorks/kaolin) (for training)

### Docker configuration

If you are familiar with Docker, you can use provided [Dockerfile](docker/Dockerfile) to configure all setting automatically.

### Additional configuration

If you want to train the network, you also need to build the kernel of PytorchEMD like followings:

```bash
cd metric/PytorchEMD/
python setup.py install --user
#cp build/lib.linux-x86_64-3.8/emd_cuda.cpython-38m-x86_64-linux-gnu.so .
cp build/lib.linux-x86_64-3.8/emd.cpython-38-x86_64-linux-gnu.so .
```

## Datasets

All training and evaluation data can be downloaded from repo of [score-denoise](https://github.com/luost26/score-denoise) and [DMRDenoise](https://github.com/luost26/DMRDenoise/).
After downloading, place the extracted files into `data` directory as list in [here](data/.gitkeep).

## Training & Denosing & Evaluation

Train the model as followings:

```bash
# train on PUSet, see train_deflow_score.py for tuning parameters
python models/deflow/train_deflow_score.py


python models/deMflow29/train_deflow_score.py
# train on DMRSet, see train_deflow_dmr.py for tuning parameters
python models/deflow/train_deflow_dmr.py

```
```bash
python models/deMflow/denoise.py \
    --input=data/ScoreDenoise/examples/PUNet_10000_poisson_0.025 \
    --output=evaluation/deflow/PUNet_10000_poisson_0.025 \
    --patch_size=1024 --niters=1 --ckpt=pretrain/pdflow-score-LCC.pt
```

Denoising point clouds in directory as followings:

```bash
python models/deflow/denoise.py \
    --input=path/to/input_directory \
    --output=path/to/output_directory \
    --patch_size=1024 --niters=1 --ckpt=pretrain/pdflow-score-LCC.pt
```



## Citation

If this work is useful for your research, please consider citing:





