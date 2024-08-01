# Denoising Point Clouds in Latent Space via Graph Convolution and Invertible Neural Network

by Aihua Mao, Biao Yan, Ying He.

## Introduction

Official code of our CVPR 2024 paper. [[Paper & Supplement]](https://openaccess.thecvf.com/content/CVPR2024/html/Mao_Denoising_Point_Clouds_in_Latent_Space_via_Graph_Convolution_and_CVPR_2024_paper.html)

This work introduces a framework for point cloud denoising by incorporating invertible neural network and graph convolution.

## Environment

First clone the code of this repo:

```bash
git clone --recursive https://github.com/yanbiao1/PD-LTS
```

Then other settings can be configured manually.

### Manual configuration

The code is implemented with CUDA 11.5, Python 3.8, PyTorch 1.11.0 Other require libraries:
Other require libraries:

- pytorch-lightning==2.0.0 (for training)
- [knn_cuda](https://github.com/unlimblue/KNN_CUDA)
- [point-cloud-utils](https://github.com/fwilliams/point-cloud-utils) (for evaluation)
- [torch-cluster](https://github.com/rusty1s/pytorch_cluster) (for denoising)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d) (for denoising)
- [kaolin](https://github.com/NVIDIAGameWorks/kaolin) (for training)
  
## Datasets

All training and evaluation data can be downloaded from repo of [score-denoise](https://github.com/luost26/score-denoise) and [DMRDenoise](https://github.com/luost26/DMRDenoise/).
After downloading, place the extracted files into `data` directory as list in [here](data/.gitkeep).

We include [light pretrained model](product/ckpt/Denoiseflow-light-FBM.ckpt) and  [heavy pretrained model](product/ckpt/Denoiseflow-heavy-FBM.ckpt) in this repo.

## Training & Denosing & Evaluation

Train the model as followings:

```bash
# train on PUSet, see train_deflow_score.py for tuning parameters

# train light-version model
python models/model_light/train_deflow_score.py

# train heavy-version model
python models/model_heavy/train_deflow_score.py

```
Denoising a single point cloud as followings:

```bash
python models/model_light/denoise.py \
    --input=path/to/input.xyz \
    --output=path/to/output.xyz \
    --patch_size=1024 --niters=1 --ckpt=product/ckpt/Denoiseflow-light-FBM.ckpt

python models/model_heavy/denoise.py \
    --input=path/to/input.xyz \
    --output=path/to/output.xyz \
    --patch_size=1024 --niters=1 --ckpt=product/ckpt/Denoiseflow-heavy-FBM.ckpt
```
Denoising point clouds in directory as followings:

```
python models/model_light/denoise.py \
    --input=path/to/input_directory \
    --output=path/to/output_directory \
    --patch_size=1024 --niters=1 --ckpt=product/ckpt/Denoiseflow-light-FBM.ckpt

python models/model_heavy/denoise.py \
    --input=path/to/input_directory \
    --output=path/to/output_directory \
    --patch_size=1024 --niters=1 --ckpt=product/ckpt/Denoiseflow-heavy-FBM.ckpt

```

Evaluation:

```bash
python eval/eval2.py \
    --pred_dir=path/to/evaluation/directory \
    --off_dir=path/to/off_mesh/directory \
    --gt_dir=path/to/ground_truth/directory \
    --csv=path/to/evaluation/directory/result.csv
```

Denoising point clouds in directory as followings:

```bash
python models/deflow/denoise.py \
    --input=path/to/input_directory \
    --output=path/to/output_directory \
    --patch_size=1024 --niters=1 --ckpt=product/ckpt/Denoiseflow-light-FBM.ckpt
```

Evaluation:

```bash
# build executable to evaluate uniform, see build.sh for detail
bash eval/uniformity/build.sh
# evaluate all xyz files in a directory
python eval/eval3.py \
    --pred_dir=path/to/evaluation/directory \
    --off_dir=path/to/off_mesh/directory \
    --gt_dir=path/to/ground_truth/directory \
    --csv=path/to/evaluation/directory/result.csv
```
Reproduce Paper Results:

```bash
# PUSet dataset, 10K Points, light-version
python models/model_light/denoise.py \
--input=data/ScoreDenoise/examples/PUNet_10000_poisson_0.01 \
--output=evaluation/model_light/PU_10000_n0.01_i1 \
--patch_size=1024 --seed_k=5 --niters=1 --ckpt=product/ckpt/Denoiseflow-light-FBM.ckpt

python models/model_light/denoise.py \
--input=data/ScoreDenoise/examples/PUNet_10000_poisson_0.02 \
--output=evaluation/model_light/PU_10000_n0.02_i1 \
--patch_size=1024 --seed_k=5 --niters=1 --ckpt=product/ckpt/Denoiseflow-light-FBM.ckpt

python models/model_light/denoise.py \
--input=data/ScoreDenoise/examples/PUNet_10000_poisson_0.025 \
--output=evaluation/model_light/PU_10000_n0.025_i1 \
--patch_size=1024 --seed_k=5 --niters=1 --ckpt=product/ckpt/Denoiseflow-light-FBM.ckpt

# PUSet dataset, 50K Points
python models/model_light/denoise.py \
--input=data/ScoreDenoise/examples/PUNet_50000_poisson_0.01 \
--output=evaluation/model_light/PU_50000_n0.01_i1 \
--patch_size=1024 --seed_k=5 --niters=1 --ckpt=product/ckpt/Denoiseflow-light-FBM.ckpt

python models/model_light/denoise.py \
--input=data/ScoreDenoise/examples/PUNet_50000_poisson_0.02 \
--output=evaluation/model_light/PU_50000_n0.02_i1 \
--patch_size=1024 --seed_k=5 --niters=1 --ckpt=product/ckpt/Denoiseflow-light-FBM.ckpt

python models/model_light/denoise.py \
--input=data/ScoreDenoise/examples/PUNet_50000_poisson_0.025 \
--output=evaluation/model_light/PU_50000_n0.025_i1 \
--patch_size=1024 --seed_k=5 --niters=1 --ckpt=product/ckpt/Denoiseflow-light-FBM.ckpt
```

## Citation

If this work is useful for your research, please consider citing:

```bibtex
@inproceedings{mao2024denoising,
  title={Denoising Point Clouds in Latent Space via Graph Convolution and Invertible Neural Network},
  author={Mao, Aihua and Yan, Biao and Ma, Zijing and He, Ying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5768--5777},
  year={2024}
}
```
