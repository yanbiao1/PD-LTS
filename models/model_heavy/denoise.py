import os
import sys
import argparse
import torch
import math
import numpy as np
import torch.nn as nn
from glob import glob
from pathlib import Path
from tqdm import tqdm
from pytorch3d.ops import knn_points
from sklearn.cluster import KMeans
import datetime
sys.path.append(os.getcwd())

from models.model_heavy.deflow import DenoiseFlow, Disentanglement, DenoiseFlowMLP
from dataset.scoredenoise.transforms import NormalizeUnitSphere
from modules.utils.score_utils import farthest_point_sampling, remove_outliers


@torch.no_grad()
def patch_denoise_validation(network, pcl_noisy, patch_size=1000, seed_k=3, down_N=None):
    """
    pcl_noisy:  Input point cloud, [N, 3]
    """
    device = pcl_noisy.device
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be [N, 3].'
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # [1, N, 3]
    K = int(seed_k * N / patch_size)
    seed_pnts, _ = farthest_point_sampling(pcl_noisy, K)  # [1, K, 3]
    _, _, patches = knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)  # [1, K, M, 3]

    seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
    patches = torch.squeeze(patches, dim=0)  # [K, M, 3]
    patches = patches - seed_pnts_1

    # patches = patches.view(-1, patch_size, 3)
    patches_denoised = network(patches)  # [K, M, 3]
    patches_denoised = patches_denoised[-1]
    patches_denoised = patches_denoised + seed_pnts_1


    select_Num = int(patch_size / seed_k + 400)

    patches_denoised = patches_denoised[:, :select_Num].contiguous()  # [K, M, 3] > [K, select_Num, 3]

    download_sample_N = (down_N or N) + 10
    pcl_denoised, _fps_idx = farthest_point_sampling(patches_denoised.view(1, -1, d), download_sample_N)

    if True:
        pcl_denoised = remove_outliers(pcl_denoised, pcl_noisy, num_outliers=10)
    return torch.squeeze(pcl_denoised, dim=0)

@torch.no_grad()
def patch_denoise(network, pcl_noisy, patch_size=1000, seed_k=3, seed_k_alpha=5):
    """
    pcl_noisy:  Input point cloud, [N, 3]
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be [N, 3].'
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # [1, N, 3]
    num_patches = int(seed_k * N / patch_size)
    seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)  # [1, K, 3]
    patch_dists, point_idxs_in_main_pcd, patches = knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)

    patches = patches.view(-1, patch_size, 3)  # (K, M, 3)
    seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
    patches = patches - seed_pnts_1

    # Patch stitching preliminaries
    patch_dists, point_idxs_in_main_pcd = patch_dists[0], point_idxs_in_main_pcd[0]  # (K, M) (K, M)
    patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(1, patch_size)

    all_dists = torch.ones(num_patches, N) / 0
    all_dists = all_dists.cuda()
    all_dists = list(all_dists)
    patch_dists, point_idxs_in_main_pcd = list(patch_dists), list(point_idxs_in_main_pcd)

    for all_dist, patch_id, patch_dist in zip(all_dists, point_idxs_in_main_pcd, patch_dists):
        all_dist[patch_id] = patch_dist

    all_dists = torch.stack(all_dists, dim=0)  # (K, N)
    weights = torch.exp(-1 * all_dists)

    best_weights, best_weights_idx = torch.max(weights, dim=0)  # (N,) (N,) 最大权重、最大权重对应的块号
    patches_denoised = []
    i = 0
    patch_step = int(num_patches / seed_k_alpha)
    assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
    while i < num_patches:
        # print("Processed {:d}/{:d} patches.".format(i, num_patches))
        curr_patches = patches[i:i + patch_step]
        try:
            patches_denoised_temp, _, _ = network[0](curr_patches)  # [K, M, 3]
            patches_denoised_temp, _, _ = network[1](patches_denoised_temp)  # [K, M, 3]
            patches_denoised_temp, _, _ = network[2](patches_denoised_temp)  # [K, M, 3]

        except Exception as e:
            print("=" * 100)
            print(e)
            print("=" * 100)
            print("If this is an Out Of Memory error, Seed_k_alpha might need to be increased to decrease patch_step.")
            print(
                "Additionally, if using multiple args.niters and a PyTorch3D ops, KNN, error arises, Seed_k might need to be increased to sample more patches for inference!")
            print("=" * 100)
            return
        patches_denoised.append(patches_denoised_temp)
        i += patch_step

    patches_denoised = torch.cat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_pnts_1
    pcl_denoised = [patches_denoised[patch][point_idxs_in_main_pcd[patch] == pidx_in_main_pcd]
                    for pidx_in_main_pcd, patch in enumerate(best_weights_idx)]

    pcl_denoised = torch.cat(pcl_denoised, dim=0)
    return pcl_denoised

@torch.no_grad()
def large_patch_denoise_v1(network, pcl, cluster_size, seed=0, device='cpu', verbose=False):
    pcl = pcl.cpu().numpy()

    if verbose:
        print('Running KMeans to construct clusters...')
    n_clusters = math.ceil(pcl.shape[0] / cluster_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(pcl)

    pcl_parts = []
    it = tqdm(range(n_clusters), desc='Denoise Clusters') if verbose else range(n_clusters)

    for i in it:
        pts_idx = kmeans.labels_ == i

        # pcl_part_noisy = torch.FloatTensor(pcl[pts_idx]).to(device)
        pcl_part_noisy = torch.from_numpy(pcl[pts_idx]).to(device)
        pcl_part_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_part_noisy)
        pcl_part_denoised = patch_denoise(network, pcl_part_noisy, seed_k=3)
        pcl_part_denoised = pcl_part_denoised * scale + center
        pcl_parts.append(pcl_part_denoised)

    return torch.cat(pcl_parts, dim=0)


@torch.no_grad()
def large_patch_denoise_v2(network, pcl, cluster_size, seed=0, device='cpu', verbose=False):
    pcl = pcl.cpu().numpy()

    if verbose:
        print('Running KMeans to construct clusters...')
    n_clusters = math.ceil(pcl.shape[0] / cluster_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(pcl)

    pcl_parts = []
    it = tqdm(range(n_clusters), desc='Denoise Clusters') if verbose else range(n_clusters)
    expand_ratio = 1.0

    for i in it:
        pts_idx = kmeans.labels_ == i
        pcl_cluster = pcl[pts_idx]  # [N, 3]
        N_cluster, _ = pcl_cluster.shape

        cluster_center = kmeans.cluster_centers_[i]  # [3,]
        radius_squares = np.sum((pcl_cluster - cluster_center) ** 2, axis=-1)  # [N,]
        radius_square = np.average(np.sort(radius_squares)[-5:])  # scalar

        dist_square = np.sum((pcl - cluster_center) ** 2, axis=-1)  # [N,]
        expand_idx = dist_square < (radius_square * expand_ratio)
        final_pts_idx = np.logical_or(expand_idx, pts_idx)  # 把剩余距离小于半径的点放进块中

        pcl_part_noisy = torch.from_numpy(pcl[final_pts_idx]).to(device)
        pcl_part_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_part_noisy)
        pcl_part_denoised = patch_denoise(network, pcl_part_noisy, seed_k=3, down_N=N_cluster)
        pcl_part_denoised = pcl_part_denoised * scale + center

        pcl_parts.append(pcl_part_denoised)

    return torch.cat(pcl_parts, dim=0)


def denoise_loop(args, network, pcl_raw, seed_k_alpha, iters=None):
    # Normalize
    pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_raw)

    pcl_next = pcl_noisy.to(args.device)
    iters = iters or 1

    for _ in range(iters):
        pcl_next = patch_denoise(network, pcl_next, args.patch_size, args.seed_k, seed_k_alpha)
    pcl_denoised = pcl_next.cpu()
    # Denormalize
    return pcl_denoised * scale + center


def denoise_partition_loop(args, network, pcl_raw):
    N, _ = pcl_raw.shape
    nsplit = N // 10000
    _, fps_idx = farthest_point_sampling(pcl_raw.view(1, N, 3), N)  # [N,]

    pcl_subs = []
    for i in range(nsplit):
        sub_idx = fps_idx[0][i::nsplit]
        pcl_sub_noisy = pcl_raw[sub_idx]  # [N / nsplit, 3]
        pcl_sub_denoised = denoise_loop(args, network, pcl_sub_noisy, iters=None)
        pcl_subs.append(pcl_sub_denoised)
    pcl_denoised = torch.cat(pcl_subs, dim=0)  # [N, 3]

    pcl_denoised, _fps_idx = farthest_point_sampling(pcl_denoised.view(1, N, 3), N)
    pcl_denoised = torch.squeeze(pcl_denoised, dim=0)
    return pcl_denoised


def denoise_pcl(args, ipath, opath, network=None):
    if network is None:
        network = get_denoise_net(args.ckpt).to(args.device)

    pcl_raw = torch.from_numpy(np.loadtxt(ipath, dtype=np.float32))
    seed_k_alpha = pcl_raw.shape[0] / 10000
    pcl_denoised = denoise_loop(args, network, pcl_raw, seed_k_alpha, iters=args.niters)

    np.savetxt(opath, pcl_denoised.numpy(), fmt='%.8f')


def get_denoise_net(ckpt_path):
    if Disentanglement.FBM.name in ckpt_path:
        disentangle = Disentanglement.FBM
    if Disentanglement.LBM.name in ckpt_path:
        disentangle = Disentanglement.LBM
    if Disentanglement.LCC.name in ckpt_path:
        disentangle = Disentanglement.LCC

    if 'MLP' in ckpt_path:
        network = DenoiseFlowMLP(disentangle, pc_channel=3)
    else:
        network = nn.ModuleList()
        for i in range(3):
            network.append(DenoiseFlow(
                disentangle=Disentanglement.FBM,
                pc_channel=3,
                aug_channel=32,
                n_injector=10,
                num_neighbors=32,
                cut_channel=16,
                nflow_module=10,
                coeff=0.98,
                n_lipschitz_iters=None,
                sn_atol=1e-3,
                sn_rtol=1e-3,
                n_power_series=None,
                n_dist='geometric',
                n_samples=1,
                activation_fn='swish',
                n_exact_terms=2,
                neumann_grad=True,
                grad_in_forward=True,
                nhidden=2,
                idim=64,
                densenet=False,
                densenet_depth=3,
                densenet_growth=32,
                learnable_concat=True,
                lip_coeff=0.98,
            )
        )

    network.load_state_dict(torch.load(ckpt_path))
    for i in range(3):
        network[i].init_as_trained_state()
        network[i].eval()
    return network


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--input', type=str, default='path/to/file_or_directory',
                        help='Input file path or input directory')
    parser.add_argument('--output', type=str, default='path/to/file_or_directory',
                        help='Output file path or output directory')
    parser.add_argument('--ckpt', type=str, default='pretrain/deflow.ckpt', help='Path to network checkpoint')
    parser.add_argument('--seed_k', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--cluster_size', type=int, default=10000)
    parser.add_argument('--niters', type=int, default=1)
    parser.add_argument('--first_iter_partition', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    path_input = Path(args.input)
    path_output = Path(args.output)

    if os.path.isdir(path_input) and os.path.exists(path_input):
        if not os.path.exists(path_output):
            os.mkdir(path_output)

        network = get_denoise_net(args.ckpt).to(args.device)
        ipaths = glob(f'{path_input}/*.xyz')  # 返回后缀为xyz的文件

        for ipath in ipaths:
            _, f = os.path.split(ipath)
            opath = path_output / f

            denoise_pcl(args, ipath, opath, network)

    elif os.path.isfile(path_input):
        if '.' in args.output:
            # os.path.isfile(path_output):
            odir, _ = os.path.split(path_output)
        elif os.path.isdir(path_output):
            odir = path_output
            _, ofile = os.path.split(path_input)
            path_output = odir / ofile
        else:
            assert False

        if not os.path.exists(odir):
            os.mkdir(odir)

        if args.verbose:
            print(f"Denoising {path_input}")
        denoise_pcl(args, path_input, path_output, None)
    else:
        assert False, "Invalid input or output path"

    if args.verbose:
        print(f'Finish denoising {args.input}...')
