import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from torchvision import transforms

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from pytorch3d.ops import knn_points
from dataset.scoredenoise.transforms import standard_train_transforms
from dataset.scoredenoise.transforms import NormalizeUnitSphere, AddNoise, RandomScale, RandomRotate


class PointCloudDataset(Dataset):

    def __init__(self, root, dataset, split, resolution, transform=None):
        super().__init__()
        self.pcl_dir = os.path.join(root, dataset, 'pointclouds', split, resolution)
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        for fn in os.listdir(self.pcl_dir):
            if fn[-3:] != 'xyz':
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)
            pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])

        print(f'[INFO] Loaded dataset {dataset} - {resolution}')

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            'pcl_clean': self.pointclouds[idx].clone(),
            'name': self.pointcloud_names[idx]
        }
        if self.transform is not None:
            data = self.transform(data)
        return data


# def make_patches_for_pcl_pair(pcl_A, pcl_B, patch_size, num_patches, ratio):
#     """
#     Args:
#         pcl_A:  The first point cloud, (N, 3).
#         pcl_B:  The second point cloud, (rN, 3).
#         patch_size:   Patch size M.
#         num_patches:  Number of patches P.
#         ratio:    Ratio r.
#     Returns:
#         (P, M, 3), (P, rM, 3)
#     """
#     N = pcl_A.size(0)
#     seed_idx = torch.randperm(N)[:num_patches]  # (P, )
#     seed_pnts = pcl_A[seed_idx].unsqueeze(0)  # (1, P, 3)
#     _, _, pat_A = knn_points(seed_pnts, pcl_A.unsqueeze(0), K=patch_size, return_nn=True)
#     pat_A = pat_A[0]  # (P, M, 3)
#     _, _, pat_B = knn_points(seed_pnts, pcl_B.unsqueeze(0), K=int(ratio * patch_size), return_nn=True)
#     pat_B = pat_B[0]
#     return pat_A, pat_B, seed_pnts, seed_idx

def make_patches_for_pcl_pair(pcl_A, pcl_B, patch_size, num_patches, ratio):
    """
    Args:
        pcl_A:  The first point cloud, (N, 3).
        pcl_B:  The second point cloud, (rN, 3).
        patch_size:   Patch size M.
        num_patches:  Number of patches P.
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    N = pcl_A.size(0)
    seed_idx = torch.randperm(N)[:num_patches]   # (P, )
    seed_pnts = pcl_A[seed_idx].unsqueeze(0)   # (1, P, 3)
    _, idx_A, pat_A = knn_points(seed_pnts, pcl_A.unsqueeze(0), K=patch_size, return_nn=True)
    pat_A = pat_A[0]    # (P, M, 3)
    idx_A = idx_A[0][0]    # (1, 1024)
    pat_B = pcl_B[idx_A].unsqueeze(0)

    return pat_A, pat_B, seed_pnts, seed_idx

def make_patches_for_pcl(pcl, patch_size, num_patches):
    """
    Args:
        pcl:  The first point cloud, (N, 3).
        patch_size:   Patch size M.
        num_patches:  Number of patches P.
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    N = pcl.size(0)
    seed_idx = torch.randperm(N)[:num_patches]  # (P, )
    seed_pnts = pcl[seed_idx].unsqueeze(0)  # (1, P, 3)
    _, _, pat = knn_points(seed_pnts, pcl.unsqueeze(0), K=patch_size, return_nn=True)
    return pat[0], seed_pnts  # (P, M, 3)

class PairedPatchDataset(Dataset):

    def __init__(self, datasets, patch_ratio, on_the_fly=True, patch_size=1000, num_patches=1000, transform=None):
        super().__init__()
        self.datasets = datasets
        self.len_datasets = sum([len(dset) for dset in datasets])
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.on_the_fly = on_the_fly
        self.transform = transform
        self.patches = []

    def __len__(self):
        if not self.on_the_fly:
            return len(self.patches)
        else:
            return self.len_datasets * self.num_patches

    def __getitem__(self, idx):
        if self.on_the_fly:
            pcl_dset = random.choice(self.datasets)
            pcl_data = pcl_dset[idx % len(pcl_dset)]

            pat_noisy, pat_clean, seed_pts, seed_idx = make_patches_for_pcl_pair(
                pcl_data['pcl_noisy'],
                pcl_data['pcl_clean'],
                patch_size=self.patch_size,
                num_patches=1,
                ratio=self.patch_ratio
            )

            pat_std = pcl_data['noise_std']
            scale = pcl_data['scale']

            data = {
                'pcl_noisy': pat_noisy[0],
                'pcl_clean': pat_clean[0],
                'seed_pnts': seed_pts[0],
                'pcl_std': pat_std,
                'scale': scale
            }
        else:
            data = {
                'pcl_noisy': self.patches[idx][0].clone(),
                'pcl_clean': self.patches[idx][1].clone(),
            }

        return data


class ScoreDenoiseDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super(ScoreDenoiseDataModule, self).__init__()
        self.cfg = cfg

    def train_dataloader(self):
        # transform = standard_train_transforms(noise_std_min=self.cfg.noise_min, noise_std_max=self.cfg.noise_max, rotate=self.cfg.aug_rotate)
        transforms = [
            NormalizeUnitSphere(),
            AddNoise(self.cfg.noise_min, self.cfg.noise_max),
            RandomScale([0.8, 1.2]),
        ]
        if self.cfg.aug_rotate:
            transforms += [
                RandomRotate(axis=0),
                RandomRotate(axis=1),
                RandomRotate(axis=2),
            ]
        transforms = Compose(transforms)  # 将transform这些函数组合起来

        pc_datasets = [
            PointCloudDataset(root=self.cfg.dataset_root, dataset=self.cfg.dataset, split='train', resolution=resl,
                              transform=transforms)
            for resl in self.cfg.resolutions
        ]

        noise_tran = Compose([AddNoise(self.cfg.noise_min, self.cfg.noise_max)])
        train_dset = PairedPatchDataset(datasets=pc_datasets, patch_size=self.cfg.patch_size,
                                        num_patches=self.cfg.num_patches, patch_ratio=1.0, on_the_fly=True,
                                        transform=None)

        return DataLoader(train_dset, batch_size=self.cfg.train_batch_size, num_workers=self.cfg.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        transform = standard_train_transforms(noise_std_min=self.cfg.val_noise, noise_std_max=self.cfg.val_noise,
                                              rotate=False, scale_d=0.0)
        val_dset = PointCloudDataset(root=self.cfg.dataset_root, dataset=self.cfg.dataset, split='test',
                                     resolution=self.cfg.resolutions[0], transform=transform)

        return DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
