
import torch
import h5py as h5
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset


class MNISTDataset(Dataset):

    def __init__(self, rootdir, split, noise_shape, classes=None):
        super(MNISTDataset, self).__init__()

        assert split in ['train', 'valid', 'test']
        if split == 'valid':
            split = 'test'  # MNIST does not contain valid split

        with h5.File(rootdir, 'r') as h5f:
            if classes is None or classes == 'all':
                self.data   = h5f[f'{split}_data'][:]   # [B, N, 2]
                self.labels = h5f[f'{split}_label'][:]  # [B,]
            else:
                assert classes in list(range(10)), "Invalid class in MNIST"
                data   = h5f[f'{split}_data']
                labels = h5f[f'{split}_label']
                index_mask = np.equal(labels, np.array([classes], dtype=np.int32))
                self.data   = data[:][index_mask, :, :]
                self.labels = labels[index_mask]

        self.noise_shape = noise_shape

        # print(np.max(self.data), np.min(self.data))

    def __getitem__(self, index: int):
        clean_pts = self.data[index]  # [N 2]
        # label = self.labels[index]  # scalar

        noise = MNISTDataset.gen_noise(self.noise_shape, 0.0, 1.0)
        rand_idx = np.random.randint(0, 1024, self.noise_shape[0])
        noise_pts = np.copy(clean_pts)
        noise_pts[rand_idx] = noise

        noise_label = np.ones((clean_pts.shape[0],), dtype=np.float32)
        noise_label[rand_idx] = 0.0

        return clean_pts.transpose(), noise_pts.transpose(), noise_label

    def gen_noise(shape, min: float, max: float):
        return np.random.rand(*shape) * (max - min) + min

    def __len__(self):
        return len(self.labels)


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super(MNISTDataModule, self).__init__()
        self.cfg = cfg

    def train_dataloader(self):
        dataset = MNISTDataset(self.cfg.rootdir, 'train', self.cfg.noise_shape, self.cfg.classes)
        return DataLoader(dataset, self.cfg.batch_size, shuffle=True, pin_memory=True, num_workers=16, drop_last=True)

    def val_dataloader(self):
        dataset = MNISTDataset(self.cfg.rootdir, 'valid', self.cfg.noise_shape, self.cfg.classes)
        return DataLoader(dataset, self.cfg.batch_size, shuffle=False, pin_memory=False, num_workers=16, drop_last=True)


if __name__ == "__main__":

    from omegaconf import OmegaConf
    
    cfg = OmegaConf.create({
        'rootdir' : 'data/MNIST/mnist2d-pointcloud.h5',
        'batch_size': 32,  # 32
        'noise_shape': (32, 2),
        'classes': 'all',
    })
    datamodule = MNISTDataModule(cfg)
    dataloader = datamodule.train_dataloader()

    import matplotlib.pyplot as plt

    for clean, noise, labels in dataloader:
        fig = plt.figure(figsize=(5, 40))

        for i in range(1, 4 + 1):
            ax = plt.subplot(4, 1, i)

            x, y = noise[i].transpose(0, 1)
            label = labels[i]
            scatter = plt.scatter(x, y)

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            scatter.axes.invert_yaxis()
        plt.show()

        exit()

#     for pts, labels in dataloader:
#         fig = plt.figure(figsize=(50, 10))
# 
#         partial, missing = crop_points(pts, num_crop_points=512, crop_method='min-x')
# 
#         for i in range(1, 8 + 1):
# 
#             p_x, p_y = partial[i].transpose(0, 1)
#             m_x, m_y = missing[i].transpose(0, 1)
# 
#             ax1 = plt.subplot(2, 8, i)
#             scatter1 = ax1.scatter(p_x, p_y)
#             ax1.set_xlim(0.0, 1.0)
#             ax1.set_ylim(0.0, 1.0)
#             scatter1.axes.invert_yaxis()
# 
#             ax2 = plt.subplot(2, 8, i + 8)
#             scatter2 = ax2.scatter(m_x, m_y)
#             ax2.set_xlim(0.0, 1.0)
#             ax2.set_ylim(0.0, 1.0)
#             scatter2.axes.invert_yaxis()
#             
#         plt.show()
#         exit()
