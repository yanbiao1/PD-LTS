import math

import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.getcwd())

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from dataset.scoredenoise.dataset import ScoreDenoiseDataModule
from models.model_heavy.deflow import DenoiseFlow, Disentanglement, DenoiseFlowMLP
from models.model_heavy.denoise import patch_denoise_validation
from metric.loss import MaskLoss, ConsistencyLoss
from metric.loss import EarthMoverDistance as EMD
from modules.utils.score_utils import chamfer_distance_unit_sphere
from modules.utils.modules import print_progress_log
import models.layers.base as base_layers
import models.layers as layers

ACT_FNS = {
    'softplus': lambda b: nn.Softplus(),
    'elu': lambda b: nn.ELU(inplace=b),
    'swish': lambda b: base_layers.Swish(),
    'LeakyLSwish': lambda b: base_layers.LeakyLSwish(),
    'CLipSwish': lambda b: base_layers.CLipSwish(),
    'ALCLipSiLU': lambda b: base_layers.ALCLipSiLU(),
    'pila': lambda b: base_layers.Pila(),
    'CPila': lambda b: base_layers.CPila(),
    'lcube': lambda b: base_layers.LipschitzCube(),
    'identity': lambda b: base_layers.Identity(),
    'relu': lambda b: base_layers.MyReLU(inplace=b),
    'CReLU': lambda b: base_layers.CReLU(),
}
summaryWriter = SummaryWriter(log_dir="logs/30_32dim3feataug")


# -----------------------------------------------------------------------------------------
class TrainerModule(pl.LightningModule):

    def __init__(self, cfg):
        super(TrainerModule, self).__init__()

        self.disentangle_method = Disentanglement.FBM
        self.network = nn.ModuleList()
        for i in range(3):
            self.network.append(DenoiseFlow(
                disentangle=self.disentangle_method,
                pc_channel=cfg.pc_channel,
                aug_channel=cfg.aug_channel,
                n_injector=cfg.n_injector,
                num_neighbors=cfg.num_neighbors,
                cut_channel=cfg.cut_channel,
                nflow_module=cfg.nflow_module,
                coeff=cfg.coeff,
                n_lipschitz_iters=cfg.n_lipschitz_iters,
                sn_atol=cfg.sn_atol,
                sn_rtol=cfg.sn_rtol,
                n_power_series=cfg.n_power_series,
                n_dist=cfg.n_dist,
                n_samples=cfg.n_samples,
                activation_fn=cfg.activation_fn,
                n_exact_terms=cfg.n_exact_terms,
                neumann_grad=cfg.neumann_grad,
                grad_in_forward=cfg.grad_in_forward,
                nhidden=cfg.nhidden,
                idim=cfg.idim,
                densenet=cfg.densenet,
                densenet_depth=cfg.densenet_depth,
                densenet_growth=cfg.densenet_growth,
                learnable_concat=cfg.learnable_concat,
                lip_coeff=cfg.lip_coeff,
            )
        )
        # self.network = DenoiseFlowMLP(self.disentangle_method)

        self.loss_emd = EMD()
        self.mloss = MaskLoss()
        self.closs = ConsistencyLoss()

        self.epoch = 41
        self.cfg = cfg
        self.validation_step_outputs = []
        self.min_CD = 5.0
        # nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0, norm_type=2)
        self.training_step_outputs = []

    def forward(self, p: Tensor, **kwargs):
        patch_denoised = []
        for i in range(3):
            p, _, _ = self.network[i](p, **kwargs)
            patch_denoised.append(p)
        return patch_denoised

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.sched_patience,
                                                               factor=self.cfg.sched_factor, min_lr=self.cfg.min_lr)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'emd'}}

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.sched_patience, factor=self.cfg.sched_factor, min_lr=self.cfg.min_lr)
        # return { "optimizer": optimizer, 'scheduler': scheduler }
        # return optimizer

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        pcl_noisy, pcl_clean, seed_pnts, pcl_std, scale = batch['pcl_noisy'], batch['pcl_clean'], batch['seed_pnts'], batch['pcl_std'], batch['scale']
        seed_pnts = seed_pnts.repeat(1, pcl_noisy.size(1), 1)
        pcl_noisy = pcl_noisy - seed_pnts
        pcl_clean = pcl_clean - seed_pnts
        pcl_target_lower_noise0 = pcl_clean + torch.randn_like(pcl_clean) * (pcl_std * scale / 5.3).unsqueeze(1).unsqueeze(2)
        pcl_target_lower_noise1 = pcl_clean + torch.randn_like(pcl_clean) * (pcl_std * scale / (5.3 * 5.3)).unsqueeze(1).unsqueeze(2)
        denoised = self(pcl_noisy)
        emd0 = self.loss_emd(denoised[0], pcl_target_lower_noise0)
        emd1 = self.loss_emd(denoised[1], pcl_target_lower_noise1)
        emd2 = self.loss_emd(denoised[2], pcl_clean)
        emd = emd0 + emd1 + emd2

        loss = emd * 0.1
        self.training_step_outputs.append(loss.item())
        self.log('emd', emd.detach().cpu().item() * 0.1, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        loss.backward()
        # return loss
        opt.step()
        opt.zero_grad()
        update_lipschitz(self.network)

    def on_train_epoch_end(self):
        epoch_average = torch.mean(torch.tensor(self.training_step_outputs))
        print('epoch_average_loss:', epoch_average)
        if (self.epoch > 1):
            summaryWriter.add_scalar("loss", epoch_average, self.epoch)

        self.training_step_outputs.clear()

    def on_validation_epoch_start(self):
        update_lipschitz(self.network)

    def validation_step(self, batch, batch_idx):
        pcl_noisy, pcl_clean = batch['pcl_noisy'], batch['pcl_clean']
        pcl_denoised = patch_denoise_validation(self, pcl_noisy.squeeze(), patch_size=1024)  # Fix patch size

        self.validation_step_outputs.append({
            'denoised': pcl_denoised,
            'clean': pcl_clean.squeeze(),
        })
        return {
            'denoised': pcl_denoised,
            'clean': pcl_clean.squeeze(),
        }

    def on_validation_epoch_end(self):
        all_denoise = torch.stack([x['denoised'] for x in self.validation_step_outputs])
        all_clean = torch.stack([x['clean'] for x in self.validation_step_outputs])
        # check_weight_grad(self.network)
        avg_chamfer = chamfer_distance_unit_sphere(all_denoise, all_clean, batch_reduction='mean')[0].item() * 1e4

        extra = []
        # if avg_chamfer < self.min_CD:
        #     self.min_CD = avg_chamfer
        #     save_path = f'runs/ckpt/DenoiseFlow-{self.disentangle_method.name}-scoreset-minCD.ckpt'
        #     torch.save(self.network.state_dict(), save_path)
        #     extra.append('CD')
        self.validation_step_outputs.clear()
        print_progress_log(self.epoch, {'CD': avg_chamfer}, extra=extra)
        # print(self.epoch)
        if (self.epoch > 1):
            summaryWriter.add_scalar("CD", avg_chamfer, self.epoch)
        self.epoch += 1


# -----------------------------------------------------------------------------------------
def model_specific_args():
    parser = ArgumentParser()
    # Network
    parser.add_argument('--net', type=str, default='DenoiseFlow')
    parser.add_argument('--pc_channel', type=int, default=3)
    parser.add_argument('--aug_channel', type=int, default=32)

    parser.add_argument('--n_injector', type=int, default=10)
    parser.add_argument('--num_neighbors', type=int, default=32)
    parser.add_argument('--cut_channel', type=int, default=16)
    parser.add_argument('--nflow_module', type=int, default=10)

    parser.add_argument('--coeff', type=float, default=0.98)
    parser.add_argument('--n_lipschitz_iters', type=int, default=None)
    parser.add_argument('--sn_atol', type=float, default=1e-3)
    parser.add_argument('--sn_rtol', type=float, default=1e-3)
    parser.add_argument('--n_power_series', type=int, default=None)
    parser.add_argument('--n_dist', choices=['geometric', 'poisson'], default='geometric')
    parser.add_argument('--n_samples', type=int, default=1)
    # parser.add_argument('--activation_fn', type=str, choices=ACT_FNS.keys(), default='pila')
    # parser.add_argument('--activation_fn', type=str, choices=ACT_FNS.keys(), default='CLipSwish')
    parser.add_argument('--activation_fn', type=str, choices=ACT_FNS.keys(), default='swish')
    parser.add_argument('--n_exact_terms', type=int, default=2)
    parser.add_argument('--neumann_grad', type=eval, choices=[True, False], default=True)
    parser.add_argument('--grad_in_forward', type=eval, choices=[True, False], default=True)
    parser.add_argument('--nhidden', type=int, default=2)
    parser.add_argument('--idim', type=int, default=64)
    parser.add_argument('--densenet', type=eval, choices=[True, False], default=False)
    parser.add_argument('--densenet_depth', type=int, default=3)
    parser.add_argument('--densenet_growth', type=int, default=32)
    parser.add_argument('--learnable_concat', type=eval, choices=[True, False], default=True)
    parser.add_argument('--lip_coeff', help='Lipschitz coeff for DenseNet', type=float, default=0.98)
    # Optimizer and scheduler
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--sched_patience', default=5, type=int)
    parser.add_argument('--sched_factor', default=0.5, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    # Training
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--seed', default=2023, type=int)

    return parser


def dataset_specific_args():
    parser = ArgumentParser()

    parser.add_argument('--noise_min', default=0.005, type=float)  # 0.005
    parser.add_argument('--noise_max', default=0.020, type=float)  # 0.020
    parser.add_argument('--val_noise', default=0.015, type=float)
    parser.add_argument('--aug_rotate', default=True, choices=[True, False])
    parser.add_argument('--dataset_root', default='./data/ScoreDenoise', type=str)
    parser.add_argument('--dataset', default='PUNet', type=str)
    parser.add_argument('--resolutions', default=['10000_poisson', '30000_poisson', '50000_poisson'], type=list)
    # parser.add_argument('--resolutions', default=['10000_poisson'], type=list)
    parser.add_argument('--patch_size', type=int, default=1024)
    parser.add_argument('--num_patches', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', default=4, type=int)

    return parser


# -----------------------------------------------------------------------------------------
def train(phase='Train', finished_path=None, resume_checkpoint_path=None):
    comment = 'scoreset'
    cfg = model_specific_args().parse_args()
    pl.seed_everything(cfg.seed)
    # dataset/scoredenoise/transforms.py AddNoise

    dataset_cfg = dataset_specific_args().parse_args()
    datamodule = ScoreDenoiseDataModule(dataset_cfg)

    trainer_config = {
        'default_root_dir': './runs/30_32dim3feataug',
        'accelerator': 'gpu',  # Set this to None for CPU training
        'devices': -1,
        # 'gpus'                 : 2,
        # 'strategy' : "ddp_find_unused_parameters_false", # 一般情况下，如果是单机多卡，建议使用 ddp模式，因为dp模式需要非常多的data和model传输，非常耗时
        'fast_dev_run': False,  # fast_dev_run项可以配置train/val/test阶段的循环次数,跑完就停止代码,快速查看各流程代码正确性
        'max_epochs': 150,  # cfg.max_epoch, 没问题了
        'precision': 32,  # 32, 16, 'bf16'
        'gradient_clip_val': 1e-3,
        'deterministic': False,  # 训练前默认先执行再次validation_step，避免训练后验证阶段出错
        'num_sanity_val_steps': 0,  # -1,  # -1 or 0
        # 'logger'               : TensorBoardLogger("tb_logs", name="my_model")
        # 'enable_checkpointing' : False, # 是否保存version文件
        # 'callbacks'            : [TimeTrainingCallback(), LightningProgressBar()],
        # 'resume_from_checkpoint' : resume_checkpoint_path,
        # 'profiler'             : "pytorch",
    }

    module = TrainerModule(cfg)
    trainer = pl.Trainer(**trainer_config)
    trainer.is_interrupted = False

    if phase == 'Train':
        if comment is not None:
            print(f'\nComment: \033[1m{comment}\033[0m')
        # if resume_checkpoint_path is not None:
        #     state_dict = torch.load(resume_checkpoint_path)
        #     module.network.load_state_dict(state_dict)
        #     module.network.init_as_trained_state()

        trainer.fit(model=module, datamodule=datamodule, ckpt_path=resume_checkpoint_path)

        if finished_path is not None and trainer_config['fast_dev_run'] is False and trainer.is_interrupted is False:
            if trainer_config["max_epochs"] > 10:
                save_path = finished_path + f'-epoch{trainer_config["max_epochs"]}.ckpt'
                torch.save(module.network.state_dict(), save_path)
                print(f'Model has been save to \033[1m{save_path}\033[0m')
    else:  # Test
        state_dict = torch.load(resume_checkpoint_path)
        module.network.load_state_dict(state_dict)
        module.network.init_as_trained_state()
        trainer.test(model=module, datamodule=datamodule)


def update_lipschitz(model):
    # print('update')
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, base_layers.SpectralNormLinear):
                m.compute_weight(update=True)
            if isinstance(m, base_layers.InducedNormLinear):
                m.compute_weight(update=True)


# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    finished_path = 'product/ckpt/1'
    # checkpoint_path = './runs/lightning_logs/version_2/checkpoints'
    # resume_checkpoint_dir = './runs/30_32dim3feataug/lightning_logs/version_3/checkpoints/'
    # checkpoint_path = os.listdir(resume_checkpoint_dir)[0]
    # resume_checkpoint_path = resume_checkpoint_dir + checkpoint_path
    train('Train', finished_path, None)           # Train from begining, save network params after finish
    # train('Train', finished_path, resume_checkpoint_path)  # Train from previous checkpoint, save network params after finish
    # train('Train', None, None)  # Train from begining, and save nothing after finish
    # train('Test', checkpoint_path, None)            # Test with given checkpoint
