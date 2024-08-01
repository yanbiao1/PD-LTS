import os
import sys
import argparse
import numpy as np
import csv
import torch
import point_cloud_utils as pcu

from glob import glob
from tqdm import tqdm

import matplotlib.cm as cm
from matplotlib import colors

sys.path.append(os.getcwd())

from modules.utils.score_utils import chamfer_distance_unit_sphere
from modules.utils.score_utils import hausdorff_distance_unit_sphere
from modules.utils.score_utils import point_mesh_bidir_distance_single_unit_sphere
from modules.utils.score_utils import pointwise_p2m_distance_normalized

from eval.uniformity.uniform import point_uniformity, UNIFORM_PRECENTAGE_NAMES


@torch.no_grad()
def evaluation(args):
    metric_global = {'name': 'Total', 'CD': 0.0, 'P2M': 0.0, 'HD': 0.0}
    for uniform in UNIFORM_PRECENTAGE_NAMES:
        metric_global[uniform] = 0.0
    if args.csv is not None:
        Resdir, _ = os.path.split(args.csv)

        fcsv = open(args.csv, 'w', encoding='utf-8')
        csv_writer = csv.writer(fcsv)
        headers = ['name', 'CD', 'P2M', 'HD']

        if not args.not_eval_uniform:
            for uniform in UNIFORM_PRECENTAGE_NAMES:
                headers.append(uniform)
        csv_writer.writerow(headers)

    pred_paths = glob(f'{args.pred_dir}/*.xyz')
    iters = enumerate(pred_paths) if args.verbose else pred_paths
    for pred_path in iters:
        if args.verbose:
            print(f'Evaluate {pred_path}...')

        _, file_name = os.path.split(pred_path)
        mesh_path = os.path.join(args.off_dir, file_name).replace('.xyz', '.off')
        gt_path = os.path.join(args.gt_dir, file_name)

        pc_pred = torch.from_numpy(np.loadtxt(pred_path, dtype=np.float32)).to(args.device)  # [N, 3]
        pc_gt = torch.from_numpy(np.loadtxt(gt_path, dtype=np.float32)).to(args.device)  # [N, 3]

        bpc_pred = pc_pred.view(1, pc_pred.shape[0], pc_pred.shape[1])
        bpc_gt = pc_gt.view(1, pc_gt.shape[0], pc_gt.shape[1])

        verts, faces = pcu.load_mesh_vf(mesh_path)
        verts = torch.from_numpy(verts.astype(np.float32)).to(args.device)
        faces = torch.from_numpy(faces.astype(np.int64)).to(args.device)
        # =======================================================================
        # loss_cd = pytorch3d.loss.chamfer_distance(bpc_pred, bpc_gt)[0].item()
        p2f_path = os.path.join(Resdir, 'xyzRGB', file_name)
        dist_p2f = pointwise_p2m_distance_normalized(pcl=pc_pred, verts=verts, faces=faces)

        vmax = 16e-4
        dist_p2f = dist_p2f.cpu().numpy()
        dist_p2f = dist_p2f * 1.0
        rgb = colorize(dist_p2f, vmin=0.0, vmax=vmax, norm=args.norm, identical_color=False)
        xyzrgb = np.concatenate([pc_pred.cpu().numpy(), rgb], axis=1)
        if not os.path.exists(os.path.join(Resdir, 'xyzRGB')):
            os.mkdir(os.path.join(Resdir, 'xyzRGB'))
        np.savetxt(p2f_path, xyzrgb, fmt='%.6f')

        # ========================================================================
        loss_cd = chamfer_distance_unit_sphere(bpc_pred, bpc_gt)[0].item()
        loss_hd = hausdorff_distance_unit_sphere(bpc_pred, bpc_gt)[0].item()
        loss_p2m = point_mesh_bidir_distance_single_unit_sphere(pcl=pc_pred, verts=verts, faces=faces).item()

        if not args.not_eval_uniform:
            loss_uniforms = point_uniformity(pred_path, mesh_path, args.cache_idx)

        metric_local = {}
        metric_local['name'] = file_name
        metric_local['CD'] = loss_cd
        metric_local['P2M'] = loss_p2m
        metric_local['HD'] = loss_hd

        if not args.not_eval_uniform:
            for (i, uniform) in enumerate(UNIFORM_PRECENTAGE_NAMES):
                metric_local[uniform] = loss_uniforms[i]

        if args.csv is not None:
            csv_writer.writerow(list(metric_local.values()))

        metric_global['CD'] += loss_cd
        metric_global['P2M'] += loss_p2m
        metric_global['HD'] += loss_hd

        if not args.not_eval_uniform:
            for (i, uniform) in enumerate(UNIFORM_PRECENTAGE_NAMES):
                metric_global[uniform] += np.nan_to_num(loss_uniforms[i])

        # print(f'{file_name}: {loss_cd}')

        if args.verbose:
            print(metric_local)

    metric_global['CD'] = metric_global['CD'] / len(pred_paths)
    metric_global['P2M'] = metric_global['P2M'] / len(pred_paths)
    metric_global['HD'] = metric_global['HD'] / len(pred_paths)

    print(f'Evaluation: [CD]{metric_global["CD"]}, [P2M]{metric_global["P2M"]}, [HD]{metric_global["HD"]}')

    if not args.not_eval_uniform:
        uniforms_str = '\t'
        for uniform in UNIFORM_PRECENTAGE_NAMES:
            uniforms_str += f'  [{uniform}]{metric_global[uniform]}'
        print(uniforms_str)

    if args.csv is not None:
        csv_writer.writerow(list(metric_global.values()))
        fcsv.close()


def colorize(p2s, vmin=0, vmax=1, norm='Normalize', identical_color=False):
    # cmap_name = 'coolwarm'
    # cmap = cm.get_cmap(cmap_name)  # PiYG

    vals = np.ones((256, 4))
    min_color = np.array([225, 225, 254]) / 256  # 118, 170, 232
    mid_color = np.array([22, 22, 135]) / 256
    max_color = np.array([247, 239, 2]) / 256

    vals[:, 0] = np.concatenate(
        [np.linspace(min_color[0], mid_color[0], 64), np.linspace(mid_color[0], max_color[0], 192)])
    vals[:, 1] = np.concatenate(
        [np.linspace(min_color[1], mid_color[1], 64), np.linspace(mid_color[1], max_color[1], 192)])
    vals[:, 2] = np.concatenate(
        [np.linspace(min_color[2], mid_color[2], 64), np.linspace(mid_color[2], max_color[2], 192)])

    cmap = colors.ListedColormap(vals)

    if identical_color:
        norm = cm.colors.Normalize(vmin=0.0, vmax=0.0)
    elif norm == 'Normalize':
        norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm == 'Power':
        norm = cm.colors.PowerNorm(2, vmin, vmax)
    elif norm == 'TwoSlope':
        norm = cm.colors.TwoSlopeNorm(vmin + (vmax - vmin) * 0.2, vmin, vmax)
    elif norm == 'Log':
        norm = cm.colors.LogNorm(vmin + 1e-5, vmax + 1e-5)
    else:
        assert False
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(p2s)[:, :3]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to predict directory')
    parser.add_argument('--off_dir', type=str, required=True, help='Path to ground-truth mesh directory')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to ground-truth points directory')
    parser.add_argument('--csv', type=str, default=None, help='Path to save csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cache_idx', type=str, default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--not_eval_uniform', default=True)
    parser.add_argument('--norm', type=str, default='TwoSlope')
    args = parser.parse_args()

    evaluation(args)
