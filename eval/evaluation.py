
import os
import sys
import argparse
import numpy as np
import csv
import point_cloud_utils as pcu

sys.path.append(os.getcwd())

from glob import glob
from pathlib import Path

from sklearn.neighbors import NearestNeighbors


def point2surface_distance(args, file_name, pred, gt):
    """
    pred: [N1, 3]
    gt  : [N2, 3]
    """
    path_normal = Path(args.gt_dir) / file_name.replace('.xyz', '.normal')
    n = np.loadtxt(path_normal, dtype=np.float32)  # [N2, 3]

    nbrs = NearestNeighbors(n_neighbors=16, algorithm='ball_tree').fit(gt)
    _, indices = nbrs.kneighbors(pred)  # [N1, k]
    indices = indices.flatten()  # [N1 * k,]

    n = n[indices].reshape(-1, 16, 3)  # [N1, k, 3]

    p1 = pred.reshape(-1, 1, 3)  # [N1, 1, 3]
    p2 = gt[indices].reshape(-1, 16, 3)   # [N1, k, 3]

    numerator1 = np.sum(p1 * n, axis=-1)   # [N1, k]
    numerator2 = np.sum(p2 * n, axis=-1)   # [N1, k]

    numerator   = np.abs(numerator1 - numerator2)   # [N1, k]
    denominator = np.sqrt(np.sum(n ** 2, axis=-1))  # [N1, k]

    p2s = numerator / denominator  # [N1, k]
    p2s = np.amin(p2s, axis=-1)  # [N1,]
    return np.average(p2s)


def evaluation(args):

    metric_global = { 'name': 'Total', 'CD': 0.0, 'P2S': 0.0, 'HD': 0.0 }
    if args.csv is not None:
        fcsv = open(args.csv, 'w', encoding='utf-8')
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(['name', 'CD', 'P2S', 'HD'])

    # chamerLoss = ChamferDistance(dim=3)
    
    pred_paths = glob(f'{args.pred_dir}/*.xyz')
    for path in pred_paths:
        # if args.verbose:
        #     print(f'Evaluate {path}...')

        _, file_name = os.path.split(path)
        gt_path = os.path.join(args.gt_dir, file_name)

        pc_pred = np.loadtxt(path, dtype=np.float32)   # [N, 3]
        pc_gt = np.loadtxt(gt_path, dtype=np.float32)  # [N, 3]

        loss_cd = pcu.chamfer_distance(pc_pred, pc_gt)
        loss_hd = pcu.hausdorff_distance(pc_pred, pc_gt)
        loss_p2f = point2surface_distance(args, file_name, pc_pred, pc_gt)

        metric_local = {}
        metric_local['name'] = file_name
        metric_local['CD']  = loss_cd
        metric_local['P2S'] = loss_p2f
        metric_local['HD']  = loss_hd
        if args.csv is not None:
            csv_writer.writerow(list(metric_local.values()))

        metric_global['CD']  += loss_cd
        metric_global['P2S'] += loss_p2f
        metric_global['HD']  += loss_hd

        if args.verbose:
            print(metric_local)

    metric_global['CD']  = metric_global['CD']  / len(pred_paths)
    metric_global['P2S'] = metric_global['P2S'] / len(pred_paths)
    metric_global['HD']  = metric_global['HD']  / len(pred_paths)

    # metric_global['CD']  = metric_global['CD']
    # metric_global['P2S'] = metric_global['P2S']
    # metric_global['HD']  = metric_global['HD']
 
    print(f'Evaluation: [CD]{metric_global["CD"]}, [P2S]{metric_global["P2S"]}, [HD]{metric_global["HD"]}')

    if args.csv is not None:
        csv_writer.writerow(list(metric_global.values()))
        fcsv.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to predict directory')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to ground-truth directory')
    parser.add_argument('--csv', type=str, default=None, help='Path to save csv')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    evaluation(args)
