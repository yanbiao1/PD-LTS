
import os
import sys
import argparse
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

SCALES  = ['0.01', '0.02', '0.025', '0.03']

sys.path.append(os.getcwd())

from dataset.scoredenoise.transforms import AddNoise
from modules.utils.score_utils import farthest_point_sampling


def evaluate(args, scale, path):
    _, file_name = os.path.split(path)
    output_path = Path(args.output_dir) / f'p{args.npoint}_n{scale}' / file_name

    noiser = AddNoise(noise_std_min=float(scale), noise_std_max=float(scale))

    raw = np.loadtxt(path, dtype=np.float32)
    raw = torch.from_numpy(raw)
    raw, fps_idx = farthest_point_sampling(raw.view(1, -1, 3), int(args.npoint))

    data = { 'pcl_clean': torch.squeeze(raw, dim=0) }
    data = noiser(data)
    np.savetxt(output_path, data['pcl_noisy'], fmt='%.6f')

    if args.normal_dir is not None:
        normal_in_path  = Path(args.normal_dir) / file_name.replace('.xyz', '.normal')
        normal_out_path = Path(args.output_dir) / f'p{args.npoint}_n{scale}' / file_name.replace('.xyz', '.normal')
        pc_normal = np.loadtxt(normal_in_path, dtype=np.float32)

        fps_idx = fps_idx[0].numpy()
        pc_normal = pc_normal[fps_idx]
        np.savetxt(normal_out_path, pc_normal, fmt='%.6f')


def mp_walkFile(func, args, directory):

    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if '.xyz' in f]

        # Single thread processing
        for scale in SCALES:
            print(f'Points: #{args.npoint}, Noise {scale}')

            outdir = Path(args.output_dir) / f'p{args.npoint}_n{scale}'
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            for path in tqdm(paths):
                func(args, scale, path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--normal_dir', type=str, default=None, help='Path to normal directory')
    parser.add_argument('--npoint', type=str, required=True)
    args = parser.parse_args()

    mp_walkFile(evaluate, args, args.input_dir)
