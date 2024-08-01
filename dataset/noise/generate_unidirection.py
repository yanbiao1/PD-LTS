
import os
import sys
import argparse
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

SCALES  = ['0.01', '0.02', '0.025']

sys.path.append(os.getcwd())

from dataset.scoredenoise.transforms import AddUniDirectional


def evaluate(args, scale, path):
    _, file_name = os.path.split(path)
    output_dir = Path(args.output_dir) / f'p{args.npoint}_n{scale}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = Path(args.output_dir) / f'p{args.npoint}_n{scale}' / file_name

    noiser = AddUniDirectional(std_factor=float(scale))

    raw = np.loadtxt(path, dtype=np.float32)
    data = { 'pcl_clean': torch.from_numpy(raw) }
    data = noiser(data)
    np.savetxt(output_path, data['pcl_noisy'], fmt='%.6f')


def mp_walkFile(func, args, directory):

    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if '.xyz' in f]

        # Single thread processing
        for scale in SCALES:
            print(f'Points: #{args.npoint}, Noise {scale}')
            for path in tqdm(paths):
                func(args, scale, path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--npoint', type=str, required=True)
    args = parser.parse_args()

    mp_walkFile(evaluate, args, args.input_dir)
