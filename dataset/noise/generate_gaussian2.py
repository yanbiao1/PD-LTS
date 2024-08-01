
import os
import sys
import argparse
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())

from dataset.scoredenoise.transforms import AddNoise


def evaluate(args, path):
    _, file_name = os.path.split(path)
    output_path = Path(args.output_dir) / file_name

    noiser = AddNoise(noise_std_min=float(args.scale1), noise_std_max=float(args.scale2))

    raw = np.loadtxt(path, dtype=np.float32)
    raw = torch.from_numpy(raw)

    data = { 'pcl_clean': torch.squeeze(raw, dim=0) }
    data = noiser(data)
    np.savetxt(output_path, data['pcl_noisy'].numpy(), fmt='%.6f')

def mp_walkFile(func, args, directory):

    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if '.xyz' in f]

        # Single thread processing
        outdir = Path(args.output_dir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        for path in tqdm(paths):
            func(args, path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--scale1', type=str, required=True)
    parser.add_argument('--scale2', type=str, required=True)
    args = parser.parse_args()

    mp_walkFile(evaluate, args, args.input_dir)
