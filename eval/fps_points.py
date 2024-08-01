
import argparse
import numpy as np


def fps_numpy(pts, K):
    def calc_distances(p0, points):
        return ((p0 - points)**2).sum(axis=1)

    farthest_pts = np.zeros((K, 3), dtype=np.float32)
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts



if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_file', type=str, required=True, help='Path to input xyz file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output xyz file')
    parser.add_argument('--num_points', type=int, required=True, help='Target number of points')
    args = parser.parse_args()

    i_xyz = np.loadtxt(args.input_file, dtype=np.float32)
    o_xyz = fps_numpy(i_xyz, K=args.num_points)

    np.savetxt(args.output_file, o_xyz, fmt='%.6f')
