
import math
import numpy as np
import re
import os

from sklearn.neighbors import NearestNeighbors

UNIFORM_PRECENTAGE_NAMES = ['0.4%', '0.6%', '0.8%', '1.0%', '1.2%']
precentages = np.array([0.004, 0.006, 0.008, 0.010, 0.012])


def cal_nearest_distance(queries, pc, k=2):
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    dis, knn_idx = knn_search.kneighbors(queries, return_distance=True)
    return dis[:,1]

def analyze_uniform(idx_file, radius_file, map_points_file):

    points = np.loadtxt(map_points_file).astype(np.float32)[:, 4:]
    radius = np.loadtxt(radius_file)
    # print('radius:',radius)
    with open(idx_file) as f:
        lines = f.readlines()

    sample_number = 1000
    rad_number = radius.shape[0]

    uniform_measure = np.zeros([rad_number,1])

    densitys = np.zeros([rad_number,sample_number])

    expect_number = precentages * points.shape[0]
    expect_number = np.reshape(expect_number, [rad_number, 1])

    for j in range(rad_number):
        uniform_dis = []

        for i in range(sample_number):

            density, idx = lines[i*rad_number+j].split(':')
            densitys[j,i] = int(density)
            coverage = np.square(densitys[j,i] - expect_number[j]) / expect_number[j]

            num_points = re.findall("(\d+)", idx)

            idx = list(map(int, num_points))
            if len(idx) < 5:
                continue

            idx = np.array(idx).astype(np.int32)
            map_point = points[idx]

            shortest_dis = cal_nearest_distance(map_point,map_point,2)
            disk_area = math.pi * (radius[j] ** 2) / map_point.shape[0]
            expect_d = math.sqrt(2 * disk_area / 1.732)##using hexagon

            dis = np.square(shortest_dis - expect_d) / expect_d
            # dis_mean = np.mean(dis)
            dis_mean = np.nanmean(dis)
            uniform_dis.append(coverage*dis_mean)

        uniform_dis = np.array(uniform_dis).astype(np.float32)
        if len(uniform_dis) == 0:
            uniform_measure[j, 0] = np.array(0.0, dtype=np.float)
        else:
            uniform_measure[j, 0] = np.mean(uniform_dis)

    return uniform_measure

def point_uniformity(xyz_path, off_path, cache_path=1):
    
    if cache_path == 1:
        cmd = '''./eval/uniformity/build/uniformity %s %s > /dev/null''' % (off_path, xyz_path)
        os.system(cmd)
        return analyze_uniform('eval/uniformity/pc_disk_idx2.txt', 'eval/uniformity/pc_radius2.txt', 'eval/uniformity/pc_point2mesh_distance2.txt')
    else:
        cmd = '''./eval/uniformity/build/uniformity2 %s %s > /dev/null''' % (off_path, xyz_path)
        os.system(cmd)
        return analyze_uniform('eval/uniformity/pc_disk_idx2.txt', 'eval/uniformity/pc_radius2.txt', 'eval/uniformity/pc_point2mesh_distance2.txt')
