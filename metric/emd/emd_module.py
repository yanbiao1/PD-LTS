# EMD approximation module (based on auction algorithm)
# memory complexity: O(n)
# time complexity: O(n^2 * iter) 
# author: Minghua Liu

# Input:
# xyz1, xyz2: [#batch, #points, 3]
# where xyz1 is the predicted point cloud and xyz2 is the ground truth point cloud 
# two point clouds should have same size and be normalized to [0, 1]
# #points should be a multiple of 1024
# #batch should be no greater than 512
# eps is a parameter which balances the error rate and the speed of convergence
# iters is the number of iteration
# we only calculate gradient for xyz1

# Output:
# dist: [#batch, #points],  sqrt(dist) -> L2 distance 
# assignment: [#batch, #points], index of the matched point in the ground truth point cloud
# the result is an approximation and the assignment is not guranteed to be a bijection

import time
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import emd
from pytorch3d.ops import knn_points


class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert (n == m)
        assert (xyz1.size()[0] == xyz2.size()[0])
        assert (n % 1024 == 0)
        assert (batchsize <= 512)

        xyz1 = xyz1.contiguous().float().cuda()  # contiguous()函数的作用：把tensor变成在内存中连续分布的形式
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd.forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx,
                    unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()

        emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None


class emdModule(nn.Module):
    def __init__(self):
        super(emdModule, self).__init__()

    def forward(self, input1, input2, eps, iters):
        return emdFunction.apply(input1, input2, eps, iters)


def bijection(pcl_A, pcl_B, knn_num):
    """
    Args:
        pcl_A:  The first point cloud, (N, 3).
        pcl_B:  The Second point cloud, (N, 3).
        num_patches:  Number of patches P.
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    N = pcl_A.size(0)
    dists = (pcl_A - pcl_B) * (pcl_A - pcl_B)
    dists = dists.sum(-1)
    sum1 = np.sqrt(dists.cpu().mean())
    knn_dists, pat_B_idx, _ = knn_points(pcl_A.unsqueeze(0), pcl_B.unsqueeze(0), K=knn_num,
                                         return_nn=True)  # (1, N, knn_num) (1, N, knn_num, 3)
    knn_dists, pat_B_idx = knn_dists[0], pat_B_idx[0]  # (N, knn_num) (N, knn_num) (N, knn_num, 3)
    for i in range(N):
        for j in range(knn_num):
            if pat_B_idx[i][j] == i:
                break
            d1 = (pcl_A[pat_B_idx[i][j]] - pcl_B[i]) ** 2
            d1 = d1.sum()
            idx, idx_matrix, i_matrix = pat_B_idx[i][j], torch.full((N, knn_num), pat_B_idx[i][j]).to(
                pat_B_idx), torch.full((N, knn_num), i).to(pat_B_idx)
            if dists[i] > knn_dists[i][j] and dists[idx] > d1:
                dists[i], dists[idx] = knn_dists[i][j], d1
                pcl_B[i], pcl_B[idx] = pcl_B[idx], pcl_B[i]
                pat_B_idx = torch.where(pat_B_idx == i,
                                        N + 1,
                                        pat_B_idx)

                pat_B_idx = torch.where(pat_B_idx == idx,
                                        i_matrix,
                                        pat_B_idx)

                pat_B_idx = torch.where(pat_B_idx == N + 1,
                                        idx_matrix,
                                        pat_B_idx)
    sum2 = np.sqrt(dists.cpu().mean())
    sum3 = np.sqrt(((pcl_A - pcl_B) * (pcl_A - pcl_B)).cpu().sum(-1).mean())
    print('sum1:', sum1)
    print('sum2:', sum2)
    print('sum3:', sum3)
    return pcl_B  # (P, M, 3)


def test_emd1():
    set1 = {10, 20, 30, 40, 50}  
    set2 = {30, 40, 50, 60, 70}  

    # 创建一个新的集合，包含两个集合中的唯一元素  
    new_set = set1.union(set2)  

    # 打印新的集合  
    print(new_set)  

    # 向新的集合中添加元素80  
    new_set.add(80)  

    # 打印添加元素80后的集合  
    print(new_set)  

    # 从新的集合中移除第一个元素  
    new_set.remove(min(new_set))  

    # 打印移除第一个元素后的集合  
    print(new_set)  

    # 清空新的集合  
    new_set.clear()  

    # 打印清空后的集合  
    print(new_set)


def test_emd():
    set1 = {10, 20, 30, 40, 50}  
    set2 = {30, 40, 50, 60, 70}  

    # 创建一个新的集合，包含两个集合中的唯一元素  
    new_set = set1.union(set2)  

    # 打印新的集合  
    print(new_set)  

    # 向新的集合中添加元素80  
    new_set.add(80)  

    # 打印添加元素80后的集合  
    print(new_set)  

    # 从新的集合中移除第一个元素  
    new_set.remove(new_set[0])  

    # 打印移除第一个元素后的集合  
    print(new_set)  

    # 清空新的集合  
    new_set.clear()  

    # 打印清空后的集合  
    print(new_set)


if __name__ == '__main__':
    test_emd()
