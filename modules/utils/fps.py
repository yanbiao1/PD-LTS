
import torch
import numpy as np


# ---------------------------------------------------------------------
# Code from PointNet++
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# ---------------------------------------------------------------------
def normalize_point_cloud_numpy(v: np.array):
    center = np.mean(v, axis=0, keepdims=True)
    v = v - center
    scale = (1 / np.abs(v).max()) * 0.999999
    v = v * scale
    return v, center, scale



# ---------------------------------------------------------------------
def farthest_point_sampling(points, n_points, RAN=True):
    """
    TODO: Documentation
    points: Point cloud data, (batch_size, num_points, channels)
    n_point: Number of samples from this point cloud
    """

    device = points.device
    batch_size, num_points, channels = points.shape

    centroids = torch.zeros((batch_size, n_points), dtype=torch.long, device=device)
    distance  = torch.ones((batch_size, num_points), device=device) * 1e10

    if RAN:
        farthest = torch.randint(low=0, high=1, size=(batch_size,), dtype=torch.long, device=device)
        #  farthest = torch.zeros((batch_size,), dtype=torch.long, device=device)
    else:
        farthest = torch.randint(low=1, high=2, size=(batch_size,), dtype=torch.long, device=device)
        #  farthest = torch.ones((batch_size,), dtype=torch.long, device=device)

    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    for i in range(n_points):
        centroids[:, i] = farthest
        # select points in current loop
        centroid = points[batch_indices, farthest, :].view(batch_size, 1, channels)
        # calc the distance between all points and selected points
        dist = torch.sum((points - centroid)**2, dim=-1, keepdim=False)
        # Update only the mininum distance to current selected points
        mask = dist < distance
        distance[mask] = dist[mask]
        # Select the maximum distance point
        _, farthest = torch.max(distance, dim=-1, keepdim=False)

    return centroids

# ---------------------------------------------------------------------
def index_points(points, idx):
    """
    Derive the subset of points by idx
    points: (batch_size, num_points, channels)
    idx   : (batch_size, n_points)
    """
    device = points.device
    batch_size, _, _ = points.shape
    view_shape = list(idx.shape)    # (batch_size, n_points)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)  # [batch_size, n_points]
    repeat_shape[0] = 1             # [1, n_points]

    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    #  print(f"view shape  : {view_shape}")
    #  print(f"repeat shape: {repeat_shape}")
    #  print(f"batch_indices shape: {batch_indices.shape}")
    #  print(f"{batch_indices[:5, :10]}")
    #  print(f"new_points shape: {new_points.shape}")

    return new_points
# ---------------------------------------------------------------------
