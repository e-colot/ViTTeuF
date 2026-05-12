import torch
import torch.nn as nn

class PillarSplit(nn.Module):
    def __init__(self, H, W, block_scale, max_voxels):
        # (x, y) in (-110, -110) (110, 110)
        # -> H = W = 220
        # block_scale = 0.25 ?
        super().__init__()

        self.H = H
        self.W = W
        self.block_scale = block_scale
        self.max_voxels = max_voxels

        self.gridWidth = int(self.H/self.block_scale)
        self.gridHeight = int(self.W/self.block_scale)

    def forward(self, point_cloud):
        r"""
        Splits the pointcloud into pillars.
        
        Input:
            point_cloud: (L x 4)
        Outputs:
            pillars: (N x max_voxels x 4)
                contains the points, grouped in pillars
            pillar_usage: (N)
                contains the number of voxels per pillar
            pillar_idx: (N x 2)
                contains the grid index of each of the pillars kept
        """
        device = point_cloud.device
        dtype = point_cloud.dtype

        L = point_cloud.shape[0]

        # shift to avoid negative coordinates
        shift = torch.tensor([self.H/2, self.W/2, 0, 0], device=device, dtype=dtype)
        point_cloud_shifted = point_cloud + shift

        x_idx = torch.div(point_cloud_shifted[:, 0], self.block_scale, rounding_mode='floor').long()
        y_idx = torch.div(point_cloud_shifted[:, 1], self.block_scale, rounding_mode='floor').long()
        x_idx = torch.clamp(x_idx, 0, self.gridWidth - 1)
        y_idx = torch.clamp(y_idx, 0, self.gridHeight - 1)

        flatPointIdx = x_idx * self.gridHeight + y_idx

        pillarCnt = self.gridWidth * self.gridHeight
        pillars = torch.zeros((pillarCnt, self.max_voxels, 4), device=device, dtype=dtype)
        pillar_usage = torch.zeros(pillarCnt, device=device, dtype=torch.int32)

        flatPointIdx = flatPointIdx.long().clamp(0, pillarCnt - 1)
        for i in range(L):
            idx = flatPointIdx[i]
            if pillar_usage[idx] < self.max_voxels:
                pillars[idx, pillar_usage[idx], :] = point_cloud[i, :]
                pillar_usage[idx] += 1

        nonEmptyMask = (pillar_usage > 0)

        pillars = pillars[nonEmptyMask, :, :]
        pillar_usage = pillar_usage[nonEmptyMask]
        pillar_idx1D = torch.arange(pillarCnt, device=device)[nonEmptyMask]

        N = pillar_idx1D.shape[0]

        pillar_idx = torch.zeros((N, 2), device=device, dtype=torch.int32)
        pillar_idx[:, 0] = torch.div(pillar_idx1D, self.gridHeight, rounding_mode='floor')
        pillar_idx[:, 1] = pillar_idx1D % self.gridHeight

        return pillars, pillar_usage, pillar_idx
