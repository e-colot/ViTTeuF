import torch
import torch.nn as nn

class PillarSplit(nn.Module):
    r"""
    Splits the pointcloud into pillars.
    Purely algorithmic, no ML layer.
    """
    def __init__(self, H, W, block_scale, max_voxels):
        super().__init__()

        self.H = H
        self.W = W
        self.block_scale = block_scale
        self.max_voxels = max_voxels

        self.gridHeight = int(H/block_scale)
        self.gridWidth = int(W/block_scale)

    def forward(self, point_cloud):
        r"""        
        Input:

            point_cloud:    L x 4

        Outputs:

            pillars:        N x max_voxels x 4
            pillar_usage:   N
            pillar_idx:     N x 2
        """
        device = point_cloud.device
        dtype = point_cloud.dtype
        L = point_cloud.shape[0]

        # Shift to be at positive coordinates
        shift = torch.tensor([self.W/2, self.H/2, 0, 0], device=device, dtype=dtype)
        pointcloud_shifted = point_cloud + shift

        x_idx = torch.div(pointcloud_shifted[:, 0], self.block_scale, rounding_mode='floor').long()
        y_idx = torch.div(pointcloud_shifted[:, 1], self.block_scale, rounding_mode='floor').long()
        x_idx = x_idx.clamp(0, self.gridWidth - 1)
        y_idx = y_idx.clamp(0, self.gridHeight - 1)

        flat_point_idx = x_idx + y_idx * self.gridWidth
        sorted_indices = torch.argsort(flat_point_idx, stable=True)
        sorted_flat_idx = flat_point_idx[sorted_indices]

        diffs = torch.cat([torch.tensor([1], device=device), (sorted_flat_idx[1:] != sorted_flat_idx[:-1]).long()])
        first_occurrence_mask = (diffs == 1)
        # True if first point of the pillar
        
        group_starts = torch.where(first_occurrence_mask)[0]
        _, counts = torch.unique_consecutive(sorted_flat_idx, return_counts=True)
        start_indices = torch.repeat_interleave(group_starts, counts)
        
        local_idx_sorted = torch.arange(L, device=device) - start_indices
        # contains the idx of the voxel inside its pillar
        
        mask_sorted = local_idx_sorted < self.max_voxels
        # limits to self.max_voxels voxels in a pillar
        
        final_pointcloud_indices = sorted_indices[mask_sorted]
        final_local_indices = local_idx_sorted[mask_sorted]
        final_pillar_ids = sorted_flat_idx[mask_sorted]

        unique_pillars, inverse_mapping, pillar_usage = torch.unique(final_pillar_ids, 
                                                return_inverse=True, return_counts=True)
        # only creates pillar if it contains points

        N = unique_pillars.shape[0]
        pillars = torch.zeros((N, self.max_voxels, 4), device=device, dtype=dtype)
        
        pillars[inverse_mapping, final_local_indices] = point_cloud[final_pointcloud_indices]
        # make use of the different indexes

        pillar_usage = pillar_usage.to(torch.int32)

        pillar_idx = torch.zeros((N, 2), device=device, dtype=torch.int32)
        pillar_idx[:, 0] = unique_pillars % self.gridWidth
        pillar_idx[:, 1] = torch.div(unique_pillars, self.gridWidth, rounding_mode='floor')

        return pillars, pillar_usage, pillar_idx
    
class PillarVFE(nn.Module):
    r"""
    Pillar Voxel Feature Extraction.
    Each pillar gets a fixed number of features using relative coordinates and linear transformations
    """
    def __init__(self, H, W, block_scale, layers):
        super().__init__()

        self.H = H
        self.W = W
        self.block_scale = block_scale

        self.gridHeight = int(H/block_scale)
        self.gridWidth = int(W/block_scale)

        self.layer_list = nn.ModuleList()
        input_dim = 9
        for i in range(len(layers)-1):
            thisStep = nn.Sequential(
                nn.Linear(input_dim, layers[i], bias=False),
                nn.LayerNorm(layers[i])
            )
            input_dim += layers[i]
            self.layer_list.append(thisStep)

        self.last_layer = nn.Linear(input_dim, layers[-1], bias = True)

    def forward(self, pillars, pillar_usage, pillar_idx):
        r"""        
        Inputs:

            pillars:        N x max_voxels x 4
            pillar_usage:   N
            pillar_idx:     N x 2

        Output:

            pillars_2D:     W x H x C
        """
        # feature augmentation
        pillar_x_centers = pillar_idx[:, 0] * self.block_scale - self.H/2.0
        pillar_y_centers = pillar_idx[:, 1] * self.block_scale - self.W/2.0

        rel_center_x = (pillars[:, :, 0] - pillar_x_centers.unsqueeze(1)).unsqueeze(-1)
        rel_center_y = (pillars[:, :, 1] - pillar_y_centers.unsqueeze(1)).unsqueeze(-1)

        pillar_centroid = pillars[:, :, 0:3].sum(dim=1, keepdim=True) / pillar_usage.view(-1, 1, 1).to(torch.float32)
        rel_centroid = pillars[:, :, 0:3] - pillar_centroid

        pillar_features = torch.cat((pillars, rel_center_x, rel_center_y, rel_centroid), dim=-1)

        for featureLayer in self.layer_list:
            x = featureLayer(pillar_features)
            x = torch.relu(x)
            pillar_features = torch.cat((pillar_features, x), dim=-1)
        
        pillar_features = self.last_layer(pillar_features)

        # keep 1 feature per pillar -> N x C
        pillar_features = torch.max(pillar_features, dim=1)[0]

        # Scatter to a 2D map
        pillars_2D = torch.zeros(self.gridWidth, self.gridHeight, pillar_features.shape[1], dtype=pillar_features.dtype, device = pillar_features.device)
        pillars_2D[pillar_idx[:,0], pillar_idx[:,1],:] = pillar_features

        return pillars_2D