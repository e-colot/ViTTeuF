from utils import load_utils
from model import PointPillar
import torch
import matplotlib.pyplot as plt

PATH_TO_DATA = './data' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

feeder = load_utils.Feeder(PATH_TO_DATA, device)

H, W = 256, 256
block_scale = 0.125
max_voxels = 256

VFE_layers = [18, 64]

bev_in = VFE_layers[-1]
bev_out = 128
bev_hidden = [64, 128, 128, 256]
bev_stride = [2, 2, 1, 2]
bev_kernel = 3
bev_spatial_compression = 2

split = PointPillar.PillarSplit(H, W, block_scale, max_voxels).to(device)
vfe = PointPillar.PillarVFE(H, W, block_scale, VFE_layers).to(device)
bev = PointPillar.PillarConv(bev_in, bev_out, bev_hidden, bev_kernel, bev_stride, bev_spatial_compression).to(device)


for points, anns, tokens in feeder:    
    # points is a list of tensors of shape:
    # (N, 4)
    # tensor of Nx4 elements
    # N (number of lidar points) is varying
    # 4 is the points features (x, y, z, intensity)

    # anns (for annotations) is a 2D list of dictionaries
    # [N_el]
    # B is the batch size and N_el the number of elements in the scene
    # The dictionary of each element contains:
    #   'center': numpy array of 3 elements, (x, y, z)
    #   'size': numpy array of 3 elements, (x, y, z)
    #   'name': string
    #   'token': string

    # tokens, a string identifying the frame

    pillars, pillar_usage, pillar_idx = split(points)

    pillars_2D = vfe(pillars, pillar_usage, pillar_idx)

    pillar_out = bev(pillars_2D)

    print(pillar_out.shape)

    break


