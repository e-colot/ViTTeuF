from utils import load_utils
from model import PointPillar
import torch

PATH_TO_DATA = './data' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feeder = load_utils.Feeder(PATH_TO_DATA, device)


model = PointPillar.PillarSplit(220, 220, 1/16, 64)




model.to(device)

max_points = 0
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
    pillars, pillar_usage, pillar_idx = model(points)
    maxPointsPerPillar = max(pillar_usage)
    max_points = max(max_points, maxPointsPerPillar)

print(f"Max points: {max_points}")

