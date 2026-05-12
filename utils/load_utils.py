import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os

class Feeder:
    def __init__(self, path_to_data, device):
        self.train_loader = get_dataloader(dataroot=path_to_data)
        self.device = device

    def __iter__(self):
        for points, anns, tokens in self.train_loader:
            points = points.to(self.device).squeeze(0)
            anns = anns.to(self.device)
            tokens = tokens.to(self.device)

            yield points, anns, tokens

max_points = 0

class NuScenesLidarDataset(Dataset):
    def __init__(self, nusc, sensor_name='LIDAR_TOP'):
        self.nusc = nusc
        self.sensor_name = sensor_name
        self.tokens = [s['token'] for s in self.nusc.sample]

        print(f"{'=' * 15} DATA LOADER CREATION {'=' * 15}")
        print(f"Dataset loaded with {len(self)} samples")
        print(f"{'-' * 50}\n")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        sample_token = self.tokens[idx]
        sample = self.nusc.get('sample', sample_token)
        
        # 1. Load LiDAR Points
        sd_token = sample['data'][self.sensor_name]
        data_path, _, _ = self.nusc.get_sample_data(sd_token)
        pc = LidarPointCloud.from_file(data_path)
        points = torch.tensor(pc.points.T, dtype=torch.float32) # [N, 4] (x, y, z, intensity)

        # 2. Get Annotations (Bounding Boxes)
        # We get boxes in the global frame or ego frame
        _, boxes, _ = self.nusc.get_sample_data(sd_token)
        
        annotations = []
        for box in boxes:
            # Storing translation (x, y, z), size (w, l, h), and class
            annotations.append({
                'center': box.center,
                'size': box.wlh,
                'name': box.name,
                'token': box.token
            })

        return points, annotations, sample_token

def collate_fn(batch):
    """Custom collate because LiDAR points have different shapes per frame."""
    points = [item[0] for item in batch]
    anns = [item[1] for item in batch]
    tokens = [item[2] for item in batch]
    return points, anns, tokens

def get_dataloader(dataroot, version='v1.0-mini', batch_size=1):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    dataset = NuScenesLidarDataset(nusc)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


