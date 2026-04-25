import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from nuscenes.nuscenes import NuScenes
from PIL import Image
import os

class NuScenesDataset(Dataset):
    def __init__(self, nusc, sensor_name='CAM_FRONT', transform=None):
        """
        Args:
            nusc: The initialized NuScenes object.
            sensor_name: Which sensor to load (e.g., 'CAM_FRONT', 'LIDAR_TOP').
            transform: PyTorch transforms for the data.
        """
        self.nusc = nusc
        self.sensor_name = sensor_name
        self.transform = transform
        # Each 'sample' in nuScenes is a timestamp with multi-modal data
        self.tokens = [s['token'] for s in self.nusc.sample]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        sample_token = self.tokens[idx]
        sample_rec = self.nusc.get('sample', sample_token)
        
        # Get the data token for the specific sensor
        sd_token = sample_rec['data'][self.sensor_name]
        
        # Returns the actual file path on your disk
        data_path, _, _ = self.nusc.get_sample_data(sd_token)
        
        # Load image
        image = Image.open(data_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # For now, we return the image and the token 
        # (You can expand this to return 3D boxes or labels later)
        return image, sample_token

def get_dataloader(dataroot, version='v1.0-mini', batch_size=4, sensor='CAM_FRONT'):
    # 1. Initialize the NuScenes DB
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    
    # 2. Define standard transforms
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Create Dataset and Loader
    dataset = NuScenesDataset(nusc, sensor_name=sensor, transform=data_transform)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0 # Set to 0 if you encounter Windows multiprocessing issues
    )
    
    return loader

