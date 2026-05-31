import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
import torchvision.transforms as transforms
import os


class Feeder:
    def __init__(self, path_to_data, model, loss_fn, optimizer, device, batch_size=1):
        self.loader = get_dataloader(dataroot=path_to_data, batch_size=batch_size)
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, point_clouds, histories, futures) in enumerate(self.loader):
            # Send standard batch tensors to device
            images = images.to(self.device)         # [Batch, 3, 224, 224]
            histories = histories.to(self.device)   # [Batch, 4, 2]
            futures = futures.to(self.device)       # [Batch, 12, 2]
            
            # Since batch_size=1, grab the first raw point cloud item from the collated list
            pc_single = point_clouds[0].to(self.device) # [L, 4] | list[tensor[L,4]]-> list[0] for batch_size=1

            # Optimization Step
            self.optimizer.zero_grad()
            
            # Forward pass through the unified MultiModalTrajectoryPredictor
            pred_trajs, pred_logits = self.model(images, pc_single, histories)
            
            # Loss Calculation
            loss, loss_reg, loss_cls = self.loss_fn(pred_trajs, pred_logits, futures)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx:03d} | Loss: {loss.item():.4f} (Reg: {loss_reg.item():.3f}, Cls: {loss_cls.item():.3f})")
                if loss_reg.item() > 100 :
                    model_guess = torch.max(pred_logits, dim=1)[1].item()
                    print(f" Mode prediction is {model_guess}")
                
        return epoch_loss / len(self.loader)

class NuScenesMultiModalDataset(Dataset):
    def __init__(self, nusc, camera_name='CAM_FRONT', lidar_name='LIDAR_TOP', past_seconds=2, future_seconds=6, frequency=2):
        self.nusc = nusc
        self.camera_name = camera_name
        self.lidar_name = lidar_name
        
        self.past_steps = int(past_seconds * frequency)    # 2s * 2Hz = 4 steps
        self.future_steps = int(future_seconds * frequency) # 6s * 2Hz = 12 steps
        
        # Standard Camera Normalization
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # Standard size for ResNet18
            transforms.ToTensor(), 
            # Modification of the value from the range [0,255] into [0,1] + [H,W,C] -> [C,H,W]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            # Normalization around 0, mean and std from ImageNet dataset
        ])
        
        # Build list of trackable vehicle instances
        self.samples_list = self._build_trajectory_dataset()
        
        print(f"{'=' * 15} MULTI-MODAL TRAJECTORY DATASET {'=' * 15}")
        print(f"Dataset loaded with {len(self)} vehicle tracks.")
        print(f"{'-' * 50}\n")

    def _build_trajectory_dataset(self):
        valid_sequences = []
        for sample in self.nusc.sample:
            for ann_token in sample['anns']: # [Token_car, Token_ped, Token_car2, etc]
                ann = self.nusc.get('sample_annotation', ann_token) # Get data from the Object content
                # Only track moving cars with enough sequential history and future frames
                if 'vehicle.car' in ann['category_name'] and self._has_enough_history_and_future(ann_token):
                    valid_sequences.append({
                        'sample_token': sample['token'], # Scene Token
                        'ann_token': ann_token, # Object token (Token allocated to the object changing for each scene sample)
                        'instance_token': ann['instance_token'] # Object Unique ID (do not change)
                    })
        return valid_sequences

    def _has_enough_history_and_future(self, ann_token):
        # Verify past sequence depth
        curr_ann = self.nusc.get('sample_annotation', ann_token)
        # Rewind the past of the Object and check if associated previous scene exist or not
        # Saying in other words, verify if the object is still present in the past steps (4 frames before)
        for _ in range(self.past_steps):
            if not curr_ann['prev'] or curr_ann['prev'] == '':
                return False # Do not consider the Object for the training (not possible to track)
            curr_ann = self.nusc.get('sample_annotation', curr_ann['prev'])
            
        # Verify future sequence depth
        # Rewind the future of the Object and check if associated future scene exist or not
        # Saying in other words, verify if the object is still present in the future steps (12 frames after)
        curr_ann = self.nusc.get('sample_annotation', ann_token)
        for _ in range(self.future_steps):
            if not curr_ann['next'] or curr_ann['next'] == '':
                return False
            curr_ann = self.nusc.get('sample_annotation', curr_ann['next'])
        return True

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        meta = self.samples_list[idx]
        sample = self.nusc.get('sample', meta['sample_token'])
        
        # 1. LOAD CAMERA DATA
        cam_token = sample['data'][self.camera_name]
        img_path, _, _ = self.nusc.get_sample_data(cam_token)
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.img_transform(image) # [3, 224, 224]

        # 2. LOAD LIDAR POINT CLOUD
        lidar_token = sample['data'][self.lidar_name]
        lidar_path, _, _ = self.nusc.get_sample_data(lidar_token)
        pc = LidarPointCloud.from_file(lidar_path)
        # Transpose to get [L, 4] -> (x, y, z, intensity)
        point_cloud_tensor = torch.tensor(pc.points.T, dtype=torch.float32)

        # 3. COORDINATE CALCULATIONS (RELATIVE TRANSLATIONS)
        curr_ann = self.nusc.get('sample_annotation', meta['ann_token'])
        origin_xy = np.array(curr_ann['translation'][:2])

        # Past Trajectory History
        history_coords = []
        temp_ann = curr_ann
        for _ in range(self.past_steps):
            temp_ann = self.nusc.get('sample_annotation', temp_ann['prev'])
            history_coords.append(np.array(temp_ann['translation'][:2]) - origin_xy)
        history_coords.reverse() # Reverse the list to get the oldest scene at the first index
        history_tensor = torch.tensor(np.array(history_coords), dtype=torch.float32) # [4, 2]

        # Future Trajectory Ground Truth
        future_coords = []
        temp_ann = curr_ann
        for _ in range(self.future_steps):
            temp_ann = self.nusc.get('sample_annotation', temp_ann['next'])
            future_coords.append(np.array(temp_ann['translation'][:2]) - origin_xy)
        future_tensor = torch.tensor(np.array(future_coords), dtype=torch.float32) # [12, 2]

        return image_tensor, point_cloud_tensor, history_tensor, future_tensor
    
def multimodal_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    # Leave point clouds as a raw list of varying-length tensors
    point_clouds = [item[1] for item in batch]  # Cannot stak tensor of different size ! pointcloud varies from 1 scene to another
    histories = torch.stack([item[2] for item in batch], dim=0)
    futures = torch.stack([item[3] for item in batch], dim=0)
    
    return images, point_clouds, histories, futures

def get_dataloader(dataroot, version='v1.0-mini', batch_size=1):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    dataset = NuScenesMultiModalDataset(nusc)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=multimodal_collate)