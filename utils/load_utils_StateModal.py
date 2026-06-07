import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.bitmap import BitMap
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from pyquaternion import Quaternion


class Feeder:
    def __init__(self, path_to_data, model, loss_fn, optimizer, device, save_dir, visu_flag, K, batch_size=1):
        self.train_loader = get_dataloader(dataroot=path_to_data, batch_size=batch_size, split = 'train')
        self.val_loader = get_dataloader(dataroot=path_to_data, batch_size=batch_size, split = 'val')
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.traj_visu = visu_flag
        self.K = K

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        total_loss_cls = 0.0
        total_loss_s = 0.0
        total_loss_ts = 0.0
        
        for batch_idx, (images, point_clouds, histories, futures, map) in enumerate(self.train_loader):
            images = images.to(self.device)         # [Batch, 3, 224, 224]
            histories = histories.to(self.device)   # [Batch, 4, 2]
            futures = futures.to(self.device)       # [Batch, 12, 2]
            
            # Since batch_size=1, grab the first raw point cloud item from the collated list
            pc_single = point_clouds[0].to(self.device) # [L, 4] | list[tensor[L,4]]-> list[0] for batch_size=1

            self.optimizer.zero_grad()
            
            #pred_trajs, pred_logits = self.model(images, pc_single, histories)
            pred_trajectories, pred_headings, pred_logits = self.model(images, pc_single, histories)
            
            #loss, loss_reg, loss_cls = self.loss_fn(pred_trajs, pred_logits, futures)
            loss, loss_ts, loss_s, loss_cls = self.loss_fn(pred_trajectories,pred_headings, pred_logits, futures)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            total_loss_cls += loss_cls.item()
            total_loss_s += loss_s.item()
            total_loss_ts += loss_ts.item()
            
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx:03d} | Loss: {loss.item():.4f} (Reg: {loss_ts.item():.3f}, Cls: {loss_cls.item():.3f}, State : {loss_s:.3f})")
                if loss_ts.item() > 100 :
                    model_guess = torch.max(pred_logits, dim=1)[1].item()
                    print(f" Mode prediction is {model_guess}")
        

                
        return {
            "avg_loss": epoch_loss / len(self.train_loader),
            "avg_cls": total_loss_cls / len(self.train_loader),
            "avg_ts": total_loss_ts / len(self.train_loader),
            "avg_s": total_loss_s / len(self.train_loader)
        }
    
    def validation_run(self):
        self.model.eval()

        totalADE = 0.0
        totalFDE = 0.0
        totalMISS = 0.0
        total_samples = 0
        max_plots = 200
        plotted_count = 0

        nusc_instance = self.val_loader.dataset.nusc

        with torch.no_grad():
            for batch_idx, (images, point_clouds, histories, futures, map) in enumerate(self.val_loader):
                images = images.to(self.device)         # [Batch, 3, 224, 224]
                histories = histories.to(self.device)   # [Batch, 4, 2]
                futures = futures.to(self.device)       # [Batch, 12, 2]
                pc_single = point_clouds[0].to(self.device) # [L, 4]

                pred_trajectories,_,pred_logits = self.model(images, pc_single, histories)

                metrics_out = compute_metrics(pred_trajectories, pred_logits, futures, self.K, miss_threshold = 2.0)

                batch_size = futures.size(0)
                totalADE += metrics_out["MinADE"] * batch_size
                totalFDE += metrics_out["MinFDE"] * batch_size
                totalMISS += metrics_out["Missrate"] * batch_size
                total_samples += batch_size

                #if batch_idx % 20 == 0 :
                #   print(f"Batch {batch_idx:03d} |PER SAMPLE => MinADE: {metrics_out['MinADE']:.4f},MinFDE: {metrics_out['MinFDE']:.4f} , Missrate : {metrics_out['Missrate']:.4f}")

                if batch_idx % 25 == 0 and plotted_count < max_plots and self.traj_visu == True:

                    ground_truth_traj = futures[0].cpu().numpy()          # [12, 2]
                    pred_traj = pred_trajectories[0].cpu().numpy()        # [6, 12, 2]
                    pred_mode = pred_logits[0]  # [6]
                    
                    # Convert in percents
                    probs = torch.softmax(pred_mode, dim=-1)

                    best_mode_idx = torch.argmax(probs).item()
                    
                    plt.figure(figsize=(10, 7))

                    # Add ego origin
                    origin = np.array([[0.0, 0.0]])
                    ground_truth_traj = np.vstack([origin, ground_truth_traj])

                    pred_traj_new = np.zeros((pred_traj.shape[0], pred_traj.shape[1] + 1, 2))
                    for k in range(pred_traj.shape[0]):
                        pred_traj_new[k] = np.vstack([origin,pred_traj[k]])

                    plot_filename = os.path.join(self.save_dir, f"Scene_{batch_idx}.png")
                    
                    projection_map(
                        nusc=nusc_instance,
                        map_dict=map[0],
                        ground_truth_traj=ground_truth_traj,
                        pred_traj=pred_traj_new,
                        best_mode_idx=best_mode_idx,
                        probs=probs,
                        plot_filename=plot_filename
                    )
                    
                    plotted_count += 1
                    print(f" Saved in: {plot_filename}")

        return {
            "MinADE": totalADE / total_samples,
            "MinFDE": totalFDE / total_samples,
            "Missrate": totalMISS / total_samples
        }


class NuScenesMultiModalDataset(Dataset):

    # ############### REMINDER NUSCENE API ###############
    # GET SCENE DETAILS DICT : self.nusc.get('scene',sample['scene_token']) || BOOK OF FRAMES
    # GET FRAME DETAILS DICT : self.nusc.get('sample', sample_token) || SINGLE FRAME
    # GET OBJECT DETAILS DICT : self.nusc.get('sample_annotation', ann_token)
    # GET DATA CONTEXT TOKEN : cam_token = sample['data']['CAMERA_FRONT']
    # GET DATA CONTEXT DATA PATH : img_path, _, _ = self.nusc.get_sample_data(cam_token)
    # ####################################################

    def __init__(self, nusc, split, camera_name='CAM_FRONT', lidar_name='LIDAR_TOP', past_seconds=2, future_seconds=6, frequency=2):
        self.nusc = nusc
        self.camera_name = camera_name
        self.lidar_name = lidar_name
        self.split = split

        # Nuscene function splitting scene names between split and train
        self.valid_split_list = create_splits_scenes()[self.split]
        
        self.past_steps = int(past_seconds * frequency)    # 2s * 2Hz = 4 steps
        self.future_steps = int(future_seconds * frequency) # 6s * 2Hz = 12 steps
        
        # Camera Normalization for Rersnet18
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # Standard size for ResNet18
            transforms.ToTensor(), 
            # Modification of the value from the range [0,255] into [0,1] + [H,W,C] -> [C,H,W]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            # Normalization around 0, mean and std from ImageNet dataset
        ])
        
        # Build usable training/validation list
        self.samples_list = self._build_trajectory_dataset()

        if self.split == 'train':
        
            print(f"{'=' * 15} MULTI-MODAL TRAJECTORY DATASET FOR TRAINING {'=' * 15}")
            print(f"Dataset loaded with {len(self)} vehicle tracks.")
            print(f"{'-' * 50}\n")

        else :
            
            print(f"{'=' * 15} MULTI-MODAL TRAJECTORY DATASET FOR VALIDATION {'=' * 15}")
            print(f"Dataset loaded with {len(self)} vehicle tracks.")
            print(f"{'-' * 50}\n")


    def _build_trajectory_dataset(self):
        valid_sequences = []

        # Get Scene name and check if is in the valide scene for associated split (train or val) dataloader
        for sample in self.nusc.sample :
            scene_dict = self.nusc.get('scene',sample['scene_token'])
            if scene_dict['name'] not in self.valid_split_list :
                continue

            for ann_token in sample['anns']: # [Token_car, Token_ped, Token_car2, etc]
                ann = self.nusc.get('sample_annotation', ann_token) # Get data from the Object content
                # Check if Car has sufficient frames history
                if 'vehicle.car' in ann['category_name'] and self._has_enough_history_and_future(ann_token):
                    valid_sequences.append({
                        'sample_token': sample['token'], # Scene Token
                        'ann_token': ann_token, # Object token (Token allocated to the object changing for each scene sample)
                        'instance_token': ann['instance_token'] # Object Unique ID (do not change)
                    })
        return valid_sequences

    def _has_enough_history_and_future(self, ann_token):
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
        
        # LOAD CAMERA DATA
        cam_token = sample['data'][self.camera_name]
        img_path, _, _ = self.nusc.get_sample_data(cam_token)
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.img_transform(image) # [3, 224, 224]

        # LOAD LIDAR POINT CLOUD
        lidar_token = sample['data'][self.lidar_name]
        lidar_path, _, _ = self.nusc.get_sample_data(lidar_token)
        pc = LidarPointCloud.from_file(lidar_path)
        # Transpose to get [L, 4] -> (x, y, z, intensity)
        point_cloud_tensor = torch.tensor(pc.points.T, dtype=torch.float32)

        # COORDINATE CALCULATIONS (RELATIVE TRANSLATIONS)
        curr_ann = self.nusc.get('sample_annotation', meta['ann_token'])
        origin_xy = np.array(curr_ann['translation'][:2])

        # PAST HISTORY
        history_coords = []
        temp_ann = curr_ann
        for _ in range(self.past_steps):
            temp_ann = self.nusc.get('sample_annotation', temp_ann['prev'])
            history_coords.append(np.array(temp_ann['translation'][:2]) - origin_xy)
        history_coords.reverse() # Reverse the list to get the oldest scene at the first index
        history_tensor = torch.tensor(np.array(history_coords), dtype=torch.float32) # [4, 2]

        # FUTURE HISTORY FOR GROUND TRUTH
        future_coords = []
        temp_ann = curr_ann
        for _ in range(self.future_steps):
            temp_ann = self.nusc.get('sample_annotation', temp_ann['next'])
            future_coords.append(np.array(temp_ann['translation'][:2]) - origin_xy)
        future_tensor = torch.tensor(np.array(future_coords), dtype=torch.float32) # [12, 2]

        map_dict = {
            'sample_token': meta['sample_token'],
            'ann_token': meta['ann_token'],
            'origin_xy': origin_xy
        }

        return image_tensor, point_cloud_tensor, history_tensor, future_tensor, map_dict
    
def multimodal_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    # Leave point clouds as a raw list of varying-length tensors
    point_clouds = [item[1] for item in batch]  # Cannot stack tensor of different size ! pointcloud varies from 1 scene to another
    histories = torch.stack([item[2] for item in batch], dim=0)
    futures = torch.stack([item[3] for item in batch], dim=0)
    map = [item[4] for item in batch]
    
    return images, point_clouds, histories, futures, map

def get_dataloader(dataroot, version='v1.0-mini', batch_size=1, split='train'):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    dataset = NuScenesMultiModalDataset(nusc, split)
    shuffle_flag = False
    if split == 'train':
        shuffle_flag == True
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag, collate_fn=multimodal_collate)

@torch.no_grad()
def compute_metrics(pred_traj, pred_logits, gt_traj, K, miss_threshold):

    """
    pred_traj : [Batch, Modes, 12, 2]
    pred_logits : [Batch, Modes]
    gt_traj : [Batch, 12, 2]
    K : K best modes to consider for metrics computation
    miss_threshold : 2 meters of allowable error
    """
    batch_size, _, num_steps, coords = pred_traj.shape
    
    # Search for K best logits / Probs
    _, topk_traj_index = torch.topk(pred_logits, k=K, dim=-1, largest=True, sorted=False) # [Batch, K]
    
    # Keep top K predicted trajectories for metric computation
    pred_traj_index = topk_traj_index.unsqueeze(-1).unsqueeze(-1).expand(batch_size, K, num_steps, coords) # [Batch, K, 12, 2]
    pred_traj_k = torch.gather(pred_traj, dim=1, index=pred_traj_index)

    gt_traj = gt_traj.unsqueeze(1)
    
    # MinADE (Average)
    traj_diff = torch.norm(pred_traj_k - gt_traj, p=2, dim = -1) # [Batch, Modes, 12]

    ADE_per_batch = traj_diff.mean(dim=-1) # [Batch, Modes]
    minADE_per_batch,_ = torch.min(ADE_per_batch, dim=-1) # [Batch]

    # MinFDE (Final)
    FDE_per_batch = traj_diff[...,-1] # [Batch, Modes]
    minFDE_per_batch,_ = torch.min(FDE_per_batch,dim=-1)  # [Batch]

    # Missrate (Beyond the threshold)
    Missrate_per_batch = (minFDE_per_batch > miss_threshold).float() # Count as 1.0 if above the threshold

    return {
        "MinADE" : minADE_per_batch.mean().item(),
        "MinFDE" : minFDE_per_batch.mean().item(),
        "Missrate" : Missrate_per_batch.mean().item()
    }

def projection_map(nusc, map_dict, ground_truth_traj, pred_traj, best_mode_idx, probs, plot_filename):

    ann_data= nusc.get('sample_annotation', map_dict['ann_token'])
    sample_data = nusc.get('sample', map_dict['sample_token'])
    scene_data = nusc.get('scene', sample_data['scene_token'])
    log_data = nusc.get('log', scene_data['log_token'])

    # Get map png from the scene data record
    map_name = log_data['location']
    nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)

    origin_xy = map_dict['origin_xy']

    quat = ann_data['rotation']
    yaw_rad = Quaternion(quat).yaw_pitch_roll[0]
    yaw_deg = np.degrees(yaw_rad)

    # 1. Fetch the map mask layers as raw numpy boolean arrays (0 and 1)
    # patch_box format: (x_center, y_center, width, height)
    patch_box = (origin_xy[0], origin_xy[1], 140, 140) 
    patch_angle = -yaw_rad + (np.pi / 2)
    layer_names = ['drivable_area', 'lane', 'walkway', 'stop_line']
    canvas_size = (1000, 1000)
    
    # Returns a boolean matrix of shape [len(layer_names), 1000, 1000]
    map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
    
    # Merging the masking !
    # Background: White (255, 255, 255)
    bg_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    
    # Drivable layer: Light Gray (220, 220, 220)
    bg_img[map_mask[0] == 1] = [225, 225, 225]
    
    # Lanes layer: Gray/Blue (180, 190, 200)
    bg_img[map_mask[1] == 1] = [180, 190, 200]
    
    # Stop lines layer: Red (255, 100, 100)
    bg_img[map_mask[3] == 1] = [255, 100, 100]

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display custom merged image
    ax.imshow(bg_img, extent=[0, 1000, 1000, 0])

    # Convert meters grid into pixels
    def to_mask_pixel(local_coords):
        # Center of image is 500, 500
        pixel_x = 500 + (local_coords[:, 0] * (1000 / 140))
        pixel_y = 500 + (local_coords[:, 1] * (1000 / 140)) 
        return pixel_x, pixel_y

    # Plot Ground truth into the ax.image
    gt_px, gt_py = to_mask_pixel(ground_truth_traj)
    ax.plot(gt_px, gt_py, 'g-o', label='Ground Truth Target', linewidth=3, markersize=6, zorder=5)
    ax.plot(500, 500, 'ro', markersize=10, label='Agent Position (0,0)', zorder=6)
                        
    # Plot model prediction modes
    for k in range(pred_traj.shape[0]):
        mode_prob = probs[k].item()
        px, py = to_mask_pixel(pred_traj[k])

        if k == best_mode_idx:
            lw = 4.0
            label_text = f'Mode {k} ({mode_prob:.1%}) [MAX PROB]'
            alpha = 1.0
            color = 'blue'
        else:
            lw = 1.5
            label_text = f'Mode {k} ({mode_prob:.1%})'
            alpha = 0.65
            color = 'orange'
                            
        ax.plot(px, py, '-o', color=color, label=label_text, linewidth=lw, markersize=4, alpha=alpha, zorder=4)
        ax.plot(px[-1], py[-1], 'D', color=color, markersize=9 if k == best_mode_idx else 6, alpha=alpha, zorder=4)
                        
    ax.set_title(f"Multi-Modal Tracking Layout | Scene: {map_name}", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1000)
    #ax.set_ylim(1000, 0) # Invert Y axis
    ax.axis('off')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    # Save 
    fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def Convert_to_global_coords(Rot_mat, local_coords, origin_xy):
    return (local_coords @ Rot_mat.T) + origin_xy





