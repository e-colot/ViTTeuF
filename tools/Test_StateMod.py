import os
import sys

##### IF WITHOUT BASH SCRIPT ######
# Add Directory folder in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.MultiStateModLoss import SpatioTemporalTrajectoryLoss
from model.MultiStateModTraj import AdvancedTrajectoryPredictor
from utils.load_utils_StateModal import Feeder
import torch
import matplotlib.pyplot as plt


if __name__ == "__main__":

    num_traj_pred = 5

    weights_dir = "./checkpoints"
    os.makedirs(weights_dir, exist_ok=True) # Creates the folder if it doesn't exist
    best_weights_path = os.path.join(weights_dir, "best_multimodal_model.pth")

    # 1. Setup Runtime Environments
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_to_nuscenes_mini = "./data"

    # 2. Instantiate Model Pieces 
    model = AdvancedTrajectoryPredictor(d_model=128, num_modes=num_traj_pred, future_steps=12)
    loss_function = SpatioTemporalTrajectoryLoss(cls_weight=1.0,state_weight=2.0, num_mode=num_traj_pred)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 3. Create Feeder Loop Pipeline
    feeder = Feeder(
        path_to_data=path_to_nuscenes_mini,
        model=model,
        loss_fn=loss_function,
        optimizer=optimizer,
        device=device,
        batch_size=1 # Kept at 1 for pointcloud compatibility
    )

    epoch_max = 10

    avg_loss_list = []
    avg_loss_cls = []
    avg_loss_ts = []
    avg_loss_s = []
    MinADE_list = []
    MinFDE_list = []
    Missrate_list = []
    avg_loss_best = 300000
    # Train and Validate
    print("Starting training and validation loop...")
    for epoch in range(epoch_max):
        print(f"\n--- Epoch {epoch+1}/{epoch_max} ---")

        loss_out = feeder.train_epoch()
        avg_loss_list.append(loss_out['avg_loss'])
        avg_loss_cls.append(loss_out['avg_cls'])
        avg_loss_s.append(loss_out['avg_s'])
        avg_loss_ts.append(loss_out['avg_ts'])

        metrics_out = feeder.validation_run()
        MinADE_list.append(metrics_out['MinADE'])
        MinFDE_list.append(metrics_out['MinFDE'])
        Missrate_list.append(metrics_out['Missrate'])

        if avg_loss_best > loss_out['avg_loss'] :
            avg_loss_best = loss_out['avg_loss']
            torch.save(model.state_dict(), best_weights_path)

        print(f"Epoch {epoch+1} Complete. Average Loss: {loss_out['avg_loss']:.4f}")
        print(f"############## Validation Results for K = {num_traj_pred} ##############")
        print(f"Validation minADE  : {metrics_out['MinADE']:.3f} meters")
        print(f"Validation minFDE  : {metrics_out['MinFDE']:.3f} meters")
        print(f"Validation Miss Rate: {metrics_out['Missrate'] * 100:.1f}%")
            
    print("\nTraining Finished!")
    print(f"Loss per training iteration is :{[x for x in avg_loss_list]}")
    print(f"Loss_cls per training iteration is :{[x for x in avg_loss_cls]}")
    print(f"Loss_ts per training iteration is :{[x for x in avg_loss_ts]}")
    print(f"Loss_s per training iteration is :{[x for x in avg_loss_s]}")
    print(f"############## Validation Results for K = {num_traj_pred} ##############")
    print(f"MinADE per training epoch is :{[x for x in MinADE_list]}")
    print(f"MinFDE per training epoch is :{[x for x in MinFDE_list]}")
    print(f"Missrate per training epoch is :{[x for x in Missrate_list]}")
    

