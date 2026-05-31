import os
import sys

##### IF WITHOUT BASH SCRIPT ######
# Add Directory folder in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.MultimodalLoss import MultimodalTrajectoryLoss
from model.MultimodalTraj import MultiModalTrajectoryPredictor
from utils.load_utils_modal import Feeder
import torch
import matplotlib.pyplot as plt


if __name__ == "__main__":

    weights_dir = "./checkpoints"
    os.makedirs(weights_dir, exist_ok=True) # Creates the folder if it doesn't exist
    best_weights_path = os.path.join(weights_dir, "best_multimodal_model.pth")

    # 1. Setup Runtime Environments
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_to_nuscenes_mini = "./data"

    # 2. Instantiate Model Pieces 
    model = MultiModalTrajectoryPredictor(d_model=128, num_modes=3, future_steps=12)
    loss_function = MultimodalTrajectoryLoss(cls_weight=1.0)
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

    avg_loss_list = []
    avg_loss_best = 300000
    # 4. Train!
    print("Starting training loop...")
    for epoch in range(5):
        print(f"\n--- Epoch {epoch+1}/5 ---")
        avg_loss = feeder.train_epoch()
        avg_loss_list.append(avg_loss)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        if avg_loss_best > avg_loss :
            avg_loss_best = avg_loss
            torch.save(model.state_dict(), best_weights_path)
            
    print("\nTraining Finished!")
