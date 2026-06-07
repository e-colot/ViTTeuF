import os
import sys
import json

##### IF WITHOUT BASH SCRIPT ######
# Add Directory folder in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.MultiStateModLoss import SpatioTemporalTrajectoryLoss
from model.MultiStateModTraj import AdvancedTrajectoryPredictor
from utils.load_utils_StateModal import Feeder
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict


if __name__ == "__main__":

    ################################
    ######## PARAMETER SET-UP ######
    ################################
    # Traj Nbr selection
    num_traj_pred = 10
    class_w = 1.0
    state_w = 1.7
    variance = 1.0
    features_model = 256
    # Training or Validation 
    visu_flag = True
    training_flag = False
    # Metric K selection
    K_mode = 1
    ###############################

    # Output path making

    weights_dir = "./checkpoints"
    os.makedirs(weights_dir, exist_ok=True) # Creates the folder if it doesn't exist
    best_weights_path = os.path.join(weights_dir, "best_multimodal_model.pth")

    if training_flag == True :

        output_dir = "./Results"
        os.makedirs(output_dir, exist_ok=True) # Creates the folder if it doesn't exist
        out_json_path = os.path.join(output_dir, f"Output_K_{num_traj_pred}_ClassWeight_{int(class_w)}_StateWeight_{int(state_w)}.json")

    plot_dir = "./Plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Input data path and Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_to_nuscenes_mini = "./data"

    # Initialization of the model and loss classes
    model = AdvancedTrajectoryPredictor(d_model=features_model, num_modes=num_traj_pred, future_steps=12)
    loss_function = SpatioTemporalTrajectoryLoss(cls_weight=class_w,state_weight=state_w, num_mode=num_traj_pred, variance=variance)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create Feeder - Dataloader for Train and Validation Data
    feeder = Feeder(
        path_to_data=path_to_nuscenes_mini,
        model=model,
        loss_fn=loss_function,
        optimizer=optimizer,
        device=device,
        save_dir=plot_dir,
        visu_flag = visu_flag,
        K = K_mode,
        batch_size=1 # Kept at 1 for pointcloud compatibility
    )

    epoch_max = 12

    ML_out = OrderedDict()

    avg_loss_list = []
    avg_loss_cls = []
    avg_loss_ts = []
    avg_loss_s = []
    MinADE_list = []
    MinFDE_list = []
    Missrate_list = []
    avg_loss_best = 300000

    if training_flag == True:

        # Train and Validate
        print("Starting training and validation loop...")
        for epoch in range(epoch_max):
            print(f"\n--- Epoch {epoch+1}/{epoch_max} ---")

            ML_out[epoch] = OrderedDict()

            loss_out = feeder.train_epoch()
            avg_loss_list.append(loss_out['avg_loss'])
            avg_loss_cls.append(loss_out['avg_cls'])
            avg_loss_s.append(loss_out['avg_s'])
            avg_loss_ts.append(loss_out['avg_ts'])

            print(f"Epoch {epoch+1} Complete. Average Loss: {loss_out['avg_loss']:.4f}")
            print(f" Training Reg Loss : {loss_out['avg_ts']}")
            print(f" Training Class Loss : {loss_out['avg_cls']}")
            print(f" Training State Loss: {loss_out['avg_s']}")

            metrics_out = feeder.validation_run()

            ML_out[epoch]['MinADE'] = metrics_out['MinADE']
            ML_out[epoch]['MinFDE'] = metrics_out['MinFDE']
            ML_out[epoch]['Missrate'] = metrics_out['Missrate']
            ML_out[epoch]['Total_Loss'] = loss_out['avg_loss']
            ML_out[epoch]['Class_Loss'] = loss_out['avg_cls']
            ML_out[epoch]['Reg_Loss'] = loss_out['avg_ts']
            ML_out[epoch]['State_Loss'] = loss_out['avg_s']

            MinADE_list.append(metrics_out['MinADE'])
            MinFDE_list.append(metrics_out['MinFDE'])
            Missrate_list.append(metrics_out['Missrate'])

            if avg_loss_best > loss_out['avg_loss'] :
                avg_loss_best = loss_out['avg_loss']
                torch.save(model.state_dict(), best_weights_path)

            print(f"############## Validation Results for K = {K_mode} ##############")
            print(f" Validation MinADE : {metrics_out['MinADE']}")
            print(f" Validation MinFDE : {metrics_out['MinFDE']}")
            print(f" Validation Miss Rate: {metrics_out['Missrate']}")
        
    
        print("\nTraining Finished!")
        print(f"############## Validation Results for K = {K_mode} ##############")
        print(f" minADE per training iteration is  : {[x for x in MinADE_list]}")
        print(f" minFDE per training iteration is  : {[x for x in MinFDE_list]}")
        print(f" Miss rate per training iteration is  : {[x for x in Missrate_list]}")
        print(f"Loss per training iteration is :{[x for x in avg_loss_list]}")
        print(f"Loss_cls per training iteration is :{[x for x in avg_loss_cls]}")
        print(f"Loss_ts per training iteration is :{[x for x in avg_loss_ts]}")
        print(f"Loss_s per training iteration is :{[x for x in avg_loss_s]}")

        with open(out_json_path,'w') as f:
            json.dump(ML_out, f, indent=4)

    else :
        if os.path.exists(best_weights_path):
            model.load_state_dict(torch.load(best_weights_path, map_location=device))
        
            metrics_out = feeder.validation_run()

            print(f"############## Validation Results for K = {K_mode} ##############")
            print(f"Validation minADE  : {metrics_out['MinADE']:.3f} meters")
            print(f"Validation minFDE  : {metrics_out['MinFDE']:.3f} meters")
            print(f"Validation Miss Rate: {metrics_out['Missrate'] * 100:.1f}%")



    

