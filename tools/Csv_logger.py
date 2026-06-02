import os
import json
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.abspath(os.path.join(current_dir, ".."))
dir_results_path = os.path.join(dir_path, "Results")
json_path = os.path.join(dir_results_path,"Output_K_6_ClassWeight_1_StateWeight_1_Softtarget_15_dmodel_256_epoch_15_no_vel.json")

plot_metric_dir = "./Results/Plots"
plot_path = os.path.join(plot_metric_dir,"Output_K_6_ClassWeight_1_StateWeight_1_Softtarget_15_dmodel_256_epoch_15_no_vel")
os.makedirs(plot_path,exist_ok=True)

with open(json_path,"r") as f:
    Dataset = json.load(f)

if isinstance(Dataset, str):
    Dataset = json.loads(Dataset)

MinADE_list = []
MinFDE_list = []
Missrate_list = []
Loss_list = []
Loss_cls_list = []
Loss_s_list = []
Loss_ts_list = []

for data in Dataset.values():

    if isinstance(data, str):
        data = json.loads(data)

    MinADE_list.append(data["MinADE"])
    MinFDE_list.append(data["MinFDE"])
    Missrate_list.append(data["Missrate"])
    Loss_list.append(data["Total_Loss"])
    Loss_cls_list.append(data["Class_Loss"])
    Loss_s_list.append(data["State_Loss"])
    Loss_ts_list.append(data["Reg_Loss"])

epochs = np.arange(1, len(Dataset) + 1)

######### PERFORMANCE METRICS PLOT #########

plt.figure(figsize=(10, 7))
plt.plot(epochs, MinADE_list, '-o',label=f"MinADE",markersize=4)
plt.plot(epochs, MinFDE_list, '-o',label=f"MinFDE",markersize=4)
plt.title(f"Inference Performance")
plt.xlabel("Training Epoch", fontsize=12)
plt.ylabel("Meters", fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')
plot_filename = os.path.join(plot_path, f"Metrics_Perf.png")
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.close() # Free up memory

plt.figure(figsize=(10, 7))
plt.plot(epochs, Missrate_list, '-o',label=f"Missrate",markersize=4)
plt.title(f"Inference Performance")
plt.xlabel("Training Epoch", fontsize=12)
plt.ylabel("Percents", fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')
plot_filename = os.path.join(plot_path, f"Missrate_Perf.png")
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.close() # Free up memory

######### LOSS PLOT #########

plt.figure(figsize=(10, 7))
plt.plot(epochs, Loss_list, '-o',label=f"Total Loss",markersize=4)
plt.plot(epochs, Loss_cls_list, '-o',label=f"Class loss",markersize=4)
plt.plot(epochs, Loss_ts_list, '-o',label=f"Regression Loss",markersize=4)
plt.plot(epochs, Loss_s_list, '-o',label=f"State Loss",markersize=4)
plt.title(f"Loss Evolution Per Epoch")
plt.xlabel("Training Epoch", fontsize=12)
plt.ylabel("Loss Magnitude", fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')
plot_filename = os.path.join(plot_path, f"Loss_evol.png")
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.close() # Free up memory

plt.figure(figsize=(10, 7))
plt.plot(epochs, Loss_cls_list, '-o',label=f"Class loss",markersize=4)
plt.plot(epochs, Loss_s_list, '-o',label=f"State Loss",markersize=4)
plt.title(f"Loss Evolution Per Epoch")
plt.xlabel("Training Epoch", fontsize=12)
plt.ylabel("Loss Magnitude", fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')
plot_filename = os.path.join(plot_path, f"Cls_State_evol.png")
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.close() # Free up memory




