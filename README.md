# Trajectory Forecasting Project

**Students:** Amaury Arico, Emmeran Colot, Loric Lungu-Embanzu

---

## 1. System Requirements & Prerequisites

To run this pipeline standalone, ensure your environment meets the following requirements:

* **Python Version:** 3.11 (Recommended)
* **Required Libraries:**
    * `torch`
    * `torchvision`
    * `nuscenes`
    * `os`
    * `json`
    * `scipy`
    * `matplotlib`
    * `numpy`

---

## 2. File Structure

Your project directory should be organized exactly as shown below:

```text
Project/
├── checkpoints/
│   └── [weights.path will be saved here]
├── data/
│   ├── [nuscenes mini-1.0 data]
│   └── [map_expansion v1.3 data]
├── model/
│   ├── MultiStateModTraj.py
│   ├── MultiStateModelLoss.py
│   ├── Pointpillar.py
│   └── LSTM_block.py
├── Plots/
│   └── [Scenes will be saved here]
├── Results/
│   └── [Logs will be saved here]
├── tools/
│   └── Test_StateMod.py
└── utils/
    └── load_utils_StateModal.py
```

---

## 3. Pipeline Python Files

The core pipeline consists of 6 active modules (plus one educational module):

1. **`Test_StateMod.py`**: The main run script.
2. **`load_utils_StateModal.py`**: The data loader module.
3. **`MultiStateModTraj.py`**: The model architecture build module.
4. **`MultiStateModelLoss.py`**: The custom loss function build module.
5. **`Pointpillar.py`**: PointPillar backbone module.
6. **`LSTM_block.py`**: The LSTM temporal module.
7. **`Resnet.py`** *\*(Optional)*: Developed for ResNet architecture understanding; not actively used in the main pipeline.

---

## 4. How to Run

Follow these steps to execute the pipeline:

1. Open the run script `Test_StateMod.py`
2. Select the wanted parameters for the simulation - latest model values shown below:
    ### Traj Nbr selection
        num_traj_pred = 10
        class_w = 1.0
        state_w = 1.7
        variance = 1.0
        features_model = 256
    ### Metric K selection
        K_mode = 1
3. Set the training_flag (Training or Validation test) + the scene visualization flag:

    ### Training (True) or Validation (False)
        training_flag = False
    ### Scene visu saving 
        visu_flag = True

4. Run the script (standalone)
