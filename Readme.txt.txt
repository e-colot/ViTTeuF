======================================
PROJECT README: Trajectory Forecasting Project
======================================
Students: Amaury Arico, Emmeran Colot, Loric Lungu-Embanzu

--------------------------------------------------
1. SYSTEM REQUIREMENTS & PREREQUISITES
--------------------------------------------------
To run this pipeline standalone, your environment must have:

* Python version used : 3.11 (just for indication!)
* Python Libraries:
    - torch
    - torchvision
    - nuscenes
    - os
    - json
    - scipy
    - matplotlib
    - numpy

-------------------------------------------------
2. FILE STRUCTURE 
-------------------------------------------------

Folder should be structured in the following way :

-Project
----checkpoints
-------** weight.path will be saved here
----data
-------* import nuscenes mini-1.0 data
-------* import map_expansion v1.3 data
----model
---------MultiStateModTraj.py
---------MultiStateModelLoss.py
---------Pointpillar.py
---------LSTM_block.py
----Plots
-------** Scene will be saved here
----Results
-------** Logs will be saved here
----tools
---------Test_StateMod.py
----utils
---------load_utils_StateModal.py

-------------------------------------------------
3. PIPELINE WORKFLOW
-------------------------------------------------

The core pipeline consists of 6 modules:

1. The run script (Test_StateMod.py)
2. The loader module (load_utils_StateModal.py)
3. The model build module (MultiStateModTraj.py)
4. The loss build module (MultiStateModelLoss.py)
5. Pointpillar module (Pointpillar.py)
6. The LSTM module (LSTM_block.py)
7*. The Resnet module (Resnet.py) - not used in the pipeline but developed for Resnet architecture understanding

-------------------------------------------------
4. HOW TO RUN ?
-------------------------------------------------

1. Open the run script (Test_StateMod.py)
2. Select the wanted parameters for the simulation - latest model values shwon below :
# Traj Nbr selection
    num_traj_pred = 10
    class_w = 1.0
    state_w = 1.7
    variance = 1.0
    features_model = 256
# Metric K selection
    K_mode = 1
3. Set the training_flag (Training or Validation test) + the scene visualization flag :

# Training (True) or Validation (False)
    training_flag = False
# Scene visu saving 
    visu_flag = True

4. Run the script (standalone)