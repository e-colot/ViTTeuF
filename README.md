# ViTTeuF - Vision Transformer Trajectory Forecasting

Uses the NuScenes dataset (v1.0-mini) to train a model that performs trajectory prediction. The architecture is based on a vision transformer.

## Setup

1. Download the mini [dataset](https://www.nuscenes.org/nuscenes#) and extract it to the `data/` folder at root    

2. Install the NuScenes devkit with the required libraries

    pip install nuscenes-devkit torch torchvision

## Repository organization

`data/` stores the dataset, not synced on GitHub
`papers/` contains documentation pdfs
`report/`contains our own report
`tools/` contains scripts that can be called (e.g. `train.py`)
`utils/` contains scripts called by other scripts (e.g. `load_utils.py`)
`model/` contains files relative to the model

## Calling Tools 

To ensure all the scripts can be called from eachother, every script must be run as a module **from root**:

    python -m tools.test

