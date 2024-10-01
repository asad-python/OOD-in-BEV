# Towards Robust Autonomous Driving: Out-of-Distribution Object Detection in Bird's Eye View Space

This repository contains the implementation of our research on improving Out-of-Distribution (OOD) object detection in Bird's Eye View (BeV) space for autonomous driving. We introduce two novel approaches for handling unknown objects by adding random patches and OOD objects into the BeV projection. Our work enhances the robustness and safety of autonomous vehicles by enabling better detection of known and unknown objects.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Experiment Details](#experiment-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In autonomous driving, understanding the surrounding environment is critical for safety. Most object detection systems are designed to detect known objects, but they may struggle to identify unknown or novel objects, which can pose a risk. This project introduces two methods for detecting unknown objects in the Bird's Eye View (BeV) space:

1. **Introducing Random Patches** - Covering known objects with random patches to assess the model's robustness in detecting hidden or occluded objects.
2. **Introducing Out-of-Distribution (OOD) Objects** - Inserting unknown objects, such as animals or large items, into the environment to train the model for OOD detection.

This work is built on the Lift-Splat-Shoot framework and extends it for better handling of unknown objects in autonomous driving environments.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.12.0 or higher
- CUDA 11.0 or higher (optional, for GPU acceleration)

### Setting Up the Environment

1. Clone the repository:

    ```sh
    git clone https://github.com/asad-python/OOD_PATCH.git
    cd src
    ```

2. Create a virtual environment:

    ```sh
    python -m venv env
    source env/bin/activate  # Linux/macOS
    env\Scripts\activate     # Windows
    ```

3. Install dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Dataset

We use the [nuScenes](https://www.nuscenes.org/) dataset for our experiments. This dataset contains annotated data from urban environments, collected using 360-degree cameras, LiDAR, and radar sensors.

- **Download Instructions**: Follow the instructions on the [nuScenes website](https://www.nuscenes.org/download) to download the dataset.
- **Data Preparation**: Place the dataset files in a folder named `data/nuScenes/` within the repository.

## Usage

### Training the Model

To train the model, run the following command:

```sh
python main.py train mini --dataroot=data/nuscenes --logdir=./runs --gpuid=-1

--dataroot: Path to the nuScenes dataset.
--logdir: Directory to store training logs.
--gpuid: GPU ID to use for training (-1 for CPU).


Evaluating the Model
To evaluate the trained model, use:

sh
Copy code
python main.py eval_model_iou mini --modelf=model525000.pt --dataroot=data/nuscenes
--modelf: Path to the trained model file.
--dataroot: Path to the nuScenes dataset.
Visualizing Model Predictions
To visualize the model's predictions:

sh
Copy code
python main.py viz_model_preds mini --modelf=model525000.pt --dataroot=data/nuscenes --map_folder=data/nuscenes
--modelf: Path to the trained model file.
--dataroot: Path to the nuScenes dataset.
--map_folder: Folder containing map data.
Experiment Details
Base Model: Our implementation is based on the Lift-Splat-Shoot framework.
Patch Introduction: Random patches are used to cover vehicles during inference to simulate occlusion.
OOD Object Introduction: Unknown objects from the COCO dataset are introduced during training to enhance the model's robustness.
Hardware: Our experiments were conducted using NVIDIA RTX 6000 Ada Generation GPUs.

