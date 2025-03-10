# Towards Robust Autonomous Driving: Out-of-Distribution Object Detection in Bird's Eye View Space

This repository contains the implementation of our research on Out-of-Distribution (OOD) object detection in Bird's Eye View (BeV) space for autonomous driving. We introduce two novel approaches for handling unknown objects by adding random patches and OOD objects into the BeV projection. Our work enhances the robustness and safety of autonomous vehicles by enabling better detection of known and unknown objects. The visual results are presented in MP4 files uploaded in the repository.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Experiment Details](#experiment-details)
- [Results](#results)
- [Contributing](#contributing)

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

We use the [NuScenes](https://www.NuScenes.org/) dataset for our experiments. This dataset contains annotated data from urban environments, collected using 360-degree cameras, LiDAR, and radar sensors.

- **Download Instructions**: Follow the instructions on the [NuScenes website](https://www.NuScenes.org/download) to download the dataset.
- **Data Preparation**: Place the dataset files in a folder named `data/NuScenes/` within the repository.

## Usage

### Training the Model

To train the model, run the following command:

```sh
python main.py train mini --dataroot=data/NuScenes --logdir=./runs --gpuid=-1

--dataroot: Path to the NuScenes dataset.
--logdir: Directory to store training logs.
--gpuid: GPU ID to use for training (-1 for CPU).


Evaluating the Model
To evaluate the trained model, use:

sh
Copy code
python main.py eval_model_iou mini --modelf=model525000.pt --dataroot=data/NuScenes
--modelf: Path to the trained model file.
--dataroot: Path to the NuScenes dataset.
Visualizing Model Predictions
To visualize the model's predictions:

sh
Copy code
python main.py viz_model_preds mini --modelf=model525000.pt --dataroot=data/NuScenes --map_folder=data/NuScenes
--modelf: Path to the trained model file.
--dataroot: Path to the NuScenes dataset.
--map_folder: Folder containing map data.
Experiment Details
Base Model: Our implementation is based on the Lift-Splat-Shoot framework.
Patch Introduction: Random patches are used to cover vehicles during inference to simulate occlusion.
OOD Object Introduction: Unknown objects from the COCO dataset are introduced during training to enhance the model's robustness.
Hardware: Our experiments were conducted using NVIDIA RTX 6000 Ada Generation GPUs.

## Experiment Details

Our experiments focus on improving Out-of-Distribution (OOD) object detection in the Bird's Eye View (BeV) space using our **NuScenesOOD** dataset and **patch introduction** techniques.

### Base Model

We build upon the **Lift-Splat-Shoot (LSS)** framework, which projects multi-camera images into a BeV representation for scene understanding in autonomous driving.

### Patch Introduction for OOD Detection

- **Motivation**: Patches are introduced to simulate unknown objects by occluding known vehicles.
- **Implementation**:
  - During **training**, patches are randomly placed over **10% of vehicles** in each scene.
  - Vehicles are selected using unique instance tokens to ensure **consistency across frames**.
  - The model is trained to classify objects as **known** or **unknown** without explicit OOD labels.

### Introducing OOD Objects

- **NuScenesOOD Dataset**:
  - Augmented vehicles with patterns from **StyleGAN** to create **OOD objects**.
  - Additional **random objects** (e.g., fire hydrants, animals) from the **COCO dataset** to simulate real-world OOD challenges.
  - BeV projection is modified to represent unknown objects separately.

### Training Setup

- **Batch Size**: 8
- **Learning Rate**: 0.001 (cosine decay)
- **Optimizer**: AdamW
- **Hardware**: NVIDIA RTX 6000 Ada GPUs
- **Training Time**: ~12 hours for 50 epochs

---

## Results

Our method **significantly improves OOD object detection** while maintaining strong performance for known objects.

### 1. Object Segmentation Performance (IoU)

| Method | Car (IoU) | Vehicles (IoU) |
|--------|----------|---------------|
| CNN \cite{mani2020monolayout} | 22.78 | 24.25 |
| Fiery \cite{fiery} | - | 35.08 |
| Frozen Encoder \cite{roddick2020pon} | 25.51 | 26.83 |
| OFT \cite{roddick2018oft} | 29.72 | 30.05 |
| Lift-Splat-Shoot \cite{philion2020lift} | 32.06 | 32.07 |
| PON \cite{roddick2020pon} | 24.70 | - |
| FISHING \cite{hendy2020fishingnet} | - | 30.00 |
| SimpleBEV \cite{simplebev} | - | **55.7** |
| **Ours (NuScenesOOD + Patches)** | **35.67** | **36.32** |

➡ **Our approach improves segmentation for unknown objects while maintaining high accuracy for known vehicles.**

### 2. Impact of Patch Introduction on IoU

| Patch Type | Mean IoU | Observation |
|------------|----------|-------------|
| Random (10% for each scene) | 11.0 | Significant IoU drop due to patches |
| Vehicles without Patch | 28.0 | Decreased IoU due to occlusions |
| Optimized Placement | **49.0** | Best performance with refined patching |

➡ **Patching significantly impacts segmentation and forces the model to learn better OOD object detection.**

### 3. OOD Object Segmentation Results

| Object Type | Mean IoU | Observation |
|------------|---------|-------------|
| Known Objects (Vehicles) | **47.56** | High accuracy for regular vehicles |
| OOD Objects (Augmented Patterns) | **38.91** | Strong performance in detecting unknown objects |

➡ **Our model successfully learns to segment unknown objects from the NuScenesOOD dataset!**

### 4. Visual Results

#### ✅ BeV Segmentation Without Patches
![BeV Without Patches](images/bev_no_patch.png)

#### ❌ BeV Segmentation With Patches (Simulating OOD)
![BeV With Patches](images/bev_patch.png)

#### 🚀 NuScenesOOD Dataset Testing on YOLO-V8 (Fails to Detect OOD)
![NuScenesOOD on YOLO](images/yolo_fail.png)

➡ **Even state-of-the-art object detection models struggle with OOD detection, highlighting the importance of our approach!**

---

## Conclusion

Our research presents a **novel approach to detecting unknown objects in autonomous driving using Bird’s Eye View (BeV) perception.** By introducing **random patches** and **OOD objects** through our **NuScenesOOD dataset**, we demonstrate significant improvements in **OOD detection** while maintaining high accuracy for known objects.

## Model Performance on NuScenes and NuScenesOOD

### Lift-Splat-Shoot on NuScenes
![LSS NuScenes](https://raw.githubusercontent.com/asad-python/OOD-in-BEV/main/data/nuscenes/lss_nuscenes-ezgif.com-video-to-gif-converter.gif)

### Lift-Splat-Shoot on NuScenesOOD
![LSS OOD NuScenes](https://raw.githubusercontent.com/asad-python/OOD-in-BEV/main/data/nuscenes/lss_OOD_nuscenes-ezgif.com-video-to-gif-converter.gif)

### YOLO on StyleGAN-Augmented Dataset
![StyleGAN YOLO](https://raw.githubusercontent.com/asad-python/OOD-in-BEV/main/data/nuscenes/STYLE_GAN_YOLO-ezgif.com-video-to-gif-converter.gif)

### Detection on StyleGAN-Augmented Dataset
![Detection StyleGAN YOLO](https://raw.githubusercontent.com/asad-python/OOD-in-BEV/main/data/nuscenes/DET_STYLE_GAN_YOLO1-ezgif.com-video-to-gif-converter.gif)


🚀 **NuScenesOOD will be released soon! Stay tuned for updates!**




