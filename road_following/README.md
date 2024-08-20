# JetRacer


## Overview


JetRacer is an AI-powered autonomous racecar built using NVIDIA's Jetson Nano and JetBot platform. This repository contains various notebooks and scripts that demonstrate how to control the car, perform road following, and optimize models for real-time lane detection and navigation.


## Project Structure


- **`basic_motion.ipynb`**: Demonstrates basic motion control of the JetRacer, including throttle and steering commands. This notebook is a good starting point for understanding how to manually control the car.


- **`interactive_regression.ipynb`**: A notebook for performing interactive regression tasks, likely used for training or testing a model to predict steering angles based on input data.


- **`optimize_model.ipynb`**: This notebook is focused on optimizing the road-following model, particularly by converting it to TensorRT for improved inference speed on the Jetson Nano.


- **`reset_camera_service.ipynb`**: Provides a service to reset the camera settings with the ability to adjust parameters live. This is useful for fine-tuning the camera feed during operation.


- **`road_following.ipynb`**: The main notebook for training and running the road-following AI. It guides the JetRacer to autonomously follow lanes on a track by using a pre-trained neural network model.


- **`road_following_live.ipynb`**: A variation of the `road_following.ipynb` notebook, designed for live parameter adjustments. It includes sliders for real-time tuning of the road-following behavior.


- **`utils.py`**: A utility script containing common functions used across various notebooks, such as image processing, model loading, and controlling the car's movements.


- **`xy_dataset.py`**: Script for handling the XY dataset, which is used to train models that map input images (X) to steering commands (Y).


## Getting Started


### Requirements


#### Hardware
- NVIDIA JetRacer
- CSI Camera
- MicroSD Card (with JetPack installed)
- Game controller (optional, for manual control)


#### Software
- JetPack SDK (includes CUDA, cuDNN, TensorRT, and other necessary components)
- Python 3.x
- Required Python libraries:
  - `jetcam`
  - `jetracer`
  - `torch`
  - `torchvision`
  - `opencv-python`
  - `numpy`


### Installation


1. **Install Dependencies**: Ensure all the required Python libraries are installed on the Jetson Nano:


   ```bash
   pip install jetcam jetracer torch torchvision opencv-python numpy
   ```


2. **Clone the Repository**:


   ```bash
   git clone https://github.com/NVIDIA-AI-IOT/jetracer.git
   cd jetracer
   ```


3. **Connect the Hardware**:
   - Attach the CSI camera to the JetRacer.
   - Ensure the JetRacer is powered and connected to your local network.


4. **Run Notebooks**: Use Jupyter Notebook to open and run any of the provided notebooks to start exploring the capabilities of the JetRacer.


## Usage


- Start with `basic_motion.ipynb` to get familiar with controlling the JetRacer manually.
- Use `road_following.ipynb` to train or run the autonomous road-following model.
- Fine-tune the camera settings or road-following parameters using the `reset_camera_service.ipynb` and `road_following_live.ipynb` notebooks.
- Optimize your models for better performance using `optimize_model.ipynb`.


## Customization


- **Training Your Model**: You can collect your own dataset using the `xy_dataset.py` script and train a custom road-following model.
- **Model Optimization**: Use the `optimize_model.ipynb` notebook to convert models to TensorRT, making them more efficient for real-time inference on Jetson Nano.
- **Manual Overrides**: Optionally connect a game controller to manually control the JetRacer when needed.

## Result

## Troubleshooting


- **Car Not Moving as Expected**: Ensure all connections are secure, and the correct model is loaded.
- **Camera Issues**: Use `reset_camera_service.ipynb` to adjust camera parameters live and reset the camera service if needed.
- **Model Performance**: If the road-following performance is poor, consider retraining the model with more data or optimizing it using TensorRT.
- **memory allocation in static TLS block**: Sometimes, when executing code in Jupyter, the following error may occur:
OSError: /usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block
So far, I have not found any reliable solution. Even rebooting Jetson may not always solve it. Should it happen, the easies thing to do is to ssh to Jetson, run Python3 from notebook's working directory, and execute all commands there.
