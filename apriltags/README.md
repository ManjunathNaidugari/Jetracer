# AprilTag Detection for Intersection Navigation using JetRacer


## Overview


This project demonstrates the use of NVIDIA's JetRacer and a CSI camera for detecting AprilTags and executing commands based on the detected tags. The AprilTag detection is performed using the `pupil_apriltags` library. The system identifies specific AprilTags and triggers commands when the tags are within a certain distance from the camera.


## Project Structure


- `apriltag.ipynb`: The main notebook containing the code to initialize the JetRacer, capture video from the camera, detect AprilTags, and execute commands based on the detected tags.


## Requirements


### Hardware
- NVIDIA JetRacer
- CSI Camera


### Software
- JetPack SDK (includes CUDA, cuDNN, TensorRT, and other necessary components)
- Python 3.x
- Required Python libraries:
  - `jetcam`
  - `jetracer`
  - `pupil_apriltags`
  - `opencv-python`
  - `numpy`


## Setup Instructions


1. **Install Dependencies**: Ensure all the required Python libraries are installed. You can install them using pip:


   ```bash
   pip install jetcam jetracer pupil_apriltags opencv-python numpy
   ```


2. **Connect the Hardware**:
   - Attach the CSI camera to the NVIDIA JetRacer.
   - Ensure the JetRacer is powered and connected.


3. **Run the Notebook**:
   - Open the `apriltag.ipynb` notebook on your JetRacer.
   - Run all the cells to start the AprilTag detection process.


## How It Works


1. **Initialization**:
   - The JetRacer (`NvidiaRacecar`) and CSI Camera (`CSICamera`) are initialized.
   - An AprilTag detector (`Detector`) is configured to detect tags belonging to the "tag36h11" family.


2. **AprilTag Detection**:
   - The camera continuously captures frames.
   - Each frame is converted to grayscale for AprilTag detection.
   - Detected tags are analyzed to check for specific IDs (e.g., 57 and 26).


3. **Distance Calculation**:
   - The pose (position) of each detected tag is estimated.
   - The Euclidean distance to the tag is calculated using the translation vector `pose_t`.


4. **Command Execution**:
   - If Tag 57 is detected and is within a distance of 2 meters, the car's steering is adjusted.
   - If Tag 26 is detected, the car stops moving.
   - If no specific tag is detected, the car continues with its default behavior.


## Customization


- **Tag IDs**: You can change the tag IDs (57 and 26) in the code to detect different tags.
- **Tag Distance**: Adjust the distance threshold (currently set to 2 meters) to change when commands are triggered.
- **Commands**: Modify the steering, throttle, and other commands executed when specific tags are detected.


## Conclusion


This project demonstrates a basic implementation of autonomous behavior using AprilTag detection with NVIDIA's JetRacer. By detecting tags and responding based on their distance, the JetRacer can perform specific actions such as steering or stopping, showcasing its potential for more complex autonomous tasks.


Feel free to explore and modify the project to suit your needs. Happy experimenting!
