import os
import torch
# import cv2  
import numpy as np 
from torchvision.models import resnet18
from torch2trt import TRTModule
import torchvision.transforms as transforms
import PIL.Image
import torch.nn.functional as F
from jetcam.csi_camera import CSICamera
from nvidia_racecar import NvidiaRacecar
from pupil_apriltags import Detector
from utils import preprocess
import time

# Set the environment variables for GPU usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Device configuration
device = torch.device('cuda')

# Load road following model
model_trt = TRTModule().to(device)
model_trt.load_state_dict(torch.load('jetracer/notebooks/road_following/road_following_model_new_trt.pth', map_location=device))
 
# Load collision avoidance model
model_collision = TRTModule().to(device)
model_collision.load_state_dict(torch.load('jetracer/notebooks/Collission Avoidance/best_model_resnet18_trt.pth', map_location=device))
 
# Initialize the AprilTag detector
at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# Initialize Nvidia Racecar and camera
car = NvidiaRacecar()
camera = CSICamera(width=224, height=224, capture_fps=65)

# Set car control parameters
car.steering_gain = 1
car.steering_offset = 0.0
car.throttle = 0.0  # Start with throttle set to zero
car.steering = 0.0  # Start with steering set to neutral


prob_blocked = 0
COLLISION_THRESHOLD = 0.92
AVOIDANCE_DURATION = 2.0  # Time to turn left (in seconds)
RETURN_DURATION = 1.5  # Time to turn right (in seconds)
STRAIGHT_DURATION = 1.0  # Time to go straight before resuming normal operation

# Possible states
NORMAL = 0
AVOIDING = 1
RETURNING = 2
STRAIGHTENING = 3

state = NORMAL
action_start_time = 0
    
try:
    while True:
        # Capture image from camera
        image = camera.read()
        
        # Preprocess image for road following
        image_preprocessed = preprocess(image)
        
        #april tags
        # Convert the image to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        tags = at_detector.detect(image, estimate_tag_pose = True, camera_params = [400.2333557174174,  533.2837800786184, 331.4549671721432, 263.3567321479512], tag_size = 0.065)
        tag_57_detected = False
        tag_26_detected = False
        tag_57 = None
        tag_26 = None
        
        for tag in tags:
            if tag.tag_id == 57:
                tag_57_detected = True
                tag_57 = np.linalg.norm(np.array(tag.pose_t))
                break
            if tag.tag_id == 26:
                tag_26_detected = True
                tag_26 = np.linalg.norm(np.array(tag.pose_t))
                break
        
        # collision loop
        output = model_trt(image_preprocessed)
        y = F.softmax(output, dim=1)
        prob_blocked = float(y.flatten()[0])
        
         # road_following_loop(image)
        output = model_trt(image_preprocessed).detach().cpu().numpy().flatten()
        x = float(output[0])
        x = max(-1, min(1, x))

#         print(f"Collision probability: {prob_blocked:.4f}, State: {state}")

        current_time = time.time()

        if state == NORMAL:
            if prob_blocked > COLLISION_THRESHOLD:
                print(f"Obstacle detected! Probability: {prob_blocked:.4f}")
                car.steering = 0.9  # Turn left
                state = AVOIDING
                action_start_time = current_time
            else:
                car.steering = 0.0  # Go straight

        elif state == AVOIDING:
            if current_time - action_start_time >= AVOIDANCE_DURATION:
                print("Avoidance complete, returning to track")
                car.steering = -0.9  # Turn right
                state = RETURNING
                time.sleep(0.05)
                action_start_time = current_time

        elif state == RETURNING:
            if current_time - action_start_time >= RETURN_DURATION:
                print("Return complete, straightening")
                car.steering = 0.0  # Go straight
                state = STRAIGHTENING
                action_start_time = current_time

        elif state == STRAIGHTENING:
            if current_time - action_start_time >= STRAIGHT_DURATION:
                print("Resuming normal operation")
                state = NORMAL
        elif tag_57_detected and tag_57 < 2:
            print("Tag 57 detected")
            car.steering = 1
            car.steering_gain = 1
            time.sleep(0.1)
        elif tag_26_detected:
            print("Tag 26 detected")
            car.throttle = 0
        else:
            car.throttle = 0.1
            car.steering = -x

        
        tag_list = []
       
        
except KeyboardInterrupt:
    # Safely stop the car when interrupted
    car.throttle = 0.0
    car.steering = 0.0
    print("Program interrupted, stopping the car.")

    
    
    
 


