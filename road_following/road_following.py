import sys
# print(sys.path)
sys.path.append('jetracer/jetracer')
import torch
from torch2trt import TRTModule
from nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera
from utils import preprocess
import numpy as np

device = torch.device('cuda') 
model_trt = TRTModule().to(device)
model_trt.load_state_dict(torch.load('jetracer/notebooks/road_following/road_following_model_new_trt.pth', map_location=device))
try:
    car = NvidiaRacecar()
    camera = CSICamera(width=224, height=224, capture_fps=65)

#     STEERING_GAIN = 1
#     STEERING_BIAS = 0.00

#     car.throttle = 0.2
    car.steering_gain = 1.0
    car.steering_offset = 0.0
    car.steering = 0.0
    car.throttle = 0.0
    cou = False

    while True:
        image = camera.read()
        image = preprocess(image).half()
        output = model_trt(image).detach().cpu().numpy().flatten()
        x = float(output[0])
        x = max(-1, min(1, x))
        
        if x> 0.8:
            car.throttle = 0.1
            time.sleep(0.5)
            cou = True
        if x<0.8:
            

#         if x > 1:
#             x = 1
#         if x < -1:
#             x = -1
            
#         if abs(x) <0.2:
#             car.throttle = 0.25
#         else:
        car.throttle = 0.3

        car.steering = -x
        print(-x)
    
    
except KeyboardInterrupt:
    car.throttle = 0.0
    car.steering = 0.0
    raise SystemExit
    
    
    
    
    
    
    