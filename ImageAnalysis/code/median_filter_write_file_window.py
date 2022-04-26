import numpy as np
import cv2

codec = 0
fps=20
scale = 0.25
cap = cv2.VideoCapture('walking.mp4')
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
width = int(float(width)*scale)
height = int(float(height)*scale)



out = cv2.VideoWriter('outputWINDOW.avi', codec, fps, (width,height))

frames = []


numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

windowSize = 25
for f in range(0,numFrames):
    ret, frame = cap.read()
    smaller_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale) 
    frames.append(smaller_frame)
    if f>windowSize:
        medianFrame = np.median(frames[-windowSize:], axis=0).astype(dtype=np.uint8) 
        print("Writing frame ", f)
        out.write(medianFrame)
    
    
out.release()

