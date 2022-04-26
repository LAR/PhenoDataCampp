import numpy as np
import cv2

codec = 0    #for setting a particular output format
fps=20       #frames per second
scale = 0.25 #shrink the frame size. 

#The input file
cap = cv2.VideoCapture('walking.mp4')

#Getting the width and height of each frame. 
#And adjusting for the scale.
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
width = int(float(width)*scale)
height = int(float(height)*scale)

#This is the name of the output video
out = cv2.VideoWriter('outputFULLVID.avi', codec, fps, (width,height))

frames = []
# The next line gets the number of frames in the video.
# you can replace this with a smaller number if it takes too long
# e.g. numFrames = 50
numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


for f in range(0,numFrames):
    ret, frame = cap.read()
    smaller_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale) 
    frames.append(smaller_frame)
    
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8) 
    print("Writing frame ", f)
    out.write(medianFrame)
    
    
out.release()

