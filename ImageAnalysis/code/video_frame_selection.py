

import cv2

video = cv2.VideoCapture('../data/sample.mp4')

fps = video.get(cv2.CAP_PROP_FPS)

print('frames per second =',fps)

minutes = 0
seconds = 28

frame_id = int(fps*(minutes*60 + seconds))
print('frame id =',frame_id)

video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
ret, frame = video.read()

"""
# alternative method
t_msec = 1000*(minutes*60 + seconds)

video.set(cv2.CAP_PROP_POS_MSEC, t_msec)
ret, frame = video.read()
"""


# Display and save frame
cv2.imshow('frame', frame); cv2.waitKey(0)

cv2.imwrite('screenshot.png',frame)

