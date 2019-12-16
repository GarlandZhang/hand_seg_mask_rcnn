import cv2
import numpy as np

import time

start_time = time.time()
backSub = cv2.createBackgroundSubtractorMOG2()
tracker = cv2.TrackerKCF_create()

# Defining HSV Threadholds
lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

for i in range(0, 900):
  frame = cv2.imread('img2.jpg')

  fgMask = backSub.apply(frame)
  ret, box = tracker.update(frame)

  # Single Channel mask,denoting presence of colours in the about threshold
  skinMask = cv2.inRange(frame, lower_threshold, upper_threshold)

  # Cleaning up mask using Gaussian Filter
  skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

  # Extracting skin from the threshold mask
  skin = cv2.bitwise_and(frame, frame, mask=skinMask)

end_time = time.time()

print('Time: {}'.format(end_time - start_time))
