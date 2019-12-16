import cv2

cap = cv2.VideoCapture(0)

frame_count = -1
while True:
  frame_count += 1
  ret, frame = cap.read()

  cv2.imwrite('frames/frame' + str(frame_count) + '.jpg', frame)
  cv2.imshow('frame', frame)
  if cv2.waitKey(30) == ord('q'):
    break

cv2.destroyAllWindows()