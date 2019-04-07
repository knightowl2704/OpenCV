import cv2
import numpy as np

windowname = "Live webcam video capture"
cap = cv2.VideoCapture(0)
cv2.namedWindow(windowname)
if cap.isOpened():
    ret,frame = cap.read()
else:
    ret = False

while ret:
    ret,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #blue
    low = np.array([100,50,50])
    high  = np.array([140,255,255])


    imagemask = cv2.inRange(hsv,low,high)
    output  = cv2.bitwise_and(frame,frame, mask = imagemask)
    cv2.imshow("Imagemask",imagemask)
    cv2.imshow("frame",frame)
    cv2.imshow("Output", output )



    if cv2.waitKey(1) ==27 :
        break

cv2.destroyAllWindows()
videofileoutput.release()
cap.release()
