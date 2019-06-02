import cv2
import imutils as imutils
import numpy as np
import matplotlib.pyplot as plt

fgbg = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    # cv2.rectangle(img, (70, 100), (250, 350), (0, 255, 0), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("video" , img)
    cv2.imshow("gray" , gray)
    # ret,thresh = cv2.threshold(gray,150,255,cv2.THRESH_OTSU)
    # cv2.imshow("thresh" , thresh)
    # ret,thresh1 = cv2.threshold(gray,100,225,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # cv2.imshow("ADaptive",thresh1)
    ret, thresh2 = cv2.threshold(gray, 110, 225, cv2.ADAPTIVE_THRESH_MEAN_C)
    roi = thresh2[0:300, 50:250]

    cv2.imshow("roi",roi)
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area=0

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area > max_area):
            max_area = area
            ci = i
    cnt = contours[ci]
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

    # thresh2 = cv2.erode(thresh2, None, iterations=2)
    # thresh2 = cv2.dilate(thresh2, None, iterations=2)
    # # fgmask = fgbg.apply(thresh2)

    # cv2.imshow('frame', fgmask)
    cv2.imshow("thresh2", thresh2)
    hull = cv2.convexHull(cnt)

    cv2.imshow("drawing",drawing)

    k = cv2.waitKey(10)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()