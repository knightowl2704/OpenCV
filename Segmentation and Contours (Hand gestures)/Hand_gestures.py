import cv2
import numpy as np


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
    ret, thresh2 = cv2.threshold(gray, 189, 225, cv2.ADAPTIVE_THRESH_MEAN_C)
    roi = thresh2[0:300, 50:250]

    cv2.imshow("roi",roi)
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area=0
    ci = 0
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


    cv2.imshow("thresh2", thresh2)

    hull = cv2.convexHull(cnt)
    moments = cv2.moments(cnt)
    # HULL

    if moments['m00'] != 0 :
        Cx = int(moments["m10"]/moments['m00'])
        Cy = int(moments["m01"]/moments['m00'])

    centr = (Cx,Cy)
    cv2.circle(drawing,centr,5,[0,0,255],2)


    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 1)
    #
    cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    hull = cv2.convexHull(cnt, returnPoints=False)

    defects = cv2.convexityDefects(cnt, hull)
    mind = 0
    maxd = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        dist = cv2.pointPolygonTest(cnt, centr, True)
        cv2.line(roi, start, end, [0, 255, 0], 2)

        cv2.circle(roi, far, 5, [0, 0, 255], -1)
    print(i)
    i = 0

    cv2.imshow("drawing",drawing)

    k = cv2.waitKey(10)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
