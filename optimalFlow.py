import numpy as np
import cv2

cap = cv2.VideoCapture('intersection.mp4')
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
color =(150,100,3)
mask = np.zeros_like(old_frame)
cap.release()
_number_of_frames = 0

def opticalFlowTracking(pointArray2Track):
    p0 = pointArray2Track
    ret,frame=cap.read()
    frame_grey =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,frame_grey,p0,None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # Drawing the points
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow(img,"frame")
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    cv2.destroyAllWindows()
    cap.release()

