import cv2
import numpy as np
from bokeh.plotting import figure, output_file, show

global mouseX
global mouseY
global clicked
clicked = False


def onmouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global mouseX
        mouseX = x
        global mouseY
        mouseY = y
        global clicked
        clicked = True
        print(x, y)


def getTransform(windowName, image):
    # Click on points to follow
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(windowName, onmouse)
    cv2.imshow(windowName, image)
    cv2.waitKey(200)

    global clicked
    global mouseX
    global mouseY

    counter = 0
    numberOfPoints = 4
    pts_src = np.empty(shape=[0, 2], dtype=np.int32)
    while counter < numberOfPoints:
        cv2.waitKey(100)
        if clicked:
            pts_src = np.append(pts_src, [[mouseX, mouseY]], axis=0)
            clicked = False
            counter += 1

    edge = 100
    shape = image.shape
    x = shape[1]
    y = shape[0]
    pts_dst = np.array(
        [[0 + edge, 0 + edge], [x - edge, 0 + edge], [x - edge, y - edge], [0 + edge, y - edge]])

    h, status = cv2.findHomography(pts_src, pts_dst)

    size = (x, y)

    return h, size


# Plot graph of grayscale value under marked points
if False:
    cap = cv2.VideoCapture('D:/Dropbox/SDU/8_Semester/RMURV2/Videos/brakeCarVid.mp4')

    backgroundVals = []
    foregroundVals = []

    backgroundPoints = []
    foregroundPoints = []

    backgroundPoints.append([404, 49])
    backgroundPoints.append([391, 256])
    backgroundPoints.append([302, 611])
    backgroundPoints.append([83, 557])
    backgroundPoints.append([26, 98])

    foregroundPoints.append([119, 154])
    foregroundPoints.append([251, 405])
    foregroundPoints.append([125, 271])
    foregroundPoints.append([68, 326])
    foregroundPoints.append([202, 534])

    # for item in data:
    #   mylist.append(item)

    backgroundMat = []
    foregroundMat = []

    t = np.linspace(0, len(backgroundPoints) - 1, len(backgroundPoints))

    for i in t:
        backgroundMat.append([])
        foregroundMat.append([])

    numberOfFrames = 0
    while cap.isOpened():
        numberOfFrames += 1

        ret, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        counter = 0
        for x, y in backgroundPoints:
            backgroundMat[int(t[counter])].append(frame[x][y])
            counter += 1

        counter = 0
        for x, y in foregroundPoints:
            foregroundMat[int(t[counter])].append(frame[x][y])
            counter += 1

        for i in backgroundPoints:
            cv2.circle(frame, (i[1], i[0]), 2, (255, 0, 0), 3, 8, 0)

        for i in foregroundPoints:
            cv2.circle(frame, (i[1], i[0]), 2, (0, 255, 0), 3, 8, 0)

        cv2.namedWindow('Image')

        cv2.setMouseCallback('Image', onmouse)

        cv2.imshow('Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    plot = figure(plot_width=1000, plot_height=1000, title="Back- and foreground values for brakeCarVid")
    plot.grid.grid_line_alpha = 0.3
    plot.xaxis.axis_label = 'Frame'
    plot.yaxis.axis_label = 'Grayscale value'

    t = np.linspace(0, numberOfFrames - 1, numberOfFrames)

    for i in backgroundMat:
        plot.line(t, i, color='red', legend='Background')

    lineColor = [0, 0, 255]
    for i in foregroundMat:
        plot.line(t, i, color=(lineColor[0], lineColor[1], lineColor[2]), legend='Foreground')
        lineColor[1] += 50
        lineColor[2] -= 50
        lineColor[0] += 50

    plot.legend.location = "top_left"

    show(plot)

# Subtract background. Good vals: KernelSize:  3 .   Blur:  5 .   It0:  2 .   It1:  13 .   It2:  0
if False:
    cap = cv2.VideoCapture('D:/Dropbox/SDU/8_Semester/RMURV2/Videos/video3.mp4')
    fgbg = cv2.createBackgroundSubtractorMOG2(5, 16, False)
    blur = 1
    kernelSize = 1
    it0 = 0
    it1 = 0
    it2 = 0
    while True:

        ret, frame = cap.read()

        #cv2.imshow('frame', frame)
        #cv2.waitKey(50)
        #cv2.waitKey()

        frame = cv2.blur(frame, (blur, blur))

        fgmask = fgbg.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
        fgmask = cv2.erode(fgmask, kernel, iterations=it0)
        fgmask = cv2.dilate(fgmask, kernel, iterations=it1)
        fgmask = cv2.erode(fgmask, kernel, iterations=it2)

        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

        fgmask = cv2.addWeighted(frame, 0.5, fgmask, 0.5, 1.0)

        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(100) & 0xff
        changed = False
        if k == 27:
            break
        if k != 255:
            changed = True
        if k == 112:
            blur = blur + 2
        if k == 111 and blur >= 3:
            blur = blur - 2
        if k == 230:
            kernelSize = kernelSize + 2
        if k == 108 and kernelSize >= 3:
            kernelSize = kernelSize - 2
        if k == 44:
            it0 = it0 + 1
        if k == 109 and it0 > 0:
            it0 = it0 - 1
        if k == 105:
            it1 = it1 + 1
        if k == 117 and it1 > 0:
            it1 = it1 - 1
        if k == 107:
            it2 = it2 + 1
        if k == 106 and it2 > 0:
            it2 = it2 - 1

        if changed:
            print(k)
            print('KernelSize: ', kernelSize, '.   Blur: ', blur, '.   It0: ', it0, '.   It1: ', it1, '.   It2: ', it2)
        #cv2.waitKey(100)
        #cv2.waitKey()

    cap.release()
    cv2.destroyAllWindows()

# Tracking using meanshift. Click on point to track
if False:
    cap = cv2.VideoCapture('D:/Dropbox/SDU/8_Semester/RMURV2/Videos/brakeCarVid.mp4')
    # take first frame of the video
    cap.set(1, 50)
    ret, frame = cap.read()
    # setup initial location of window
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', onmouse)
    cv2.imshow('Image', frame)
    cv2.waitKey(100)

    while not clicked:
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    r, h, c, w = mouseY-25, 50, mouseX-25, 50  # simply hardcoded the values
    track_window = (c, r, w, h)
    # set up the ROI for tracking
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while (1):
        ret, frame = cap.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            # Draw it on image
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv2.imshow('img2', img2)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()

# Tracking using camshift. Click on point to track
if False:
    cap = cv2.VideoCapture('brakeCarVid.mp4')
    # take first frame of the video
    cap.set(1, 50)
    ret, frame = cap.read()
    # setup initial location of window
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', onmouse)
    cv2.imshow('Image', frame)
    cv2.waitKey(100)

    while not clicked:
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    size = 30
    r, h, c, w = mouseY-int(size/2), size, mouseX-int(size/2), size  # simply hardcoded the values
    track_window = (c, r, w, h)
    # set up the ROI for tracking
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))


    #cv2.namedWindow('Test')
    #cv2.imshow('Test', mask)
    #cv2.waitKey(10000)

    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while (1):
        ret, frame = cap.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.imshow('img2', img2)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()

# Track by clicking
if False:
    cap = cv2.VideoCapture('D:/Dropbox/SDU/8_Semester/RMURV2/Videos/video3.mp4')

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Click on points to follow
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', onmouse)
    cv2.imshow('frame', old_frame)
    cv2.waitKey(200)

    p0 = np.array([[[0, 0]]], dtype=np.float32)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while (1):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Add point if clicked
        if clicked:
            pMouse = np.append(p0, [[[mouseX, mouseY]]], axis=0)
            p0 = np.array(pMouse, dtype=np.float32)
            clicked = False
            print(p0)
            print("Point added!\n")

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv2.destroyAllWindows()
    cap.release()

# Click on four points, get transformation, track by clicking
if False:
    cap = cv2.VideoCapture('D:/Dropbox/SDU/8_Semester/RMURV2/Videos/video3.mp4')

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # TEST IMAGE, REMOVE LATER
    #old_frame = cv2.imread('bb.jpg')

    # Click on points to follow
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('frame', onmouse)
    cv2.imshow('frame', old_frame)
    cv2.waitKey(200)

    counter = 0
    numberOfPoints = 4
    pts_src = np.empty(shape=[0, 2], dtype=np.int32)
    while counter < numberOfPoints:
        cv2.waitKey(100)
        if clicked:
            pts_src = np.append(pts_src, [[mouseX, mouseY]], axis=0)
            clicked = False
            counter += 1

    #pts_dst = np.array([[0, 0], [0, 500], [500, 500], [500, 0]])
    x_orig = pts_src[0][0]
    y_orig = pts_src[0][1]

    #pts_dst = np.array([[0, 0], [pts_src[1][0] - x_orig, pts_src[1][1] - y_orig], [pts_src[2][0] - x_orig, pts_src[2][1] - y_orig], [pts_src[3][0] - x_orig, pts_src[3][1] - y_orig]])
    edge = 100
    shape = old_frame.shape
    x = shape[1]
    y = shape[0]
    pts_dst = np.array(
        [[0 + edge, 0 + edge], [x - edge, 0 + edge], [x - edge, y - edge], [0 + edge, y - edge]])

    pMouse = np.empty(shape=[0, 1, 2], dtype=np.float32)

    h, status = cv2.findHomography(pts_src, pts_dst)

    size = (x, y)

    old_frame = cv2.warpPerspective(old_frame, h, size)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', old_gray)
    clicked = False
    while not clicked:
        cv2.waitKey(100)

    p0 = np.array([[[mouseX, mouseY]]], dtype=np.float32)
    clicked = False
    print(p0)
    print("Point added!\n")


    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while (1):
        ret, frame = cap.read()
        frame = cv2.warpPerspective(frame, h, size)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Add point if clicked
        if clicked:
            pMouse = np.append(p0, [[[mouseX, mouseY]]], axis=0)
            p0 = np.array(pMouse, dtype=np.float32)
            clicked = False
            print(p0)
            print("Point added!\n")

        # calculate optical flow

        print(p0, "\n\n\n")
        if len(p0) == 0:
            cv2.waitKey(1000)
            continue

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv2.destroyAllWindows()
    cap.release()

# Subtract background on perspective
# KernelSize:  3 .   Blur:  3 .   It0:  2 .   It1:  43 .   It2:  18.,
if True:
    cap = cv2.VideoCapture('D:/Dropbox/SDU/8_Semester/RMURV2/Videos/video3.mp4')

    ret, image = cap.read()
    h, size = getTransform('frame', image)

    blur = 3
    kernelSize = 3
    it0 = 2
    it1 = 40
    it2 = 18

    fgbg = cv2.createBackgroundSubtractorMOG2(50, 16, False)
    while True:
        ret, frame = cap.read()

        frame = cv2.blur(frame, (blur, blur))

        frame = cv2.warpPerspective(frame, h, size)

        fgmask = fgbg.apply(frame)

        #cv2.imshow('frame', fgmask)
        #cv2.waitKey(50)
        #cv2.waitKey()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
        fgmask = cv2.erode(fgmask, kernel, iterations=it0)
        fgmask = cv2.dilate(fgmask, kernel, iterations=it1)
        fgmask = cv2.erode(fgmask, kernel, iterations=it2)

        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

        fgmask = cv2.addWeighted(frame, 0.5, fgmask, 0.5, 0.0)

        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(10) & 0xff
        changed = False
        if k == 27:
            break
        if k != 255:
            changed = True
        if k == 112:
            blur = blur + 2
        if k == 111 and blur >= 3:
            blur = blur - 2
        if k == 230:
            kernelSize = kernelSize + 2
        if k == 108 and kernelSize >= 3:
            kernelSize = kernelSize - 2
        if k == 44:
            it0 = it0 + 1
        if k == 109 and it0 > 0:
            it0 = it0 - 1
        if k == 105:
            it1 = it1 + 1
        if k == 117 and it1 > 0:
            it1 = it1 - 1
        if k == 107:
            it2 = it2 + 1
        if k == 106 and it2 > 0:
            it2 = it2 - 1

        if changed:
            print('KernelSize: ', kernelSize, '.   Blur: ', blur, '.   It0: ', it0, '.   It1: ', it1, '.   It2: ', it2)

    cap.release()
    cv2.destroyAllWindows()