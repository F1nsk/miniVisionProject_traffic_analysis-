import cv2

import numpy as np
from threading import Thread
import sys


class VideoTracker:
    """Class with different functions for tracking objects in videos"""

    def __init__(self, video_path):
        self.videoPath = video_path
        self.video = cv2.VideoCapture(video_path)

        self.clicked = False
        self.blur = 3
        self.kernelSize = 3
        self.it0 = 3
        self.it1 = 25
        self.it2 = 0
        self.coord2Track


        self.newFrameReady = False
        self.currentFrame = 0

        self.tracker_array = []

    # Function that is called whenever the mouse is clicked. Remember to set callback.
    def on_mouse(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouseX = x
            self.mouseY = y
            self.clicked = True

    # Click on four points in a square to make a transformation.
    def click_to_make_transformation(self, window_name, frame_number=0):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, image = self.video.read()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.on_mouse)
        cv2.imshow(window_name, image)
        cv2.waitKey(200)

        counter = 0
        number_of_points = 4
        pts_src = np.empty(shape=[0, 2], dtype=np.int32)
        while counter < number_of_points:
            cv2.waitKey(100)
            if self.clicked:
                pts_src = np.append(pts_src, [[self.mouseX, self.mouseY]], axis=0)
                self.clicked = False
                counter += 1

        edge = 100
        shape = image.shape
        x = shape[1]
        y = shape[0]
        pts_dst = np.array(
            [[0 + edge, 0 + edge], [x - edge, 0 + edge], [x - edge, y - edge], [0 + edge, y - edge]])

        self.h, status = cv2.findHomography(pts_src, pts_dst)

        self.size = (x, y)

    def nothing(x, y):
        pass

    # Manipulate the parameters using trackbars
    def trackbars(self, window_name):
        cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Trackbars', 600, 270)

        cv2.createTrackbar('Blur', window_name, 1, 25, self.nothing)
        cv2.createTrackbar('KernelSize', window_name, 1, 25, self.nothing)
        cv2.createTrackbar('it0', window_name, 0, 25, self.nothing)
        cv2.createTrackbar('it1', window_name, 0, 25, self.nothing)
        cv2.createTrackbar('it2', window_name, 0, 25, self.nothing)
        cv2.createTrackbar('BG_frames', window_name, 0, 100, self.nothing)
        cv2.createTrackbar('BG_thresh', window_name, 0, 100, self.nothing)

        old_background_frames = 0
        old_background_thresh = 0

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                return

            self.blur = cv2.getTrackbarPos('Blur', window_name)
            if self.blur % 2 == 0:
                self.blur = self.blur + 1

            self.kernelSize = cv2.getTrackbarPos('KernelSize', window_name)
            if self.kernelSize % 2 == 0:
                self.kernelSize = self.kernelSize + 1

            self.it0 = cv2.getTrackbarPos('it0', window_name)
            self.it1 = cv2.getTrackbarPos('it1', window_name)
            self.it2 = cv2.getTrackbarPos('it2', window_name)

            background_frames = cv2.getTrackbarPos('BG_frames', window_name)
            background_thresh = cv2.getTrackbarPos('BG_thresh', window_name)

            if background_frames < 1:
                background_frames = 1

            if background_thresh < 1:
                background_thresh = 1

            frames_changed = background_frames != old_background_frames
            tresh_changed = background_thresh != old_background_thresh
            if frames_changed or tresh_changed:
                self.create_background_subtractor(background_frames, background_thresh)

            old_background_thresh = background_thresh
            old_background_frames = background_frames

    # Start thread with trackbars
    def display_trackbars(self, window_name):
        thread = Thread(target=self.trackbars, args=(window_name,))
        thread.start()

    def create_background_subtractor(self, frames, threshold):
        self.backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(frames, threshold, True)

    # Maybe transform after background sub?
    def subtract_background(self, frame, use_transformation=False):
        use_frame = cv2.blur(frame, (self.blur, self.blur))

        if use_transformation:
            use_frame = cv2.warpPerspective(use_frame, self.h, self.size)

        fgmask = self.backgroundSubtractor.apply(use_frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernelSize, self.kernelSize))
        fgmask = cv2.erode(fgmask, kernel, iterations=self.it0)
        fgmask = cv2.dilate(fgmask, kernel, iterations=self.it1)
        fgmask = cv2.erode(fgmask, kernel, iterations=self.it2)

        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        return use_frame, fgmask

    # Subtract the background.
    def run_background_subtractor(self, window_name, use_transformation=False):
        while True:
            ret, frame = self.video.read()
            frame, background = self.subtract_background(frame, use_transformation)

            display_image = cv2.addWeighted(frame, 0.5, background, 0.5, 0.0)

            cv2.imshow(window_name, display_image)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

    # Subtract the background, then detect and mark blobs.
    def run_background_subtractor_with_blob_detection(self, window_name, use_transformation=False):
        has_completed_one_cycle = False
        while True:
            frame, background = self.subtract_background(use_transformation)

            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

            if has_completed_one_cycle:
                self.draw_blob_detection(window_name, background, frame, 4)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

            has_completed_one_cycle = True

    # Convert array to a usable array for optical flow
    def to_array_for_optical_flow(self, centroids):
        p0 = np.array([[[0, 0]]], dtype=np.float32)
        for i, val in enumerate(centroids):
            if i == 0:
                continue
            pMouse = np.append(p0, [[[centroids[i][0], centroids[i][1]]]], axis=0)
            p0 = np.array(pMouse, dtype=np.float32)

        return p0

    # Get info from blob detection
    def blob_detection(self, detect_image, connectivity=4):
        ret, thresh = cv2.threshold(detect_image, 0, 255, cv2.THRESH_BINARY)

        # Perform the operation
        output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]

        return num_labels, labels, stats, centroids

    # Use blob detection on an image and display the found blobs on another image.
    def draw_blob_detection(self, window_name, detect_image, display_image, connectivity=4):

        num_labels, labels, stats, centroids = self.blob_dection(detect_image, connectivity)

        # Draw circles where blobs where found
        for i, val in enumerate(centroids):
            if i == 0:
                continue
            cv2.circle(display_image, (int(centroids[i][0]), int(centroids[i][1])), 10, (255, 0, 0), 3, 8, 0)

        # Show the result
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display_image)
        cv2.waitKey(1)

        new_window_name = window_name + ": " + str(num_labels)
        cv2.setWindowTitle(window_name, new_window_name)


    # Display both background subtraction and blobs detected
    def run_background_and_blobs(self, window_name_background, window_name_blobs, use_transformation=False):
        has_completed_one_cycle = False
        while True:
            ret, frame = self.video.read()
            frame, background = self.subtract_background(frame, use_transformation)

            display_image = cv2.addWeighted(frame, 0.5, background, 0.5, 0.0)

            cv2.imshow(window_name_background, display_image)

            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

            if has_completed_one_cycle:
                self.draw_blob_detection(window_name_blobs, background, frame, 8)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

            has_completed_one_cycle = True

#<<<<<<< HEAD
    # Create and return a tracker
    def create_tracker(self, tracker_nr, point, frame):
        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        tracker_type = tracker_types[tracker_nr]

        tracker = cv2.Tracker
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

        # Define an initial bounding box
        bbox = (point[0], point[1], 30, 30)

        ok = tracker.init(frame, bbox)

        return tracker

    def run_tracker(self, window_name, tracker_nr, use_transformation=False):
        thread = Thread(target=self.tracker, args=(window_name,))
        thread.start()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.on_mouse)

        has_completed_one_cycle = False
        while True:
            # if self.clicked:
            #     new_tracker = self.create_tracker(tracker_nr, (self.mouseX, self.mouseY), self.currentFrame)
            #     self.tracker_array.append(new_tracker)
            #     self.clicked = False

            ret, self.currentFrame = self.video.read()

            frame, background = self.subtract_background(self.currentFrame, use_transformation)

            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

            if has_completed_one_cycle:
                num_labels, labels, stats, centroids = self.blob_detection(background, 8)
                for point in centroids:
                    new_tracker = self.create_tracker(tracker_nr, point, self.currentFrame)
                    self.tracker_array.append(new_tracker)
                    break

            self.newFrameReady = True

            has_completed_one_cycle = True

            cv2.waitKey(1)

    def tracker(self, window_name):
        # Set up tracker.

        while True:
            while not self.newFrameReady:
                cv2.waitKey(1)

            # Start timer
            timer = cv2.getTickCount()

            good_trackers = []
            for tracker in self.tracker_array:
                # Update tracker
                ok, bbox = tracker.update(self.currentFrame)

                # Draw bounding box
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(self.currentFrame, p1, p2, (255, 0, 0), 2, 1)
                    good_trackers.append(tracker)

                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break

            self.tracker_array = good_trackers

            # Display result
            cv2.imshow(window_name, self.currentFrame)
            cv2.waitKey(1)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Display FPS on frame
            cv2.putText(self.currentFrame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
#=======

    def opticalFlowTracking(self, pointArray2Track):
        #### Optical flow defines ###
        lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        p0 = pointArray2Track
        while(True):
            ret, frame = self.video.read()
            ret, old_frame = self.video.read()
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            color = (150, 100, 3)
            mask = np.zeros_like(old_frame)


            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_grey, p0, None, **lk_params)
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # Drawing the points
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)
            cv2.imshow(img, "frame")
            old_gray = frame_grey.copy()
            p0 = good_new.reshape(-1, 1, 2)
            cv2.destroyAllWindows()
            self.video.release()
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

#>>>>>>> optimalFlow
