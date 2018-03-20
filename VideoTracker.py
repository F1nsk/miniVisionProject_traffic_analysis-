import cv2
import math
import numpy as np
from numpy import matlib
from threading import Thread
from Kalman import Kalman
import sys


class VideoTracker:
    """Class with different functions for tracking objects in videos"""

    def __init__(self, video_path):
        self.videoPath = video_path
        self.video = cv2.VideoCapture(video_path)

        self.clicked = False
        self.blur = 9
        self.kernelSize = 3
        self.it0 = 1
        self.it1 = 5
        self.it2 = 0

        self.tracker_array = []
        self.trackedPoints = []

    # Function that is called whenever the mouse is clicked. Remember to set callback.
    def on_mouse(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouseX = x
            self.mouseY = y
            self.clicked = True
            print(x, y)

    def rotateImage(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

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

    def make_predefined_transformation(self, src, dst, size):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, image = self.video.read()

        self.h, status = cv2.findHomography(src, dst)
        self.size = size

        # im_out = cv2.warpPerspective(image, self.h, size)
        # im_out = self.rotateImage(im_out, -90)
        #
        # # Display images
        # cv2.imshow("Warped Source Image", im_out)
        # cv2.waitKey(200)
        # cv2.waitKey()

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
    def subtract_background(self, frame):
        use_frame = cv2.blur(frame, (self.blur, self.blur))

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
            frame, background = self.subtract_background(frame)

            if use_transformation:
                frame = cv2.warpPerspective(frame, self.h, self.size)
                background = cv2.warpPerspective(background, self.h, self.size)

            display_image = cv2.addWeighted(frame, 0.5, background, 0.5, 0.0)

            cv2.imshow(window_name, display_image)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

    # Convert array to a usable array for optical flow.
    def to_array_for_optical_flow(self, centroids):
        p0 = np.array([[[0, 0]]], dtype=np.float32)
        for i, val in enumerate(centroids):
            if i == 0:
                continue
            pMouse = np.append(p0, [[[centroids[i][0], centroids[i][1]]]], axis=0)
            p0 = np.array(pMouse, dtype=np.float32)

        return p0

    # Get info from blob detection
    def blob_detection(self, detect_image, connectivity=8):
        # Make image binary
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

        # First element is garbage
        num_labels = num_labels - 1
        labels = labels[1:]
        stats = stats[1:]
        centroids = centroids[1:]

        return num_labels, labels, stats, centroids

    # Use blob detection on an image and display the found blobs on another image.
    def draw_blob_detection(self, window_name, detect_image, display_image, connectivity=8, minimum_size=0):

        num_labels, labels, stats, centroids = self.blob_detection(detect_image, connectivity)

        # Draw circles where blobs where found

        for i, val in enumerate(centroids):
            if stats[i, cv2.CC_STAT_AREA] > minimum_size:
                cv2.circle(display_image, (int(centroids[i][0]), int(centroids[i][1])), 10, (255, 0, 0), 3, 8, 0)

        # Show the result
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display_image)
        cv2.waitKey(1)

        new_window_name = window_name + ": " + str(num_labels)
        cv2.setWindowTitle(window_name, new_window_name)

    # Display both background subtraction and blobs detected.
    def run_background_and_blobs(self, window_name_background, window_name_blobs, use_transformation=False):
        has_completed_one_cycle = False
        cv2.namedWindow(window_name_background, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = self.video.read()

            frame, background = self.subtract_background(frame)

            if use_transformation:
                frame = cv2.warpPerspective(frame, self.h, self.size)
                background = cv2.warpPerspective(background, self.h, self.size)

            display_image = cv2.addWeighted(frame, 0.5, background, 0.5, 0.0)

            cv2.imshow(window_name_background, display_image)

            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

            # README: Just added minimum size of blobs to draw_blob_dectection. Maybe it will help, idk.
            if has_completed_one_cycle:
                self.draw_blob_detection(window_name_blobs, background, frame, 8, 200)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

            has_completed_one_cycle = True

    # Create and return a tracker
    def create_tracker(self, tracker_nr, bbox, frame):
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
        # size = 50
        # bbox = (point[0] - size/2, point[1] - size/2, size, size)
        # self.trackedPoints.append((bbox[0], bbox[1]))

        ok = tracker.init(frame, bbox)
        self.trackedPoints.append((bbox[0], bbox[1]))

        return tracker

    # Calculate distance between two points.
    def distance_to_point(self, point1, point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    # Check if a point close to 'point' is already being tracked.
    def is_point_being_tracked(self, point, min_distance):
        for trackedPoint in self.trackedPoints:
            distance = self.distance_to_point(point, trackedPoint)
            if distance < min_distance:
                return True

        return False

    # Perform tracking based on the blobs detected.
    def run_tracker(self, window_name, tracker_nr, use_transformation=False):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.on_mouse)

        has_completed_one_cycle = False
        while True:
            ret, frame = self.video.read()
            orig_frame = frame.copy()

            throw_away_frame, background = self.subtract_background(frame)

            if use_transformation:
                frame = cv2.warpPerspective(frame, self.h, self.size)
                background = cv2.warpPerspective(background, self.h, self.size)

            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

            if has_completed_one_cycle:
                new_points_to_track = self.find_new_trackpoints(background)
                for bbox in new_points_to_track:
                    new_tracker = self.create_tracker(tracker_nr, bbox, frame)
                    self.tracker_array.append(new_tracker)

            has_completed_one_cycle = True

            # Start timer
            timer = cv2.getTickCount()

            # Take next step
            frame = self.increment_trackers(orig_frame)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (50, 170, 50), 2)

            # Display result
            cv2.imshow(window_name, frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    # Take next step with all the trackers.
    def increment_trackers(self, next_frame):
        good_trackers = []
        next_points_to_track = []
        cv2.namedWindow('Tracker', cv2.WINDOW_NORMAL)
        cv2.imshow('Tracker', next_frame)
        counter = 0
        for tracker in self.tracker_array:
            # Update tracker
            ok, bbox = tracker.update(next_frame)

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                # Check if bbox has moved
                if len(self.trackedPoints) > 0:
                    old_p1 = self.trackedPoints[counter]
                    if self.distance_to_point(p1, old_p1) < 0.001:
                        continue

                cv2.rectangle(next_frame, p1, p2, (255, 0, 0), 2, 1)

                next_points_to_track.append((bbox[0], bbox[1]))
                good_trackers.append(tracker)

            else:
                print('Bad tracker!')

            counter = counter + 1

        self.trackedPoints = next_points_to_track
        self.tracker_array = good_trackers
        return next_frame

    # Detect blobs and add new points to track.
    def find_new_trackpoints(self, background):
        num_labels, labels, stats, centroids = self.blob_detection(background, 8)

        new_points_to_track = []
        counter = 0

        for point in centroids:
            if not self.is_point_being_tracked(point, 50):
                if stats[counter, cv2.CC_STAT_AREA] > 10:
                    # cv2.circle(frame, (int(point[0]), int(point[1])), 10, (0, 0, 255), 3, 8, 0)

                    bbox = self.create_bbox_with_stats(stats, counter)
                    new_points_to_track.append(bbox)
            counter = counter + 1

        return new_points_to_track

    # Create a bbox based on stats from blob detection
    def create_bbox_with_stats(self, stats, number):
        left_x = stats[number, cv2.CC_STAT_LEFT]
        top_y = stats[number, cv2.CC_STAT_TOP]
        width = stats[number, cv2.CC_STAT_WIDTH]
        height = stats[number, cv2.CC_STAT_HEIGHT]

        bbox = (left_x - 10, top_y - 10, width + 20, height + 20)
        return bbox

    def run_optical_flow(self, window_name, use_transformation=False):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.on_mouse)

        # Create a mask image for drawing purposes
        ret, old_frame = self.video.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(old_frame)

        has_completed_one_cycle = False
        while True:
            # Start timer
            timer = cv2.getTickCount()
            
            p0 = np.array([[[0, 0]]], dtype=np.float32)

            ret, frame = self.video.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            throw_away_frame, background = self.subtract_background(frame)

            if use_transformation:
                frame = cv2.warpPerspective(frame, self.h, self.size)
                background = cv2.warpPerspective(background, self.h, self.size)

            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

            if has_completed_one_cycle:
                for point in self.trackedPoints:
                    temp_array = np.append(p0, [[[point[0], point[1]]]], axis=0)
                    p0 = np.array(temp_array, dtype=np.float32)

                new_points_to_track = self.find_new_trackpoints(background)
                for bbox in new_points_to_track:
                    temp_array = np.append(p0, [[[bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]]], axis=0)
                    p0 = np.array(temp_array, dtype=np.float32)

                self.trackedPoints.clear()

            has_completed_one_cycle = True

            # Perform optical flow
            old_gray = self.increment_optical_flow(old_gray, frame_gray, mask, frame, p0)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (50, 170, 50), 2)

            # Display result
            cv2.imshow(window_name, frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    def increment_optical_flow(self, old_frame, next_frame, mask, frame, p0):
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(5, 5),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, next_frame, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            if self.distance_to_point((a, b), (c, d)) < 0.001:
                continue

            #mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (255, 0, 0), -1)
            self.trackedPoints.append((a, b))

        # Now update the previous frame and previous points
        old_frame = next_frame.copy()

        # p0 = good_new.reshape(-1, 1, 2)

        return old_frame

    def run_optical_flow_with_kalman(self, window_name, use_transformation=False):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.on_mouse)

        # Init kalman filter params
        deltaT = 1.

        F = np.matrix([[1, 0, deltaT, 0], [0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.matrix([[deltaT ** 2 / 2, 0], [0, deltaT ** 2 / 2], [deltaT, 0], [0, deltaT]])
        H = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        model_noise = 1
        measurement_noise = 100

        R = measurement_noise ** 2 * np.matlib.identity(4) / deltaT
        Q = model_noise ** 2 * np.matlib.identity(4) * deltaT

        init_mu = np.matrix([[0., 0], [0., 0]])  # This is the acceleration estimate
        init_P = np.matlib.identity(4)

        kalman_array = []
        init_x = np.matrix([[0], [0], [0], [0]])
        new_kalman = Kalman(F, B, Q, H, R, init_x, init_mu, init_P, deltaT)
        kalman_array.append(new_kalman)
        # Init params done

        # Create a mask image for drawing purposes
        ret, old_frame = self.video.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        if use_transformation:
            old_gray = cv2.warpPerspective(old_gray, self.h, self.size)
            old_gray = self.rotateImage(old_gray, -90)

        mask = np.zeros_like(old_frame)

        has_completed_one_cycle = False
        counter = 0
        while True:
            # Start timer
            timer = cv2.getTickCount()

            counter = counter + 1

            p0 = np.array([[[0, 0]]], dtype=np.float32)

            ret, frame = self.video.read()

            throw_away_frame, background = self.subtract_background(frame)

            if use_transformation:
                frame = cv2.warpPerspective(frame, self.h, self.size)
                background = cv2.warpPerspective(background, self.h, self.size)
                frame = self.rotateImage(frame, -90)
                background = self.rotateImage(background, -90)

            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if has_completed_one_cycle:
                for point in self.trackedPoints:
                    temp_array = np.append(p0, [[[point[0], point[1]]]], axis=0)
                    p0 = np.array(temp_array, dtype=np.float32)

                new_points_to_track = self.find_new_trackpoints(background)
                for bbox in new_points_to_track:
                    point_x = bbox[0] + bbox[2] / 2
                    point_y = bbox[1] + bbox[3] / 2
                    temp_array = np.append(p0, [[[point_x, point_y]]], axis=0)
                    p0 = np.array(temp_array, dtype=np.float32)
                    init_x = np.matrix([[point_x], [point_y], [0], [0]])
                    new_kalman = Kalman(F, B, Q, H, R, init_x, init_mu, init_P, deltaT)
                    kalman_array.append(new_kalman)

                p0 = np.delete(p0, 0, axis=0)

                self.trackedPoints.clear()

            has_completed_one_cycle = True

            # Perform optical flow
            old_gray, kalman_array = self.increment_optical_flow_kalman(old_gray, frame_gray, mask, frame, p0, kalman_array)

            # Draw kalman
            self.draw_kalman_speed(frame, kalman_array, 2.4, 40)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (50, 170, 50), 2)

            # Display result
            cv2.imshow(window_name, frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    def increment_optical_flow_kalman(self, old_frame, next_frame, mask, frame, p0, kalman_array):
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(5, 5),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, next_frame, p0, None, **lk_params)
        # Select good points

        counter = 0
        points_removed = 0
        for point in p0:
            if st[counter] != 1:
                del kalman_array[counter - points_removed]
                points_removed = points_removed + 1
            counter = counter + 1

        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks

        points_removed = 0
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            moved_distance = self.distance_to_point((a, b), (c, d))
            if moved_distance < 0.001:
                del kalman_array[i - points_removed]
                points_removed = points_removed + 1
                continue

            #mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (255, 0, 0), -1)
            self.trackedPoints.append((a, b))

            if len(kalman_array) > 0:
                next_x = np.matrix([[a], [b], [moved_distance], [moved_distance]])
                kalman_array[i - points_removed].run_one_step(next_x)

        # Now update the previous frame and previous points
        old_frame = next_frame.copy()

        return old_frame, kalman_array

    def draw_kalman_speed(self, frame, kalman_array, seconds, min_iterations):
        for kalman in kalman_array:
            if kalman.get_number_of_iterations() > min_iterations:
                next_x = kalman.get_nextX()
                pos_x = int(round(next_x[0, 0]))
                pos_y = int(round(next_x[1, 0]))
                speed_x = next_x[2, 0]
                speed_y = next_x[3, 0]
                speed = math.sqrt(speed_x ** 2 + speed_y**2) * seconds * 1440 * 0.0059
                speed = "%.2f" % speed
                frame = cv2.circle(frame, (pos_x, pos_y), 5, (0, 255, 0), -1)
                frame = cv2.putText(frame, str(speed), (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                                    thickness=2)
        return frame
