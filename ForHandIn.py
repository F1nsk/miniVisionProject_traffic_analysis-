import cv2
import math
import numpy as np
from numpy import matlib
from threading import Thread
from Kalman import Kalman
import sys


class VideoTrackerFinal:
    """Class with different functions for tracking objects in videos"""

    def __init__(self, video_path, fps):
        self.videoPath = video_path
        self.video = cv2.VideoCapture(video_path)
        self.fps = fps

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

    def make_predefined_transformation(self, src, dst, size):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, image = self.video.read()
        
        cv2.imwrite('Before.png', image)

        self.h, status = cv2.findHomography(src, dst)
        self.size = size

        im_out = cv2.warpPerspective(image, self.h, size)
        im_out = self.rotateImage(im_out, -90)
        cv2.imwrite('After.png', im_out)

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

    # Calculate distance between two points.
    def distance_to_point(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # Check if a point close to 'point' is already being tracked.
    def is_point_being_tracked(self, point, min_distance):
        for trackedPoint in self.trackedPoints:
            distance = self.distance_to_point(point, trackedPoint)
            if distance < min_distance:
                return True

        return False

    # Detect blobs and add new points to track.
    def find_new_trackpoints(self, background):
        num_labels, labels, stats, centroids = self.blob_detection(background, 8)

        new_points_to_track = []
        counter = 0

        for point in centroids:
            if not self.is_point_being_tracked(point, 50):
                if stats[counter, cv2.CC_STAT_AREA] > 100:
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

        bbox = (left_x, top_y, width, height)
        return bbox

    def init_kalman(self):
        deltaT = 1/self.fps
        print(deltaT)

        F = np.matrix([[1, 0, deltaT, 0], [0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.matrix([[deltaT ** 2 / 2, 0], [0, deltaT ** 2 / 2], [deltaT, 0], [0, deltaT]])
        H = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        model_noise = 1
        measurement_noise = 0.1

        R = measurement_noise ** 2 * np.matlib.identity(4) / deltaT
        Q = model_noise ** 2 * np.matlib.identity(4) * deltaT

        init_mu = np.matrix([[0.], [0.]])  # This is the acceleration estimate
        init_P = np.matlib.identity(4)

        init_x = np.matrix([[0], [0], [0], [0]])
        new_kalman = Kalman(F, B, Q, H, R, init_x, init_mu, init_P, deltaT)

        return new_kalman, F, B, Q, H, R, init_mu, init_P, deltaT

    def run_optical_flow_with_kalman(self, window_name, use_transformation=False):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.on_mouse)

        # Init kalman filter params
        new_kalman, F, B, Q, H, R, init_mu, init_P, deltaT = self.init_kalman()
        kalman_array = [new_kalman]

        # Create a mask image for drawing purposes
        ret, old_frame = self.video.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        if use_transformation:
            old_gray = cv2.warpPerspective(old_gray, self.h, self.size)
            old_gray = self.rotateImage(old_gray, -90)

        mask = np.zeros_like(old_frame)

        has_completed_one_cycle = False

        while True:
            # Start timer
            timer = cv2.getTickCount()

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
            old_gray, kalman_array = self.increment_optical_flow_kalman(old_gray, frame_gray, mask, frame, p0,
                                                                        kalman_array)

            # Draw kalman
            self.draw_kalman_speed(frame, kalman_array, 40)

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

            # mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (255, 0, 0), -1)
            self.trackedPoints.append((a, b))

            if len(kalman_array) > 0:
                next_x = np.matrix([[a], [b], [a - c], [b - d]])
                kalman_array[i - points_removed].run_one_step(next_x)

        # Now update the previous frame and previous points
        old_frame = next_frame.copy()

        return old_frame, kalman_array

    def draw_kalman_speed(self, frame, kalman_array, min_iterations):
        for kalman in kalman_array:
            if kalman.get_number_of_iterations() > min_iterations:
                next_x = kalman.get_nextX()
                pos_x = int(round(next_x[0, 0]))
                pos_y = int(round(next_x[1, 0]))
                speed_x = next_x[2, 0]
                speed_y = next_x[3, 0]

                # 744 pixels on 157.5 meters

                seconds_pr_hour = 3600  # 60 * 60
                meters_pr_pixel = 0.0002115  # 157.5 / 744

                speed = math.sqrt(speed_x**2 + speed_y**2) * self.fps * seconds_pr_hour * meters_pr_pixel
                speed = "%.2f" % speed
                frame = cv2.circle(frame, (pos_x, pos_y), 5, (0, 255, 0), -1)
                frame = cv2.putText(frame, str(speed), (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                                    thickness=2)
        return frame
