import cv2
import numpy as np
from threading import Thread


class VideoTracker:
    """Class with different functions for tracking objects in videos"""

    def __init__(self, video_path):
        self.videoPath = video_path
        self.video = cv2.VideoCapture(video_path)

        self.clicked = False
        self.blur = 1
        self.kernelSize = 1
        self.it0 = 0
        self.it1 = 0
        self.it2 = 0

    # Function that is called whenever the mouse is clicked. Remember to set callback.
    def on_mouse(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouseX = x
            self.mouseY = y
            self.clicked = True

    # Click on four points in a square to make a transformation.
    def click_to_make_transformation(self, window_name, frame_number = 0):
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

    def subtract_background(self, use_transformation=False):
        ret, frame = self.video.read()

        frame = cv2.blur(frame, (self.blur, self.blur))

        if use_transformation:
            frame = cv2.warpPerspective(frame, self.h, self.size)

        fgmask = self.backgroundSubtractor.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernelSize, self.kernelSize))
        fgmask = cv2.erode(fgmask, kernel, iterations=self.it0)
        fgmask = cv2.dilate(fgmask, kernel, iterations=self.it1)
        fgmask = cv2.erode(fgmask, kernel, iterations=self.it2)

        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        return frame, fgmask

    # Subtract the background.
    def run_background_subtractor(self, window_name, use_transformation=False):
        while True:
            frame, background = self.subtract_background(use_transformation)

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
                self.blob_detection(window_name, background, frame, 4)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

            has_completed_one_cycle = True

    # Use blob detection on an image and display the found blobs on another image.
    def blob_detection(self, window_name, detect_image, display_image, connectivity=4):
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
            frame, background = self.subtract_background(use_transformation)

            display_image = cv2.addWeighted(frame, 0.5, background, 0.5, 0.0)

            cv2.imshow(window_name_background, display_image)

            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

            if has_completed_one_cycle:
                self.blob_detection(window_name_blobs, background, frame, 8)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

            has_completed_one_cycle = True