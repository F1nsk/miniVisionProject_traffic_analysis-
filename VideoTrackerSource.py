from VideoTracker import VideoTracker
from ForHandIn import VideoTrackerFinal
from threading import Thread
import numpy as np
import math
import cv2


def get_dist(p1, p2):
    dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist

"""
vt = VideoTracker(video_path='D:/Dropbox/SDU/8_Semester/RMURV2/Videos/video3.mp4')

vt.display_trackbars('Trackbars')
vt.create_background_subtractor(10, 10)
vt.run_background_and_blobs('Display', 'Blobs')

"""
vt = VideoTrackerFinal(video_path='D:/Dropbox/SDU/8_Semester/RMURV2/Videos/video3.mp4', fps=25)

# pts_src = np.array([[476, 630], [821, 66], [1121, 237], [1050, 473]])
#
# P1 = [55.386164, 10.356182]
# P2 = [55.385093, 10.356107]
# P3 = [55.385315, 10.355210]
# P4 = [55.385761, 10.355274]
#
# dist_to_pixel = 500000
# new_p1 = [770, 500]
# new_p2 = [new_p1[0] - (P1[0] - P2[0]) * dist_to_pixel, new_p1[1] - (P1[1] - P2[1]) * dist_to_pixel]
# new_p3 = [new_p1[0] - (P1[0] - P3[0]) * dist_to_pixel, new_p1[1] - (P1[1] - P3[1]) * dist_to_pixel]
# new_p4 = [new_p1[0] - (P1[0] - P4[0]) * dist_to_pixel, new_p1[1] - (P1[1] - P4[1]) * dist_to_pixel]

# pts_dst = np.array([new_p1, new_p2, new_p3, new_p4])

pts_src = np.array([[112, 709], [820, 65], [1250, 267], [1050, 473]])

shift_x = -200
shift_y = -200
factor = 0.9
new_p1 = [41 * factor - shift_x, 809 * factor - shift_y]
new_p2 = [251 * factor - shift_x, 37 * factor - shift_y]
new_p3 = [651 * factor - shift_x, 171 * factor - shift_y]
new_p4 = [530 * factor - shift_x, 433 * factor - shift_y]

pts_dst = np.array([new_p1, new_p2, new_p3, new_p4])

# vt.make_predefined_transformation(pts_src, pts_dst, (900, 720))
vt.make_predefined_transformation(pts_src, pts_dst, (1000, 1000))

vt.create_background_subtractor(100, 50)
vt.run_optical_flow_with_kalman('Display', True)
"""
"""


