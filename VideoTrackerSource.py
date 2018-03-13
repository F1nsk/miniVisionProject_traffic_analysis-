from VideoTracker import VideoTracker
from threading import Thread

vt = VideoTracker(video_path='D:/Dropbox/SDU/8_Semester/RMURV2/Videos/video3.mp4')

vt.create_background_subtractor(50, 16)
vt.run_tracker('Tracker', 2)

#vt.click_to_make_transformation('Display', 0)
#vt.display_trackbars('Trackbars')

#vt.run_background_and_blobs('Display', 'Blobs', True)