from VideoTracker import VideoTracker
from threading import Thread

vt = VideoTracker(video_path='intersection.mp4')

<<<<<<< HEAD
vt.create_background_subtractor(50, 16)
vt.run_tracker('Tracker', 2)

#vt.click_to_make_transformation('Display', 0)
#vt.display_trackbars('Trackbars')

#vt.run_background_and_blobs('Display', 'Blobs', True)
=======
vt.create_background_subtractor(100, 100)
vt.click_to_make_transformation('Display', 0)
#vt.display_trackbars('Trackbars')

#True - "kører" med transformation - False - Kører ikke med transformation
vt.run_background_and_blobs('Display', 'Blobs', True)
>>>>>>> optimalFlow
