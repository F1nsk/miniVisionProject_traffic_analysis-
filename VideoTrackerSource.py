from VideoTracker import VideoTracker
from threading import Thread

vt = VideoTracker(video_path='intersection.mp4')

vt.create_background_subtractor(100, 100)
vt.click_to_make_transformation('Display', 0)
#vt.display_trackbars('Trackbars')

#True - "kører" med transformation - False - Kører ikke med transformation
vt.run_background_and_blobs('Display', 'Blobs', True)
