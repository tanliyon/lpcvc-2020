import cv2
import os
def FrameCapture(video_path, frame_path):

    if video_path == '':
        raise ValueError('No input file provided')
    if frame_path == '':
        frame_path == './allframes/sub/'
    
    try:
        os.mkdir('./allframes')
        os.mkdir('./allframes/sub')
    except FileExistsError:
        pass

    vidObj = cv2.VideoCapture(video_path)
    count = 0
    success = 1

    while success:
        success, image = vidObj.read()
        cv2.imwrite('./allframes/sub/frame%d.jpg' % count, image)
        count += 1

    return frame_path