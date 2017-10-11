import os
from glob import glob

def avi2pngs(dir):
    #print(dir)
    videos = glob(dir+'/*.avi')
    for video in videos:
        foldername = video[:-4]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            print 'Processing video',video
            os.system('ffmpeg -i '+video+' '+foldername+'/\%05d.png -hide_banner')
            '''cap = cv2.VideoCapture(video)
            if cap.isOpened():
                success = True
            else:
                success = False
                print 'Open video failure!'
            frame_count = 1
            while(success):
                success, frame = cap.read()
                print 'Read a new frame: ',frame_count
                params = []
                #params.append(cv2.CV_IMWRITE_PXM_BINARY)
                #params.append(1)
                cv2.imwrite(foldername+'/%06d.png' % frame_count, frame)
                frame_count += 1
            cap.release()'''
