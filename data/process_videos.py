import os
from glob import glob

def avi2pngs(opt):
    #print(dir)
    dir = opt.dataroot
    fr = opt.frameRate
    isize = opt.loadSize
    videos = glob(dir+'/*.avi')
    for video in videos:
        foldername = video[:-4]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            print 'Processing video',video
	    if fr == 0:
	        os.system('ffmpeg -i '+video+' -s '+str(isize)+'x'+str(isize)+' -f image2 '+foldername+'/\%05d.jpg -hide_banner')
	    else:
                os.system('ffmpeg -i '+video+' -r '+str(fr)+' -s '+str(isize)+'x'+str(isize)+' -f image2 '+foldername+'/\%05d.jpg -hide_banner')
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
