import os
from glob import glob
import numpy as np

def avi2pngs(opt):
    #print(dir)
    dir = opt.dataroot
    fr = opt.frameRate
    isize = opt.loadSize
    videos = glob(dir+'/*.avi')
    path = dir
    for video in videos:
        foldername = video[:-4]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            print 'Processing video',video
	    if fr == 0:
	        os.system('ffmpeg -i '+video+' -f image2 '+foldername+'/\%05d.png -hide_banner')
	    else:
                os.system('ffmpeg -i '+video+' -r '+str(fr)+' -s '+str(isize)+'x'+str(isize)+' -f image2 '+foldername+'/\%05d.jpg -hide_banner')

    # for KTH dataset
    if opt.file_list != None:
        f = open(opt.file_list,"r")
        files = f.readlines()
        tfiles = np.array(files)
        files_n = len(tfiles)
        for i in xrange(files_n):
            tokens = tfiles[i].split()
            src_path = path+'/'+tokens[0]+"_uncomp"
            if os.path.exists(src_path):
                low = int(tokens[1])
                high = int(tokens[2])
                tar_path = path +'/split/'+tokens[0]+"_uncomp_"+tokens[1]+"_"+tokens[2]
                if not os.path.exists(tar_path):
                    os.makedirs(tar_path)
                    for t in range(low,high+1):
                        src_file = src_path + "/%05d.png"%t
                        if os.path.isfile(src_file):
                            tar_file = tar_path+"/%05d.png"%t
                            os.system('cp '+src_file+' '+tar_file)
                        else:
                            print src_file+' is not exist!'

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
