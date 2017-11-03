import os.path
import random
import math
import torchvision.transforms as transforms
import torch
from data.base_dataset import get_transform
from data.video_folder import make_dataset
from PIL import Image
import cv2
import time
import random
import imageio
import scipy.misc
import numpy as np
from joblib import Parallel, delayed
import time

class VidDataset():
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.seq_stride = opt.seq_stride
        self.videos = []
        self.indexs = []
        self.seq_len = opt.seq_len
        self.pre_len = opt.pre_len
        self.len = self.seq_len + self.pre_len
        self.total_len = 0
        self.batchSize = opt.batchSize
        self.loadSize = opt.loadSize
        self.fineSize = opt.fineSize
        self.flip = opt.flip
        self.input_nc = opt.input_nc
        #self.isFlip = opt.isFilp
        #videonames = sorted(os.listdir(self.root))
	    #videonames = filter(lambda x: os.path.isdir(os.path.join(self.root,x)), videonames)
        
        self.videos = sorted(make_dataset(self.root))
        #for vname in self.videos:
	    #    print vname
       
        self.num_videos = len(self.videos)
        
        self.transform = get_transform(opt)

    def get_minibatch(self):
        batch_idx = [random.randint(0,self.num_videos-1) for _ in range(0,self.batchSize)]
        flip = 0
        X = [] 
        Y = []
        
        itr_time = time.time()
        with Parallel(n_jobs=4) as parallel:
            output = parallel(delayed(load_data)(f,self.videos,self.loadSize,self.seq_len,self.pre_len,self.flip,self.input_nc) for f in batch_idx)
            '''for j in xrange(self.seq_len):
                xt = torch.FloatTensor(self.batchSize,self.input_nc, self.fineSize, self.fineSize).fill_(0)
                for i in xrange(self.batchSize):
                    xt[i] = output[i][0][j]
                X.append(xt)
            for k in xrange(self.pre_len):
                yt = torch.FloatTensor(self.batchSize,self.input_nc, self.fineSize, self.fineSize).fill_(0)
                for i in xrange(self.batchSize):
                    yt[i] = output[i][1][k]
                Y.append(yt)
            '''
            
        #output = [self.load_data(f) for f in batch_idx]   
        print len(output)
        print time.time()-itr_time
        return output

    def load_data(self, file_idx):
        vid_path = self.videos[file_idx]
        img_size = [self.loadSize, self.loadSize]
        K = self.seq_len
        T = self.pre_len
        X = []
        Y = []
        seq = []
        while True:
            try:
                vid = imageio.get_reader(vid_path,"ffmpeg")
                low = 1
                high = vid.get_length()-K-T+1
                if low == high:
                    stidx = 0
                else:
                    stidx = np.random.randint(low=low, high=high)
          
                for t in xrange(K+T):
                    img = cv2.resize(vid.get_data(stidx+t), (img_size[0],img_size[1]))
                    '''img = to_tensor(transform(img))
                    if self.input_nc == 1:
                        tmp =  img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
                        img = tmp.unsqueeze(0)'''
                    seq.append(img)
                if self.flip == 1:
                    tmpFlag = random.randint(0,1)
                    if tmpFlag == 1:
                        seq.reverse()
                X = [seq[i] for i in range(0,K)]
                Y = [seq[i] for i in range(K,K+T)]
                break
            except Exception:
            # In case the current video is bad load a random one 
                rep_idx = np.random.randint(low=0, high=self.num_videos)
                f_path = self.videos[rep_idx]
                vid_path = f_path

        return X, Y

    def get_minibatch_idx(self,shuffle=False):
        idx_list = np.arange(n, dtype="int32")
        if shuffle:
            random.shuffle(idx_list)
        minibatches = []
        minibatch_start = 0 
        for i in range(self.num_videos // self.batchSize):
            minibatches.append(idx_list[minibatch_start:
                                minibatch_start + self.batchSize])
            minibatch_start += self.batchSize

        if (minibatch_start != n): 
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)



    def __len__(self):
        return self.num_videos

    def name(self):
        return 'VidDataset'


def transform(image):
    return image/127.5 - 1.

def to_tensor(ndarray):
    return torch.from_numpy(ndarray)

def load_data(file_idx, videos, loadSize, K,T,flip,input_nc):
    vid_path = videos[file_idx]
    img_size = [loadSize, loadSize]
    num_videos = len(videos)
    X = []
    Y = []
    seq = []
    while True:
        try:
            vid = imageio.get_reader(vid_path,"ffmpeg")
            low = 1
            high = vid.get_length()-K-T+1
            if low == high:
                stidx = 0
            else:
                stidx = np.random.randint(low=low, high=high)
          
            for t in xrange(K+T):
                img = cv2.resize(vid.get_data(stidx+t), (img_size[0],img_size[1]))
                img = to_tensor(transform(img))
                #print img.size()
                if input_nc == 1:
                    tmp =  img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
                    img = tmp.unsqueeze(0)
                seq.append(img)
            if flip == 1:
                tmpFlag = random.randint(0,1)
                if tmpFlag == 1:
                    seq.reverse()
            X = [seq[i] for i in range(0,K)]
            Y = [seq[i] for i in range(K,K+T)]
            break
        except Exception:
        # In case the current video is bad load a random one 
            rep_idx = np.random.randint(low=0, high=num_videos-1)
            f_path = videos[rep_idx]
            vid_path = f_path

    return X,Y