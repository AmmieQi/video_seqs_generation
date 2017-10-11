import os.path
import random
import math
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SeqDataset(BaseDataset):
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
        videonames = sorted(os.listdir(self.root))
	videonames = filter(lambda x: os.path.isdir(os.path.join(self.root,x)), videonames)
        for vname in videonames:
	    vdir = os.path.join(self.root, vname)
            self.videos += [sorted(make_dataset(vdir))]
            self.total_len += len(self.videos[-1])
        #self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.num_videos = len(self.videos)
        #assert(opt.resize_or_crop == 'resize_and_crop')
        self.transform = get_transform(opt)

        for iv in xrange(len(self.videos)):
            end_j = int(math.floor((len(self.videos[iv]) - self.len)/self.seq_stride))
	    for ii in xrange(end_j+1):
                self.indexs += [[iv,ii*self.seq_stride]]
	#print len(self.indexs)
    def __getitem__(self, index):
        video = self.videos[self.indexs[index][0]]
        num_imgs = len(video)
        index_begin = self.indexs[index][1] #random.randint(0, num_imgs - self.len)
        index_stop = index_begin + self.len - 1
        X = []
        Y = []
        X_paths = []
        Y_paths = []
	input_nc = self.opt.input_nc
        for i in range(0, self.seq_len):
            index_img = index_begin + i
            path_img = video[index_img]
            img = Image.open(path_img).convert('RGB')
            img = self.transform(img)
	    if input_nc == 1: # RGB to gray
	        tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
                img = tmp.unsqueeze(0)
            #img = img.unsqueeze(0)
	    X.append(img)
            X_paths.append(path_img)

	    #print X[-1].size()

        for i in range(self.seq_len, self.len):
            index_img = index_begin + i
            path_img = video[index_img]
            img = Image.open(path_img).convert('RGB')
            img = self.transform(img)
	    if input_nc == 1: # RGB to gray
	        tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
                img = tmp.unsqueeze(0)
 	    #img = img.unsqueeze(0)
            Y.append(img)
            Y_paths.append(path_img)

        return X, Y  #{'X': X, 'Y': Y,
                 #'X_paths': X_paths, 'Y_paths': Y_paths}

    def __len__(self):
        return len(self.indexs)

    def name(self):
        return 'SeqDataset'
