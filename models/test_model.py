import numpy as np
import random
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        seq_len = opt.seq_len
        self.seq_len = opt.seq_len
        self.pre_len = opt.pre_len
        low_shape = (opt.fineSize/16, opt.fineSize/16)
        high_shape = (opt.fineSize/4, opt.fineSize/4)
        #self.input_seq = self.Tensor(nb, seq_len, opt.input_nc, size, size)
        self.input_nc = opt.input_nc
        
        self.content_nc = 64
    
        self.netCE = networks.content_E(opt.input_nc, 1, self.content_nc, opt.latent_nc,
                                                  opt.ngf, opt.which_model_E, opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type)
        self.netOPL = networks.offsets_P(low_shape, seq_len, opt.ngf*8, opt.ngf*8, opt.ngf, opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type,relu='leakyrelu')
        self.netOPH = networks.offsets_P(high_shape, seq_len, opt.ngf*2, opt.ngf*2, opt.ngf, opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type,relu='leakyrelu')

        self.netME = networks.motion_E(opt.input_nc, opt.latent_nc,
                                                opt.ngf, 'resnet_3blocks', opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type)
        self.netMP = networks.motion_P(low_shape, seq_len, opt.latent_nc,
                                            opt.latent_nc, opt.ngf, opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type)
      
        which_epoch = opt.which_epoch
        self.load_network(self.netCE, 'CE', which_epoch)
        #self.load_network(self.netOPH, 'OPH', which_epoch)
        #self.load_network(self.netOPL, 'OPL', which_epoch)
        self.load_network(self.netME, 'ME', which_epoch)
        self.load_network(self.netMP, 'MP', which_epoch)

        self.criterionPixel = torch.nn.MSELoss()
        self.criterionPre   = torch.nn.L1Loss()
        self.criterionFeat = torch.nn.MSELoss()
        self.criteriondiff = torch.nn.L1Loss()
        self.criterionSim = torch.nn.MSELoss()
        self.criterionFlow = torch.nn.MSELoss()
        #self.criterionTrip = torch.nn.TripletMarginLoss(margin=1, p=2)
        self.criterionTrip = networks.TripLoss(p=2)
        self.criterionGDL = networks.GDLLoss(opt.input_nc, tensor=self.Tensor)

        print('---------- Networks initialized -------------')
        print('------------- Content Encoder ---------------')
        networks.print_network(self.netCE)
        print('------------ Offsets Predictor Low --------------')
        networks.print_network(self.netOPL)
        print('------------ Offsets Predictor High --------------')
        networks.print_network(self.netOPH)
        print('-------------- Motion Encoder ---------------')
        networks.print_network(self.netME)
        print('------------- Motion Predictor --------------')
        networks.print_network(self.netMP)
        #print('------------ Generator/Decoder --------------')
        #networks.print_network(self.netG)


    def set_input(self,input):
        # X: batchsize, seq_len_x, inchains, size(0), size(1)
        # Y: batchsize, seq_len_y, inchains, size(0), size(1)
        input_X = input[0]
        input_Y = input[1]
        self.image_paths   = input[2]
        #self.input_X.resize_(input_X.size()).copy_(input_X)
        #self.input_Y.resize_(input_Y.size()).copy_(input_Y)
        if len(self.gpu_ids) > 0:
            self.input_X = [x.clone().cuda() for x in input_X]
            self.input_Y = [y.clone().cuda() for y in input_Y]
        else:
            self.input_X = [x.clone() for x in input_X]
            self.input_Y = [y.clone() for y in input_Y]
        #self.image_paths = input['X_paths']

    def forward_G(self):
        self.real_X = [Variable(x) for x in self.input_X]
        self.real_Y = [Variable(y) for y in self.input_Y]

    def test(self):
        self.real_X = [Variable(x,volatile=True) for x in self.input_X]
        self.real_Y = [Variable(y,volatile=True) for y in self.input_Y]
        self.backward_G()
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):
        lambda_fea = self.opt.lambda_fea
        lambda_pix = self.opt.lambda_pix
        lambda_dif = self.opt.lambda_dif
        lambda_pre = self.opt.lambda_pre
        lambda_gan = self.opt.lambda_gan
        lambda_tra = 10
        lambda_gdl = 5
        #for p in self.netME.parameters(): # reset requires_grad
        #    p.requires_grad = True # they are set to False below in netG update
        # forward
        batch_size = self.real_X[0].size(0)

        self.latent_y = [self.netME.forward(realy) for realy in self.real_Y]#     
        #self.encs_xs = [self.netCE.forward(realx,[Variable(self.latent_y[0].data)]) for realx in self.real_X]
        #self.encs_ys = [self.netCE.forward(realy,[Variable(self.latent_y[0].data)]) for realy in self.real_Y] 
        
       
        '''hidden_state_OPL = self.netOPL.init_hidden(batch_size)
        hidden_state_OPH = self.netOPH.init_hidden(batch_size)
        # global_feature_maps
        #initialization of the Offsets Predictor
        ln = 3
        hn = 1
        FL = self.encs_xs[0][0][ln] #[bs,,nc,s0,s1] low level feature
        HL = self.encs_xs[0][0][hn] # high level feature
        init_offsets_l = self.netOPL.init_offset(batch_size)
        init_offsets_h = self.netOPH.init_offset(batch_size)
        
        hidden_state_OPL, pred_offsets_l, SFL = self.netOPL.forward(FL,init_offsets_l,hidden_state_OPL)
        hidden_state_OPH, pred_offsets_h, SHL = self.netOPH.forward(HL,init_offsets_h,hidden_state_OPH)
        self.loss_flow_l = self.criterionFlow(SFL, Variable(self.encs_xs[1][0][ln].data))
        self.loss_flow_h = self.criterionFlow(SHL, Variable(self.encs_xs[1][0][hn].data))
        self.loss_flow_trip_h = self.criterionTrip(SHL,self.encs_xs[1][0][hn],HL) 
        self.loss_flow_trip_l = self.criterionTrip(SFL,self.encs_xs[1][0][ln],FL)
        FL = self.encs_xs[1][0][ln]
        HL = self.encs_xs[1][0][hn]
        for t in range(2, self.seq_len):
            hidden_state_OPL, pred_offsets_l, SFL = self.netOPL.forward(FL,pred_offsets_l,hidden_state_OPL)
            hidden_state_OPH, pred_offsets_h, SHL = self.netOPH.forward(HL,pred_offsets_h,hidden_state_OPH)
            self.loss_flow_l += self.criterionFlow(SFL, Variable(self.encs_xs[t][0][ln].data))
            self.loss_flow_h += self.criterionFlow(SHL, Variable(self.encs_xs[t][0][hn].data))
            self.loss_flow_trip_h += self.criterionTrip(SHL,self.encs_xs[t][0][hn],HL) 
            self.loss_flow_trip_l += self.criterionTrip(SFL,self.encs_xs[t][0][ln],FL)
            FL = self.encs_xs[t][0][ln]
            HL = self.encs_xs[t][0][hn]

        ##------------------------------------------prediction------------------------------------------##

        hidden_state_OPL, pred_offsets_l, SFL = self.netOPL.forward(FL,pred_offsets_l,hidden_state_OPL)
        hidden_state_OPH, pred_offsets_h, SHL = self.netOPH.forward(HL,pred_offsets_h,hidden_state_OPH)
        self.loss_flow_l += self.criterionFlow(SFL, Variable(self.encs_ys[0][0][ln].data))
        self.loss_flow_h += self.criterionFlow(SHL, Variable(self.encs_ys[0][0][hn].data))
        self.loss_flow_trip_h += self.criterionTrip(SHL,self.encs_ys[0][0][hn],HL)
        self.loss_flow_trip_l += self.criterionTrip(SFL,self.encs_ys[0][0][ln],FL)
        self.low_feats = [SFL]
        self.high_feats =[SHL]

        for t in range(1,self.pre_len):
            hidden_state_OPL, pred_offsets_l, SFL = self.netOPL.forward(SFL,pred_offsets_l,hidden_state_OPL)
            hidden_state_OPH, pred_offsets_h, SHL = self.netOPH.forward(SHL,pred_offsets_h,hidden_state_OPH)
            self.loss_flow_l += self.criterionFlow(SFL, Variable(self.encs_ys[t][0][ln].data))
            self.loss_flow_h += self.criterionFlow(SHL, Variable(self.encs_ys[t][0][hn].data))
            self.loss_flow_trip_h += self.criterionTrip(SHL,self.encs_ys[t][0][hn],HL)
            self.loss_flow_trip_l += self.criterionTrip(SFL,self.encs_ys[t][0][ln],FL)
            self.low_feats += [SFL]
            self.high_feats +=[SHL]
        '''
        latent = [self.latent_y[0]]
        enc_xt, fake = self.netCE.forward(self.real_X[self.seq_len-1],latent)
        self.fakes = [fake]
        #self.loss_pix = self.criterionPixel(self.fakes[0], self.real_Y[0])
        #self.loss_gdl = self.criterionGDL(self.fakes[0], self.real_Y[0])

        for t in range(1,self.pre_len):
            latent = [self.latent_y[t]]
            enc_xt, fake = self.netCE.forward(self.real_X[self.seq_len-1],latent)
            self.fakes += [fake]
            #self.loss_pix += self.criterionPixel(self.fakes[t],self.real_Y[t])
            #self.loss_gdl += self.criterionGDL(self.fakes[t], self.real_Y[t])


        #self.loss_flow_l = self.loss_flow_l*lambda_tra
        #self.loss_flow_h = self.loss_flow_h*lambda_tra
        #self.loss_flow_trip_h = self.loss_flow_trip_h/10
        #self.loss_flow_trip_l = self.loss_flow_trip_l/10
        #self.loss_pix = self.loss_pix*lambda_pix
        #self.loss_gdl = self.loss_gdl*lambda_gdl

    def optimize_generator(self):
        self.forward_G()
        self.backward_G()

    def optimize_parameters(self):
        self.optimize_generator()

    def get_current_errors(self):        
        GAN = 0#self.loss_gan.data[0]
        PIX = self.loss_pix.data[0]
        GDL = self.loss_gdl.data[0]
        FLOW_L = self.loss_flow_l.data[0]
        FLOW_H = self.loss_flow_h.data[0]        
        TRIP_L = self.loss_flow_trip_l.data[0]
        TRIP_H = self.loss_flow_trip_h.data[0]
        #DES = self.loss_flow_trip_y.data[0]
        SIM = self.loss_sim.data[0]
        return OrderedDict([('Pixel', PIX), ('GDL', GDL), ('Trans_2',FLOW_H),('Trans_4',FLOW_L),('TRIP_2',TRIP_H),('TRIP_4',TRIP_L),('SIM',SIM)])       

    def get_current_visuals(self):
        images = []
        for i in xrange(self.seq_len):
            name = 'gt_%04d'%i
            image = util.tensor2im(self.real_X[i].data)
            images += [(name,image)]
        
        for i in xrange(self.pre_len):
            name = 'gt_%04d'%(i+self.seq_len)
            image = util.tensor2im(self.real_Y[i].data)
            images += [(name,image)]

        for i in xrange(self.seq_len):
            name = 'pred_%04d'%i
            image = util.tensor2im(self.real_X[i].data)
            images += [(name,image)]

        for j in xrange(self.pre_len):
            name = 'pred_%04d'%(j+self.seq_len)            
            image = util.tensor2im(self.fakes[j].data)
            images += [(name,image)]
              
        return OrderedDict(images)
