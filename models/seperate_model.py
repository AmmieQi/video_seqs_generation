﻿import numpy as np
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


class SeperateModel(BaseModel):
    def name(self):
        return 'SeperateModel'

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
        self.use_cycle = opt.use_cycle
        
        # Content_Encoder, Motion_Encoder, Motion_Predictor, Overall_Generator, Overall_D
        # opt.input_nc = 3
        # opt.output_nc = 3
        #self.input_nc = opt.input_nc * opt.seq_len

        #self.netCE = networks.content_E(self.input_nc, opt.latent_nc,
        #                                          opt.ngf, opt.which_model_E, opt.norm, not opt.no_dropout, self.gpu_ids)
        # output_nc opt.ngf*8 256
        assert(opt.which_model_E=='unet_128')
        assert(opt.which_model_netG=='unet_128')
        self.netCE = networks.content_E(opt.input_nc*seq_len, 64,
                                                  opt.ngf, opt.which_model_E, opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type)
        self.netOPL = networks.offsets_P(low_shape, seq_len, 64, 64, opt.ngf, opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type)
        self.netOPH = networks.offsets_P(high_shape, seq_len, opt.ngf*2, opt.ngf*2, opt.ngf, opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type)

        self.netME = networks.motion_E(opt.input_nc, opt.latent_nc,
                                                opt.ngf/2, 'resnet_3blocks', opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type)
        self.netMP = networks.motion_P(low_shape, seq_len, opt.latent_nc,
                                            opt.latent_nc, opt.ngf, opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type)
        self.netG = networks.define_G(low_shape, seq_len, 64,opt.latent_nc,
                                            opt.output_nc, opt.ngf,opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,opt.init_type)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.fineSize, opt.output_nc, 256,opt.ndf,
                                        opt.which_model_netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids,opt.init_type)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netCE, 'CE', which_epoch)
            self.load_network(self.netOPH, 'OPH', which_epoch)
            self.load_network(self.netOPL, 'OPL', which_epoch)
            self.load_network(self.netME, 'ME', which_epoch)
            self.load_network(self.netMP, 'MP', which_epoch)
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionPixel = torch.nn.L1Loss()#torch.nn.SmoothL1Loss()#
            self.criterionPre   = torch.nn.L1Loss()
            self.criterionFeat = torch.nn.MSELoss()
            self.criteriondiff = torch.nn.L1Loss()
            self.criterionSim = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_CE = torch.optim.Adam(self.netCE.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_OPH = torch.optim.Adam(self.netOPH.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_OPL = torch.optim.Adam(self.netOPL.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_ME = torch.optim.Adam(self.netME.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_MP = torch.optim.Adam(self.netMP.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_CE)
            self.optimizers.append(self.optimizer_OPH)
            self.optimizers.append(self.optimizer_OPL)
            self.optimizers.append(self.optimizer_ME)
            self.optimizers.append(self.optimizer_MP)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))


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
        print('------------ Generator/Decoder --------------')
        networks.print_network(self.netG)
        if self.isTrain:
            print('-------------- Discriminator ---------------')
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self,input):
        # X: batchsize, seq_len_x, inchains, size(0), size(1)
        # Y: batchsize, seq_len_y, inchains, size(0), size(1)
        input_X = input[0]
        input_Y = input[1]
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

    def forward_P(self):
        self.real_X = [Variable(x) for x in self.input_X]
        self.real_Y = [Variable(y) for y in self.input_Y]

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)

    # get image paths
    def get_image_paths(self):
        return 0#self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        #one = Variable(torch.FloatTensor(pred_real.size(0),1).fill_(1).cuda(),requires_grad=False)
        loss_D_real = self.criterionGAN(pred_real, True) #True
        # Fake
        pred_fake = netD.forward(fake.detach())
        #zero = Variable(torch.FloatTensor(pred_fake.size(0),1).fill_(0).cuda(),requires_grad=False)
        loss_D_fake = self.criterionGAN(pred_fake, False) #False
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake = self.fake_pool.query(self.fakes)
        self.real = Variable(torch.cat(self.input_Y, 0))
        self.loss_D = self.backward_D_basic(self.netD, self.real, fake)

    def backward_G(self):
        #lambda_idt = self.opt.identity
        lambda_fea = self.opt.lambda_fea
        lambda_pix = self.opt.lambda_pix
        lambda_dif = self.opt.lambda_dif
        lambda_pre = self.opt.lambda_pre
        lambda_gan = self.opt.lambda_gan

        #for p in self.netME.parameters(): # reset requires_grad
        #    p.requires_grad = True # they are set to False below in netG update
        # forward
        self.realX = torch.cat(self.real_X,0)
        self.X_T = torch.cat(self.real_X,1)
        self.realY = torch.cat(self.real_Y,0)
        self.Y_T = torch.cat(self.real_Y,1)
        batch_size = self.real_X[0].size(0)

        self.T = random.randint(0,self.pre_len-1)


        self.encs = self.netCE.forward(self.X_T)
        #self.encs_y = self.netCE.forward(self.Y_T)
        #self.loss_sim = self.criterionSim(self.encs[-1],Variable(self.encs_y[-1].data))*10
        ## self.encs[0] is the global feature
        hidden_state_OPL = self.netOPL.init_hidden(batch_size)
        hidden_state_OPH = self.netOPH.init_hidden(batch_size)
        # global_feature_maps
        #initialization of the Offsets Predictor
        FL = self.encs[-1].unsqueeze(1) #[bs,1,nc,s0,s1] low level feature
        HL = self.encs[1].unsqueeze(1) # high level feature
        init_offsets_l = Variable(torch.zeros(batch_size,1,FL.size(2)*2,FL.size(3),FL.size(3))).cuda()
        init_offsets_h = Variable(torch.zeros(batch_size,1,HL.size(2)*2,HL.size(3),HL.size(3))).cuda()
        hidden_state_OPL, pred_offsets_l = self.netOPL.forward(torch.cat([FL,init_offsets_l],2),hidden_state_OPL)
        self.offsets_l = [pred_offsets_l]
        hidden_state_OPH, pred_offsets_h = self.netOPH.forward(torch.cat([HL,init_offsets_h],2),hidden_state_OPH)
        self.offsets_h = [pred_offsets_h]


        self.motion_codes_y = torch.split(self.netME.forward(self.realY),batch_size,dim=0)  ############################### 1
        
        '''if self.T == 0:
            offsets = [self.offsets_h[0].squeeze(1), self.offsets_l[0].squeeze(1)]
            self.fakes = [self.netG.forward(self.encs,self.motion_codes_y[0].squeeze(1), offsets)]
            #self.fakes = [self.netG.forward(self.encs,tmp.squeeze(1), self.offsets[0].squeeze(1))]
            self.loss_gan = self.criterionGAN(self.netD.forward(self.fakes[0]), True)*lambda_gan
            self.loss_pix = self.criterionPixel(self.fakes[0],self.real_Y[0])*lambda_pix
        else:
            for t in range(1,self.T+1):
                hidden_state_OPL, pred_offsets_l = self.netOPL.forward(torch.cat([FL,pred_offsets_l],2),hidden_state_OPL)
                self.offsets_l += [pred_offsets_l]
                hidden_state_OPH, pred_offsets_h = self.netOPH.forward(torch.cat([HL,pred_offsets_h],2),hidden_state_OPH)
                self.offsets_h += [pred_offsets_h]
                
            offsets = [self.offsets_h[self.T].squeeze(1), self.offsets_l[self.T].squeeze(1)]
            self.fakes = [self.netG.forward(self.encs,self.motion_codes_y[self.T].squeeze(1),offsets)]
            # self.fakes += [self.netG.forward(self.encs,tmp.squeeze(1),self.offsets[t].squeeze(1))]
            self.loss_gan = self.criterionGAN(self.netD.forward(self.fakes[0]), True)*lambda_gan
            self.loss_pix = self.criterionPixel(self.fakes[0],self.real_Y[self.T])*lambda_pix
            # caculate motion_feats
        '''
        
        offsets = [self.offsets_h[0].squeeze(1), self.offsets_l[0].squeeze(1)]
        self.fakes = [self.netG.forward(self.encs,self.motion_codes_y[0].squeeze(1), offsets)]
        #self.fakes = [self.netG.forward(self.encs,tmp.squeeze(1), self.offsets[0].squeeze(1))]
        self.loss_gan = self.criterionGAN(self.netD.forward(self.fakes[0]), True)*lambda_gan
        self.loss_pix = self.criterionPixel(self.fakes[0],self.real_Y[0])*lambda_pix
        
        for t in range(1,self.pre_len):
            hidden_state_OPL, pred_offsets_l = self.netOPL.forward(torch.cat([FL,pred_offsets_l],2),hidden_state_OPL)
            self.offsets_l += [pred_offsets_l]
            hidden_state_OPH, pred_offsets_h = self.netOPH.forward(torch.cat([HL,pred_offsets_h],2),hidden_state_OPH)
            self.offsets_h += [pred_offsets_h]
                
            offsets = [self.offsets_h[-1].squeeze(1), self.offsets_l[-1].squeeze(1)]
            self.fakes += [self.netG.forward(self.encs,self.motion_codes_y[t].squeeze(1),offsets)]
        
            self.loss_gan += self.criterionGAN(self.netD.forward(self.fakes[t]), True)*lambda_gan
            self.loss_pix += self.criterionPixel(self.fakes[t],self.real_Y[t])*lambda_pix
        


        '''self.motion_codes = torch.split(self.netME.forward(self.realX), batch_size,dim=0)
        hidden_state_MP = self.netMP.init_hidden(batch_size)
        #print hidden_state_MP[0][0].size()
        #print self.motion_codes[-1].size()
        hidden_state_M, pre_feat = self.netMP.forward(torch.cat(self.motion_codes,1), hidden_state_MP)
        self.pre_feats = [pre_feat]
        self.loss_pre = self.criterionPre(pre_feat,Variable(self.motion_codes_y[0].data))*lambda_pre

        self.fakes = [self.netG.forward(self.encs,pre_feat.squeeze(1))]
        self.loss_gan = self.criterionGAN(self.netD.forward(self.fakes[0]), True)*lambda_gan
        self.loss_pix = self.criterionPixel(self.fakes[0],self.real_Y[0])*lambda_pix
        for t in range (1, self.pre_len):
            hidden_state_M, pre_feat = self.netMP.forward(self.pre_feats[-1],hidden_state_M)
            self.pre_feats += [pre_feat]
            self.loss_pre += self.criterionPre(pre_feat,Variable(self.motion_codes_y[t].data))*lambda_pre
            self.fakes += [self.netG.forward(self.encs,pre_feat.squeeze(1))]
            self.loss_gan += self.criterionGAN(self.netD.forward(self.fakes[t]), True)*lambda_gan
            self.loss_pix += self.criterionPixel(self.fakes[t],self.real_Y[t])*lambda_pix
        '''
        #fakes = torch.cat(self.fakes,0)
        #for p in self.netME.parameters(): # reset requires_grad
        #    p.requires_grad = False # they are set to False below in netG update

        '''p0 = random.randint(0,self.pre_len-1)
        src_diff = self.fakes[0] - self.fakes[p0]
        tar_diff = self.real_Y[0] - self.real_Y[p0]
        self.loss_dif = self.criteriondiff(src_diff,tar_diff)*lambda_dif/self.pre_len
        #self.loss_dif = self.criteriondiff(src_diff,tar_diff)*lambda_dif/self.pre_len
        for t in xrange(1,self.pre_len):
            pt = random.randint(0,self.pre_len-1)
            src_diff = self.fakes[t] - self.fakes[pt]
            tar_diff = self.real_Y[t] - self.real_Y[pt]
            self.loss_dif += self.criteriondiff(src_diff,tar_diff)*lambda_dif/self.pre_len
        '''
        #self.fake_feats = self.netME.forward(fakes)
        #self.real_feats = Variable(torch.cat(self.motion_codes_y,0).data)
        #all_pre_feats = torch.cat(self.pre_feats,0)

        #self.loss_gan = self.criterionGAN(self.netD.forward(fakes), True)*lambda_gan
        #self.loss_pix = self.criterionPixel(fakes, self.realY)*lambda_pix
        #self.loss_fea = self.criterionFeat(self.fake_feats,self.real_feats)*lambda_fea
        #self.loss_pre = self.criterionPre(all_pre_feats, self.real_feats)*lambda_pre

        #self.loss_dif.backward()
        self.loss_G = self.loss_pix + self.loss_gan #+ self.loss_dif #+ self.loss_sim #+ self.loss_fea #+ self.loss_pre
        self.loss_G.backward()

    def backward_cycle(self):
        #lambda_idt = self.opt.identity
        lambda_fea = self.opt.lambda_fea
        lambda_pix = self.opt.lambda_pix
        lambda_dif = self.opt.lambda_dif
        lambda_pre = self.opt.lambda_pre
        lambda_gan = self.opt.lambda_gan

        #for p in self.netME.parameters(): # reset requires_grad
        #    p.requires_grad = True # they are set to False below in netG update
        # forward

        flag = random.randint(0,1)
        if flag == 1:
            tmp = list(reversed(self.real_Y))
            self.real_Y = list(reversed(self.real_X))
            self.real_X = tmp

        realX = torch.cat(self.real_X,0)
        realX_T = torch.cat(self.real_X,1)
        realY = torch.cat(self.real_Y,0)
        #realY_T = torch.cat(self.real_Y,1)

        self.real_X_INV = list(reversed(self.real_X))
        self.real_Y_INV = list(reversed(self.real_Y))

        realX_INV = torch.cat(self.real_X_INV,0)
        realX_T_INV = torch.cat(self.real_X_INV,1)
        #realY_INV = torch.cat(self.real_Y_INV,0)
        #realY_T_INV = torch.cat(self.real_Y_INV,1)

        batch_size = self.real_X[0].size(0)

        ############################################################################################################################################
        ########### X --> Y'
        encs_X = self.netCE.forward(realX_T)
        motion_y = torch.split(self.netME.forward(realY),batch_size,dim=0)  ############################### 1
        self.fakes_y = []
        for t in xrange(self.pre_len):
            self.fakes_y += [self.netG.forward(encs_X,motion_y[t].squeeze(1))]
        # caculate motion_feats
        motion_x = torch.split(self.netME.forward(realX), batch_size,dim=0)
        hidden_state_MP_y = self.netMP.init_hidden(batch_size)
        hidden_state_M_y, pre_feat_y = self.netMP.forward(torch.cat(motion_x,1), hidden_state_MP_y)
        self.pre_feats_y = [pre_feat_y]
        #self.loss_pre = self.criterionPre(pre_feat,Variable(motion_codes_y[0].data))*lambda_pre
        #self.fakes_y = [self.netG.forward(encs_X,pre_feat_y.squeeze(1))]
        for t in range (1, self.pre_len):
            hidden_state_M_y, pre_feat_y = self.netMP.forward(self.pre_feats_y[-1],hidden_state_M_y)
            self.pre_feats_y += [pre_feat_y]
            #self.loss_pre += self.criterionPre(pre_feat,Variable(self.motion_codes_y[t].data))*lambda_pre
            #self.fakes_y += [self.netG.forward(encs_X,pre_feat_y.squeeze(1))]

        fakes_y = torch.cat(self.fakes_y,0)
        #for p in self.netME.parameters(): # reset requires_grad
        #    p.requires_grad = False # they are set to False below in netG update

        p0 = random.randint(0,self.pre_len-1)
        self.loss_dif = self.criteriondiff((self.fakes_y[0] - self.fakes_y[p0]),(self.real_Y[0]-self.real_Y[p0]))*lambda_dif/self.pre_len
        for t in xrange(1,self.pre_len):
            pt = random.randint(0,self.pre_len-1)
            self.loss_dif += self.criteriondiff((self.fakes_y[t] - self.fakes_y[pt]),(self.real_Y[t]-self.real_Y[pt]))*lambda_dif/self.pre_len

        #self.fake_feats = self.netME.forward(fakes)
        #self.real_feats = Variable(torch.cat(self.motion_codes_y,0).data)
        #all_pre_feats = torch.cat(self.pre_feats,0)
        self.loss_gan = self.criterionGAN(self.netD.forward(fakes_y), True)*lambda_gan
        self.loss_pix = self.criterionPixel(fakes_y, realY)*lambda_pix
        #self.loss_fea = self.criterionFeat(self.fake_feats,self.real_feats)*lambda_fea
        #self.loss_pre = self.criterionPre(all_pre_feats, self.real_feats)*lambda_pre

        #self.loss_pre.backward()
        ########################################################################
        ##### Fakes
        self.fakes_y.reverse()
        fake_Y_T = torch.cat(self.fakes_y,1)
        fakeY = torch.cat(self.fakes_y,0)
        ############################################################################################################################################
        ########### Y' --> X''
        encs_fake_Y = self.netCE.forward(fake_Y_T)
        motion_fake_y = torch.split(self.netME.forward(fakeY),batch_size,dim=0)

        motion_x_inv = torch.split(self.netME.forward(realX_INV),batch_size,dim=0)
        self.fakes_x = []
        for t in xrange(self.pre_len):
            self.fakes_x += [self.netG.forward(encs_fake_Y,motion_x_inv[t].squeeze(1))]

        hidden_state_MP_x = self.netMP.init_hidden(batch_size)
        hidden_state_M_x, pre_feat_x = self.netMP.forward(torch.cat(motion_fake_y,1), hidden_state_MP_x)
        self.pre_feats_x = [pre_feat_x]
        #self.loss_pre = self.criterionPre(pre_feat,Variable(motion_codes_y[0].data))*lambda_pre
        #self.fakes_x = [self.netG.forward(encs_fake_Y,pre_feat_x.squeeze(1))]
        for t in range (1, self.pre_len):
            hidden_state_M_x, pre_feat_x = self.netMP.forward(self.pre_feats_x[-1],hidden_state_M_x)
            self.pre_feats_x += [pre_feat_x]
            #self.loss_pre += self.criterionPre(pre_feat,Variable(self.motion_codes_y[t].data))*lambda_pre
            #self.fakes_x += [self.netG.forward(encs_fake_Y,pre_feat_x.squeeze(1))]

        fakes_x = torch.cat(self.fakes_x,0)
        #for p in self.netME.parameters(): # reset requires_grad
        #    p.requires_grad = False # they are set to False below in netG update

        p0 = random.randint(0,self.pre_len-1)
        self.loss_dif_x = self.criteriondiff((self.fakes_x[0] - self.fakes_x[p0]),(self.real_X_INV[0]-self.real_X_INV[p0]))*lambda_dif/self.pre_len
        for t in xrange(1,self.pre_len):
            pt = random.randint(0,self.pre_len-1)
            self.loss_dif_x += self.criteriondiff((self.fakes_x[t] - self.fakes_x[pt]),(self.real_X_INV[t]-self.real_X_INV[pt]))*lambda_dif/self.pre_len

        #self.fake_feats = self.netME.forward(fakes)
        #self.real_feats = Variable(torch.cat(self.motion_codes_y,0).data)
        #all_pre_feats = torch.cat(self.pre_feats,0)
        self.loss_gan_x = self.criterionGAN(self.netD.forward(fakes_x), True)*lambda_gan
        self.loss_pix_x = self.criterionPixel(fakes_x, realX_INV)*lambda_pix

        ##############################################################################3
        ######## Overall loss
        self.fakes = list(reversed(self.fakes_y)) #+ self.fakes_x
        self.fakes_x.reverse()
        #self.loss_G = self.loss_gan + self.loss_gan_x + self.loss_dif + self.loss_dif_x #+ self.loss_pre
        self.loss_G = self.loss_pix + self.loss_gan + self.loss_pix_x + self.loss_gan_x + self.loss_dif + self.loss_dif_x #+ self.loss_pre
        self.loss_G.backward()
        #print 'generation loss: ', self.loss_gan.data[0], ' pixel loss: ', self.loss_pix.data[0], ' feature loss: ', self.loss_fea.data[0], ' overall loss: ', self.loss_G.data[0]

    def backward_P(self):
        # forward
        self.motion_codes_p = [self.netME.forward(real) for real in self.real_X] # [bs,1, oc, s0, s1]*len(self.real_X)
        batch_size = self.motion_codes_p[0].size(0)
        hidden_state_MP = self.netMP.init_hidden(batch_size)
        self.mcode_p = self.netMP.forward(torch.cat(self.motion_codes_p,1), hidden_state_MP)
        hidden_state_M = self.mcode_p[0]
        self.pre_feats_p = [self.mcode_p[1]]
        self.real_fea_p = [self.netME.forward(self.real_Y[0])]
        self.loss_pre = self.criterionPre(self.pre_feats_p[-1], Variable(self.real_fea_p[-1].data))

        for t in xrange(1, self.pre_len):
            hidden_state_M, outmcode = self.netMP.forward(self.pre_feats_p[-1],hidden_state_M)
            self.pre_feats_p += [outmcode]
            self.real_fea_p += [self.netME.forward(self.real_Y[t])]
            self.loss_pre += self.criterionPre(self.pre_feats_p[-1],Variable(self.real_fea_p[-1].data))

        self.loss_P = self.loss_pre
        self.loss_P.backward()
        #print 'output shape: ', len(self.pre_feats_p), ' output size: ', self.pre_feats_p[0].size(), 'Loss_P: ', self.loss_P.data[0]

    def optimize_generator(self):
        # forward_G
        self.forward_G()
        #Encoder
        self.optimizer_CE.zero_grad()
        self.optimizer_OPH.zero_grad()
        self.optimizer_OPL.zero_grad()
        self.optimizer_ME.zero_grad()
        self.optimizer_MP.zero_grad()
        # Decoder
        self.optimizer_G.zero_grad()
        

        if self.use_cycle:
            self.backward_cycle()
        else:
            self.backward_G()

        self.optimizer_CE.step()
        self.optimizer_OPH.step()
        self.optimizer_OPL.step()
        self.optimizer_ME.step()
        self.optimizer_MP.step()

        self.optimizer_G.step()
        
        # D

    def optimize_discriminator(self):
        #for p in netD.parameters(): # reset requires_grad
        #    p.requires_grad = True # they are set to False below in netG update
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def optimize_predictor(self):
        # forward_P
        self.forward_P()
        # Encoder
        self.optimizer_ME.zero_grad()
        # Predictor
        self.optimizer_MP.zero_grad()

        self.backward_P()

        self.optimizer_ME.step()
        self.optimizer_MP.step()

    def optimize_parameters(self):
    #    self.optimize_predictor()
        self.optimize_generator()
        self.optimize_discriminator()
        #self.optimizer_D.zero_grad()
        #self.backward_D()

    def get_current_errors(self):
        #print self.loss_D.data[0].cpu().float()
        #PRE = 0#self.loss_pre.data[0]
        GAN = self.loss_gan.data[0]
        #FEA = self.loss_fea.data[0]
        PIX = self.loss_pix.data[0]
        #DIF = self.loss_dif.data[0]
        DES = self.loss_D.data[0]
        SIM = 0 #self.loss_sim.data[0]
        return OrderedDict([('G', GAN), ('Pixel', PIX), ('D',DES),('SIM',SIM)])
        #return OrderedDict([('Prediction', PRE), ('G', GAN), ('Feature', FEA), ('Pixel', PIX),('Diff',DIF), ('D',DES)])

    def get_current_visuals(self):
        images = []
        for i in xrange(self.seq_len):
            name = 'frame_'+str(i)
            image = util.tensor2im(self.real_X[i].data)
            images += [(name,image)]
            if self.use_cycle:
                fake_name = 'fake_'+str(i)
                fake_image = util.tensor2im(self.fakes_x[i].data)
                images += [(fake_name,fake_image)]

        real_name = 'frame_gt'+str(self.seq_len+self.T)
        real_image = util.tensor2im(self.real_Y[self.T].data)
        fake_name = 'fake_pred'+str(self.seq_len+self.T)
        fake_image = util.tensor2im(self.fakes[self.T].data)
        images += [(real_name, real_image),(fake_name,fake_image)]

        '''for j in xrange(self.pre_len):
            real_name = 'frame_'+str(self.seq_len+j)
            real_image = util.tensor2im(self.real_Y[j].data)
            fake_name = 'fake_'+str(self.seq_len+j)
            fake_image = util.tensor2im(self.fakes[j].data)
            images += [(real_name, real_image),(fake_name,fake_image)]
        '''
        return OrderedDict(images)
        #return OrderedDict([('real_Began', real_Y_0), ('Pred_Began', fake_Y_0), ('real_End', real_Y_E),
        #                    ('Pred_End', fake_Y_E)])

    def save(self, label):
        self.save_network(self.netCE, 'CE', label, self.gpu_ids)
        self.save_network(self.netOPH, 'OPH', label, self.gpu_ids)
        self.save_network(self.netOPL, 'OPL', label, self.gpu_ids)
        self.save_network(self.netME, 'ME', label, self.gpu_ids)
        self.save_network(self.netMP, 'MP', label, self.gpu_ids)
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    '''def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_CE.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_SE.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_ME.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_MP.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr'''
