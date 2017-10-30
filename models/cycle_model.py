import numpy as np
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

import random

class CycleModel(BaseModel):
    def name(self):
        return 'CycleModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        seq_len = opt.seq_len
        self.seq_len = opt.seq_len
        self.pre_len = opt.pre_len
        shape = (opt.fineSize/128, opt.fineSize/128)

        self.netCE = networks.content_E_3D(opt.input_nc, 512,
                                                  opt.ngf, opt.which_model_E, opt.norm, not opt.no_dropout, self.gpu_ids)
        #self.netSE = networks.seq_E(shape,opt.ngf*8,opt.ngf*6, opt.norm, not opt.no_dropout, self.gpu_ids)

        #self.netP = networks.motion_P(shape, seq_len, opt.latent_nc,
        #                                    opt.latent_nc, opt.latent_nc*2, opt.norm, not opt.no_dropout, self.gpu_ids)
        self.netG = networks.define_G_3D(shape, seq_len, 512, opt.latent_nc,
                                            opt.output_nc, opt.ngf,'unet_128_3D_0', opt.norm, not opt.no_dropout, self.gpu_ids)


        #self.netME = networks.content_E(opt.input_nc, opt.latent_nc,
        #                                          opt.ngf/4, opt.which_model_E, opt.norm, not opt.no_dropout, self.gpu_ids)
        #self.netME = networks.motion_E(opt.input_nc, opt.ngf*4,
        #                                        opt.ngf, opt.which_model_E, opt.norm, not opt.no_dropout, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            feat_nc = 256
            #assert(opt.which_model_netD == 'Multi_D')
            #self.netD = networks.define_D(opt.fineSize, opt.output_nc,feat_nc, opt.ndf,
            #                            opt.which_model_netD,
            #                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netID = networks.define_D(opt.fineSize, opt.output_nc,feat_nc, opt.ndf,
                                        opt.which_model_netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)


        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netCE, 'CE', which_epoch)
            #self.load_network(self.netSE, 'SE', which_epoch)
            #self.load_network(self.netME, 'ME', which_epoch)
            #self.load_network(self.netP, 'P', which_epoch)
            self.load_network(self.netG, 'G', which_epoch)

            if self.isTrain:
                #self.load_network(self.netD, 'D', which_epoch)
                self.load_network(self.netID, 'ID', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionRec = torch.nn.L1Loss()
            self.criterionPre   = torch.nn.MSELoss()
            self.criterionFeat = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_CE = torch.optim.Adam(self.netCE.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            #self.optimizer_ME = torch.optim.Adam(self.netME.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            #self.optimizer_MP = torch.optim.Adam(self.netMP.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_ID = torch.optim.Adam(self.netID.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_CE)
            #self.optimizers.append(self.optimizer_ME)
            #self.optimizers.append(self.optimizer_MP)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_ID)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netCE)
        #networks.print_network(self.netSE)
        #networks.print_network(self.netME)
        #networks.print_network(self.netP)
        networks.print_network(self.netG)
        if self.isTrain:
            #networks.print_network(self.netD)
            networks.print_network(self.netID)
        print('-----------------------------------------------')

    def set_input(self,input):
        # X: batchsize, seq_len_x, inchains, size(0), size(1)
        # Y: batchsize, seq_len_y, inchains, size(0), size(1)
        input_X = input[0]
        input_Y = input[1]
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
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)

    # get image paths
    def get_image_paths(self):
        return 0#self.image_paths

    def backward_D_basic(self, netD, real, fake, real_feat):
        # Real

        pred_real = netD.forward(real,real_feat)
        #one = Variable(torch.FloatTensor(pred_real.size(0),1).fill_(1).cuda(),requires_grad=False)
        loss_D_real = self.criterionGAN(pred_real, True) #True

        # Fake_img
        random.shuffle(self.input_X)
        realX = Variable(torch.cat(self.input_X, 0))
        pred_fake_img = netD.forward(realX, real_feat)
        loss_D_fake_img = self.criterionGAN(pred_fake_img, False)
        # Fake_feat
        pred_fake = netD.forward(fake.detach(), real_feat)
        #zero = Variable(torch.FloatTensor(pred_fake.size(0),1).fill_(0).cuda(),requires_grad=False)
        loss_D_fake = self.criterionGAN(pred_fake, False) #False
        # Combined loss
        loss_D = loss_D_real + (loss_D_fake_img + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_ID_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True) #True
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) #False
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        #fake = self.fake_pool.query(self.fakes)
        fake = Variable(torch.cat(self.fakes,0).data)
        self.real = Variable(torch.cat(self.real_Y, 0).data)
        real_feat = Variable(torch.cat(self.content_codes,0).data)#self.netE.forward(real)
        self.loss_D = self.backward_D_basic(self.netD, self.real, fake, real_feat)

    def backward_ID(self):
        fake = self.fake_pool.query(self.fakes)
        self.realID = Variable(torch.cat(self.input_Y, 0))
        self.loss_ID = self.backward_ID_basic(self.netID, self.realID, fake)

    def backward_G(self):
        #lambda_fea = self.opt.lambda_fea
        lambda_pix = self.opt.lambda_pix
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

        self.real_X_INV = list(reversed(self.real_X))
        self.real_Y_INV = list(reversed(self.real_Y))

        real_X_e = [real.unsqueeze(2) for real in self.real_X] # [bs,nc,1,H,W]
        real_Y_e = [real.unsqueeze(2) for real in self.real_Y]
        self.realX = torch.cat(real_X_e, 2) # bs,nc,D,H,W
        self.realY = torch.cat(real_Y_e, 2)

        real_X_e_INV = [real.unsqueeze(2) for real in self.real_X_INV] # [bs,nc,1,H,W]
        real_Y_e_INV = [real.unsqueeze(2) for real in self.real_Y_INV]
        self.realX_INV = torch.cat(real_X_e_INV, 2) # bs,nc,D,H,W
        self.realY_INV = torch.cat(real_Y_e_INV, 2)

        ###################################################################################
        # X --> Y'
        #batch_size = self.real_X[0].size(0)
        enc_real_x = self.netCE.forward(self.realX)
        fake_y = self.netG.forward(enc_real_x,None)
        fakes_y = torch.split(fake_y,1,dim=2)
        self.fakes = [fake.squeeze(2) for fake in fakes_y] #[bs, nc, H,W]*len
        #fakesYb = torch.cat(self.fakes,0)

        self.loss_pix = self.criterionRec(self.fakes[0], self.real_Y[0])*lambda_pix
        self.loss_gan = self.criterionGAN(self.netID.forward(self.fakes[0]),True)*lambda_gan
        for t in range(1,self.pre_len):
            self.loss_pix += self.criterionRec(self.fakes[t], self.real_Y[t])*lambda_pix
            self.loss_gan += self.criterionGAN(self.netID.forward(self.fakes[t]),True)*lambda_gan
        ###################################################################################
        # Y'--> X''
        '''self.fakes.reverse()
        fake_Y_e = [fake.unsqueeze(2) for fake in self.fakes]
        self.fakeY = torch.cat(fake_Y_e, 2)
        enc_fake_y = self.netCE.forward(self.fakeY)
        fake_x = self.netG.forward(enc_fake_y,None)
        fakes_x = torch.split(fake_x,1,dim=2)
        self.fakesX = [fake.squeeze(2) for fake in fakes_x]
        #fakesXb = torch.cat(self.fakesX,0)
        #print fakesXb.size()

        self.loss_pix_cx = self.criterionRec(self.fakesX[0], self.real_X_INV[0])*lambda_pix
        self.loss_gan_cx = self.criterionGAN(self.netID.forward(self.fakesX[0]),True)*lambda_gan
        for t in range(1,self.pre_len):
            self.loss_pix_cx += self.criterionRec(self.fakesX[t], self.real_X_INV[t])*lambda_pix
            self.loss_gan_cx += self.criterionGAN(self.netID.forward(self.fakesX[t]),True)*lambda_gan

        self.fakes.reverse()
        '''

        '''self.content_codes = torch.split(self.netE.forward(self.realX),batch_size,dim=0) #[bs,1,oc,s0,s1]*len

        hidden_state_SE = self.netSE.init_hidden(batch_size)
        hidden_state_SE, self.content_code = self.netSE.forward(torch.cat(self.content_codes,1),hidden_state_SE) #[bs,oc,s0,s1]

        self.realY = torch.cat(self.real_Y, 0)
        self.motion_codes = torch.split(self.netME.forward(self.realY),batch_size,dim=0) #[bs,1,oc,s0,s1]*len

        #print self.content_code.size()
        #print self.motion_codes[0].size()
        latent_codes = []
        for l in xrange(self.pre_len):
            latent_codes += [torch.cat([self.content_code, self.motion_codes[l].squeeze(1)], 1)]
            #latent_codes += [self.motion_codes[l].squeeze(1)]
        self.latent_code = torch.cat(latent_codes,0)
        fakes = self.netG.forward(self.latent_code)
        self.fakes = torch.split(fakes,batch_size,dim=0)
        '''


        #self.loss_pix = self.criterionRec(fakes,self.realY)*lambda_pix
        #self.loss_gan = self.criterionGAN(self.netID.forward(fakes),True)*lambda_gan
        self.loss_G = self.loss_pix + self.loss_gan #+ self.loss_pix_cx + self.loss_gan_cx
        self.loss_G.backward()

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
        #self.optimizer_SE.zero_grad()
        #self.optimizer_ME.zero_grad()
        # Decoder
        self.optimizer_G.zero_grad()
        #self.optimizer_P.zero_grad()

        self.backward_G()

        self.optimizer_CE.step()
        #self.optimizer_SE.step()
        #self.optimizer_ME.step()
        self.optimizer_G.step()
        #self.optimizer_P.step()
        # D


    def optimize_discriminator(self):
        #for p in netD.parameters(): # reset requires_grad
        #    p.requires_grad = True # they are set to False below in netG update
        self.optimizer_ID.zero_grad()
        self.backward_ID()
        self.optimizer_ID.step()

    def optimize_predictor(self):
        # forward_P
        self.forward_P()
        # Encoder
        self.optimizer_E.zero_grad()
        # Predictor
        self.optimizer_P.zero_grad()

        self.backward_P()

        self.optimizer_E.step()
        self.optimizer_P.step()

    def optimize_parameters(self):
    #    self.optimize_predictor()
        self.optimize_generator()
        self.optimize_discriminator()
        #self.optimizer_D.zero_grad()
        #self.backward_D()

    def get_current_errors(self):
        #print self.loss_D.data[0].cpu().float()
        #PRE = self.loss_pre.data.cpu().float()[0]
        GAN = self.loss_gan.data[0]
        #FEA = self.loss_fea.data.cpu().float()[0]
        PIX = self.loss_pix.data[0]
        DES = self.loss_ID.data[0]
        return OrderedDict([('G', GAN), ('R', PIX), ('D',DES)])
        #return OrderedDict([('Prediction', PRE), ('G', GAN), ('Feature', FEA), ('Pixel', PIX), ('D',DES)])

    def get_current_visuals(self):
        images = []
        for i in xrange(self.seq_len):
            name = 'frame_'+str(i)
            image = util.tensor2im(self.real_X[i].data)
            images += [(name,image)]

        for j in xrange(self.pre_len):
            real_name = 'frame_'+str(self.seq_len+j)
            real_image = util.tensor2im(self.real_Y[j].data)
            fake_name = 'fake_'+str(self.seq_len+j)
            fake_image = util.tensor2im(self.fakes[j].data)
            images += [(real_name, real_image),(fake_name,fake_image)]

        return OrderedDict(images)
        #return OrderedDict([('real_Began', real_Y_0), ('Pred_Began', fake_Y_0), ('real_End', real_Y_E),
        #                    ('Pred_End', fake_Y_E)])

    def save(self, label):
        self.save_network(self.netCE, 'CE', label, self.gpu_ids)
        #self.save_network(self.netSE, 'SE', label, self.gpu_ids)
        #self.save_network(self.netME, 'ME', label, self.gpu_ids)
        #self.save_network(self.netP, 'P', label, self.gpu_ids)
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netID, 'ID', label, self.gpu_ids)
